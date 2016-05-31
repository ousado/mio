use {convert};
use token::Token;
use time::{self, precise_time_ns};
use std::{error, fmt, usize, iter};
use std::cmp::max;
use std::time::Duration;

use self::TimerErrorKind::TimerOverflow;

pub struct Timer<T> {
    // Size of each tick in milliseconds
    tick_ms: u64,
    // Slab of timeout entries
    entries: Slab<Entry<T>>,
    // Timeout wheel. Each tick, the timer will look at the next slot for
    // timeouts that match the current tick.
    wheel: Vec<Token>,
    // Tick 0's time in milliseconds
    start: u64,
    // The current tick
    tick: u64,
    // The next entry to possibly timeout
    next: Token,
    // Masks the target tick to get the slot
    mask: u64,
}

pub struct Builder {
    // Approximate duration of each tick
    tick: Duration,
    // Number of slots in the timer wheel
    num_slots: usize,
    // Max number of timeouts that can be in flight at a given time.
    capacity: usize,
}

// Doubly linked list of timer entries. Allows for efficient insertion /
// removal of timeouts.
struct Entry<T> {
    state: T,
    links: EntryLinks,
}

#[derive(Copy, Clone)]
struct EntryLinks {
    tick: u64,
    prev: Token,
    next: Token
}

pub type Result<T> = ::std::result::Result<T, TimerError>;

// TODO: Remove
pub type OldTimerResult<T> = Result<T>;
pub type TimerError = OldTimerError;
pub type Timeout = OldTimeout;

impl Builder {
    pub fn tick_duration(mut self, duration: Duration) -> Builder {
        self.tick = duration;
        self
    }

    pub fn num_slots(mut self, num_slots: usize) -> Builder {
        self.num_slots = num_slots;
        self
    }

    pub fn capacity(mut self, capacity: usize) -> Builder {
        self.capacity = capacity;
        self
    }

    pub fn build<T>(self) -> Timer<T> {
        let num_slots = self.num_slots.next_power_of_two();
        let capacity = self.capacity.next_power_of_two();
        let mask = (num_slots as u64) - 1;

        Timer {
            tick_ms: convert::millis(self.tick),
            entries: Slab::new(capacity),
            wheel: iter::repeat(EMPTY).take(num_slots).collect(),
            start: now_ms(),
            tick: 0,
            next: EMPTY,
            mask: mask,
        }
    }
}

impl Default for Builder {
    fn default() -> Builder {
        Builder {
            tick: Duration::from_millis(100),
            num_slots: 256,
            capacity: 65_536,
        }
    }
}

impl<T> Timer<T> {

    pub fn set_timeout(&mut self, delay: Duration, state: T) -> Result<Timeout> {
        let at = now_ms() + convert::millis(delay);
        self.set_timeout_at(at, state)
    }

    fn set_timeout_at(&mut self, mut delay: u64, state: T) -> Result<Timeout> {
        // Make relative to start
        delay -= self.start;

        // Calculate tick
        let mut tick = (delay + self.tick_ms - 1) / self.tick_ms;

        // Always target at least 1 tick in the future
        if tick <= self.tick {
            tick = self.tick + 1;
        }

        self.insert(tick, state)
    }

    fn insert(&mut self, tick: u64, state: T) -> Result<Timeout> {
        // Get the slot for the requested tick
        let slot = (tick & self.mask) as usize;
        let curr = self.wheel[slot];

        // Insert the new entry
        let token = try!(
            self.entries.insert(Entry::new(state, tick, curr))
            .map_err(|_| TimerError::overflow()));

        if curr != EMPTY {
            // If there was a previous entry, set its prev pointer to the new
            // entry
            self.entries[curr].links.prev = token;
        }

        // Update the head slot
        self.wheel[slot] = token;

        trace!("inserted timout; slot={}; token={:?}", slot, token);

        // Return the new timeout
        Ok(Timeout {
            token: token,
            tick: tick
        })
    }

    pub fn cancel_timeout(&mut self, timeout: &Timeout) -> Option<T> {
        let links = match self.entries.get(timeout.token) {
            Some(e) => e.links,
            None => return None
        };

        // Sanity check
        if links.tick != timeout.tick {
            return None;
        }

        self.unlink(&links, timeout.token);
        self.entries.remove(timeout.token).map(|e| e.state)
    }

    pub fn poll(&mut self) -> Option<T> {
        let target_tick = self.current_tick();
        self.poll_to(target_tick)
    }

    fn poll_to(&mut self, target_tick: u64) -> Option<T> {
        trace!("tick_to; target_tick={}; current_tick={}", target_tick, self.tick);

        while self.tick <= target_tick {
            let curr = self.next;

            trace!("ticking; curr={:?}", curr);

            if curr == EMPTY {
                self.tick += 1;
                self.next = self.wheel[self.slot_for(self.tick)];
            } else {
                let links = self.entries[curr].links;

                if links.tick <= self.tick {
                    trace!("triggering; token={:?}", curr);

                    // Unlink will also advance self.next
                    self.unlink(&links, curr);

                    // Remove and return the token
                    return self.entries.remove(curr)
                        .map(|e| e.state);
                } else {
                    self.next = links.next;
                }
            }
        }

        None
    }

    fn unlink(&mut self, links: &EntryLinks, token: Token) {
       trace!("unlinking timeout; slot={}; token={:?}",
               self.slot_for(links.tick), token);

        if links.prev == EMPTY {
            let slot = self.slot_for(links.tick);
            self.wheel[slot] = links.next;
        } else {
            self.entries[links.prev].links.next = links.next;
        }

        if links.next != EMPTY {
            self.entries[links.next].links.prev = links.prev;

            if token == self.next {
                self.next = links.next;
            }
        } else if token == self.next {
            self.next = EMPTY;
        }
    }

    pub fn current_tick(&self) -> u64 {
        self.ms_to_tick(now_ms())
    }

    // Convert a ms duration into a number of ticks, rounds up
    fn ms_to_tick(&self, ms: u64) -> u64 {
        (ms - self.start) / self.tick_ms
    }

    fn slot_for(&self, tick: u64) -> usize {
        (self.mask & tick) as usize
    }
}

impl<T> Default for Timer<T> {
    fn default() -> Timer<T> {
        Builder::default().build()
    }
}

#[inline]
fn now_ms() -> u64 {
    time::precise_time_ns() / NS_PER_MS
}

/*
 *
 * ===== Legacy =====
 *
 */

const EMPTY: Token = Token(usize::MAX);
const NS_PER_MS: u64 = 1_000_000;

// Implements coarse-grained timeouts using an algorithm based on hashed timing
// wheels by Varghese & Lauck.
//
// TODO:
// * Handle the case when the timer falls more than an entire wheel behind. There
//   is no point to loop multiple times around the wheel in one go.
// * New type for tick, now() -> Tick
#[derive(Debug)]
pub struct OldTimer<T> {
    // Size of each tick in milliseconds
    tick_ms: u64,
    // Slab of timeout entries
    entries: Slab<Entry<T>>,
    // Timeout wheel. Each tick, the timer will look at the next slot for
    // timeouts that match the current tick.
    wheel: Vec<Token>,
    // Tick 0's time in milliseconds
    start: u64,
    // The current tick
    tick: u64,
    // The next entry to possibly timeout
    next: Token,
    // Masks the target tick to get the slot
    mask: u64,
}

#[derive(Clone)]
pub struct OldTimeout {
    // Reference into the timer entry slab
    token: Token,
    // Tick that it should matchup with
    tick: u64,
}

type Slab<T> = ::slab::Slab<T, ::Token>;

impl<T> OldTimer<T> {
    pub fn new(tick_ms: u64, mut slots: usize, mut capacity: usize) -> OldTimer<T> {
        slots = slots.next_power_of_two();
        capacity = capacity.next_power_of_two();

        OldTimer {
            tick_ms: tick_ms,
            entries: Slab::new(capacity),
            wheel: iter::repeat(EMPTY).take(slots).collect(),
            start: 0,
            tick: 0,
            next: EMPTY,
            mask: (slots as u64) - 1
        }
    }

    #[cfg(test)]
    pub fn count(&self) -> usize {
        self.entries.count()
    }

    // Number of ms remaining until the next tick
    pub fn next_tick_in_ms(&self) -> Option<u64> {
        if self.entries.count() == 0 {
            return None;
        }

        let now = self.now_ms();
        let nxt = self.start + (self.tick + 1) * self.tick_ms;

        if nxt <= now {
            return Some(0);
        }

        Some(nxt - now)
    }

    /*
     *
     * ===== Initialization =====
     *
     */

    // Sets the starting time of the timer using the current system time
    pub fn setup(&mut self) {
        let now = self.now_ms();
        self.set_start_ms(now);
    }

    fn set_start_ms(&mut self, start: u64) {
        assert!(!self.is_initialized(), "the timer has already started");
        self.start = start;
    }

    /*
     *
     * ===== Timeout create / cancel =====
     *
     */

    pub fn timeout_ms(&mut self, token: T, delay: u64) -> OldTimerResult<OldTimeout> {
        let at = self.now_ms() + max(0, delay);
        self.timeout_at_ms(token, at)
    }

    pub fn timeout_at_ms(&mut self, token: T, mut at: u64) -> OldTimerResult<OldTimeout> {
        // Make relative to start
        at -= self.start;
        // Calculate tick
        let mut tick = (at + self.tick_ms - 1) / self.tick_ms;

        // Always target at least 1 tick in the future
        if tick <= self.tick {
            tick = self.tick + 1;
        }

        self.insert(token, tick)
    }

    pub fn clear(&mut self, timeout: &OldTimeout) -> bool {
        let links = match self.entries.get(timeout.token) {
            Some(e) => e.links,
            None => return false
        };

        // Sanity check
        if links.tick != timeout.tick {
            return false;
        }

        self.unlink(&links, timeout.token);
        self.entries.remove(timeout.token);
        true
    }

    fn insert(&mut self, token: T, tick: u64) -> OldTimerResult<OldTimeout> {
        // Get the slot for the requested tick
        let slot = (tick & self.mask) as usize;
        let curr = self.wheel[slot];

        // Insert the new entry
        let token = try!(
            self.entries.insert(Entry::new(token, tick, curr))
            .map_err(|_| OldTimerError::overflow()));

        if curr != EMPTY {
            // If there was a previous entry, set its prev pointer to the new
            // entry
            self.entries[curr].links.prev = token;
        }

        // Update the head slot
        self.wheel[slot] = token;

        trace!("inserted timout; slot={}; token={:?}", slot, token);

        // Return the new timeout
        Ok(OldTimeout {
            token: token,
            tick: tick
        })
    }

    fn unlink(&mut self, links: &EntryLinks, token: Token) {
       trace!("unlinking timeout; slot={}; token={:?}",
               self.slot_for(links.tick), token);

        if links.prev == EMPTY {
            let slot = self.slot_for(links.tick);
            self.wheel[slot] = links.next;
        } else {
            self.entries[links.prev].links.next = links.next;
        }

        if links.next != EMPTY {
            self.entries[links.next].links.prev = links.prev;

            if token == self.next {
                self.next = links.next;
            }
        } else if token == self.next {
            self.next = EMPTY;
        }
    }

    /*
     *
     * ===== Advance time =====
     *
     */

    pub fn now(&self) -> u64 {
        self.ms_to_tick(self.now_ms())
    }

    pub fn tick_to(&mut self, now: u64) -> Option<T> {
        trace!("tick_to; now={}; tick={}", now, self.tick);

        while self.tick <= now {
            let curr = self.next;

            trace!("ticking; curr={:?}", curr);

            if curr == EMPTY {
                self.tick += 1;
                self.next = self.wheel[self.slot_for(self.tick)];
            } else {
                let links = self.entries[curr].links;

                if links.tick <= self.tick {
                    trace!("triggering; token={:?}", curr);

                    // Unlink will also advance self.next
                    self.unlink(&links, curr);

                    // Remove and return the token
                    return self.entries.remove(curr)
                        .map(|e| e.state);
                } else {
                    self.next = links.next;
                }
            }
        }

        None
    }

    /*
     *
     * ===== Misc =====
     *
     */

    // Timers are initialized when either the current time has been advanced or a timeout has been set
    #[inline]
    fn is_initialized(&self) -> bool {
        self.tick > 0 || !self.entries.is_empty()
    }

    #[inline]
    fn slot_for(&self, tick: u64) -> usize {
        (self.mask & tick) as usize
    }

    // Convert a ms duration into a number of ticks, rounds up
    #[inline]
    fn ms_to_tick(&self, ms: u64) -> u64 {
        (ms - self.start) / self.tick_ms
    }

    #[inline]
    fn now_ms(&self) -> u64 {
        precise_time_ns() / NS_PER_MS
    }
}

impl<T> Entry<T> {
    fn new(state: T, tick: u64, next: Token) -> Entry<T> {
        Entry {
            state: state,
            links: EntryLinks {
                tick: tick,
                prev: EMPTY,
                next: next,
            },
        }
    }
}

#[derive(Debug)]
pub struct OldTimerError {
    kind: TimerErrorKind,
    desc: &'static str,
}

impl fmt::Display for OldTimerError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}: {}", self.kind, self.desc)
    }
}

impl OldTimerError {
    fn overflow() -> OldTimerError {
        OldTimerError {
            kind: TimerOverflow,
            desc: "too many timer entries"
        }
    }
}

impl error::Error for OldTimerError {
    fn description(&self) -> &str {
        self.desc
    }
}

#[derive(Debug)]
pub enum TimerErrorKind {
    TimerOverflow,
}

impl fmt::Display for TimerErrorKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TimerOverflow => write!(fmt, "TimerOverflow"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::iter;

    #[test]
    pub fn test_timeout_next_tick() {
        let mut t = timer();
        let mut tick;

        t.set_timeout_at(100, "a").unwrap();

        tick = t.ms_to_tick(50);
        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(100);
        assert_eq!(Some("a"), t.poll_to(tick));
        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(150);
        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(200);
        assert_eq!(None, t.poll_to(tick));

        assert_eq!(count(&t), 0);
    }

    #[test]
    pub fn test_clearing_timeout() {
        let mut t = timer();
        let mut tick;

        let to = t.set_timeout_at(100, "a").unwrap();
        assert_eq!("a", t.cancel_timeout(&to).unwrap());

        tick = t.ms_to_tick(100);
        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(200);
        assert_eq!(None, t.poll_to(tick));

        assert_eq!(count(&t), 0);
    }

    #[test]
    pub fn test_multiple_timeouts_same_tick() {
        let mut t = timer();
        let mut tick;

        t.set_timeout_at(100, "a").unwrap();
        t.set_timeout_at(100, "b").unwrap();

        let mut rcv = vec![];

        tick = t.ms_to_tick(100);
        rcv.push(t.poll_to(tick).unwrap());
        rcv.push(t.poll_to(tick).unwrap());

        assert_eq!(None, t.poll_to(tick));

        rcv.sort();
        assert!(rcv == ["a", "b"], "actual={:?}", rcv);

        tick = t.ms_to_tick(200);
        assert_eq!(None, t.poll_to(tick));

        assert_eq!(count(&t), 0);
    }

    #[test]
    pub fn test_multiple_timeouts_diff_tick() {
        let mut t = timer();
        let mut tick;

        t.set_timeout_at(110, "a").unwrap();
        t.set_timeout_at(220, "b").unwrap();
        t.set_timeout_at(230, "c").unwrap();
        t.set_timeout_at(440, "d").unwrap();

        tick = t.ms_to_tick(100);
        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(200);
        assert_eq!(Some("a"), t.poll_to(tick));
        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(300);
        assert_eq!(Some("c"), t.poll_to(tick));
        assert_eq!(Some("b"), t.poll_to(tick));
        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(400);
        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(500);
        assert_eq!(Some("d"), t.poll_to(tick));
        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(600);
        assert_eq!(None, t.poll_to(tick));
    }

    #[test]
    pub fn test_catching_up() {
        let mut t = timer();

        t.set_timeout_at(110, "a").unwrap();
        t.set_timeout_at(220, "b").unwrap();
        t.set_timeout_at(230, "c").unwrap();
        t.set_timeout_at(440, "d").unwrap();

        let tick = t.ms_to_tick(600);
        assert_eq!(Some("a"), t.poll_to(tick));
        assert_eq!(Some("c"), t.poll_to(tick));
        assert_eq!(Some("b"), t.poll_to(tick));
        assert_eq!(Some("d"), t.poll_to(tick));
        assert_eq!(None, t.poll_to(tick));
    }

    #[test]
    pub fn test_timeout_hash_collision() {
        let mut t = timer();
        let mut tick;

        t.set_timeout_at(100, "a").unwrap();
        t.set_timeout_at(100 + TICK * SLOTS as u64, "b").unwrap();

        tick = t.ms_to_tick(100);
        assert_eq!(Some("a"), t.poll_to(tick));
        assert_eq!(1, count(&t));

        tick = t.ms_to_tick(200);
        assert_eq!(None, t.poll_to(tick));
        assert_eq!(1, count(&t));

        tick = t.ms_to_tick(100 + TICK * SLOTS as u64);
        assert_eq!(Some("b"), t.poll_to(tick));
        assert_eq!(0, count(&t));
    }

    #[test]
    pub fn test_clearing_timeout_between_triggers() {
        let mut t = timer();
        let mut tick;

        let a = t.set_timeout_at(100, "a").unwrap();
        let _ = t.set_timeout_at(100, "b").unwrap();
        let _ = t.set_timeout_at(200, "c").unwrap();

        tick = t.ms_to_tick(100);
        assert_eq!(Some("b"), t.poll_to(tick));
        assert_eq!(2, count(&t));

        t.cancel_timeout(&a);
        assert_eq!(1, count(&t));

        assert_eq!(None, t.poll_to(tick));

        tick = t.ms_to_tick(200);
        assert_eq!(Some("c"), t.poll_to(tick));
        assert_eq!(0, count(&t));
    }

    const TICK: u64 = 100;
    const SLOTS: usize = 16;
    const CAPACITY: usize = 32;

    fn count<T>(timer: &Timer<T>) -> usize {
        timer.entries.count()
    }

    fn timer() -> Timer<&'static str> {
        Timer {
            tick_ms: TICK,
            entries: super::Slab::new(CAPACITY),
            wheel: iter::repeat(super::EMPTY).take(SLOTS).collect(),
            start: 0,
            tick: 0,
            next: super::EMPTY,
            mask: (SLOTS as u64) - 1,
        }
    }
}
