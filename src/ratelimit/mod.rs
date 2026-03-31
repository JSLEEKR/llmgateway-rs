//! Rate limiting module — token bucket and sliding window algorithms.
//!
//! Supports per-key rate limiting with configurable capacity and refill rates.
//! Also includes budget tracking for daily spend limits.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Result of a rate limit check.
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitResult {
    /// Request is allowed. Contains remaining capacity.
    Allowed { remaining: u64 },
    /// Request is denied. Contains retry-after duration.
    Denied { retry_after: Duration },
}

impl RateLimitResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, RateLimitResult::Allowed { .. })
    }
}

// ---------------------------------------------------------------------------
// Token Bucket
// ---------------------------------------------------------------------------

/// A single token bucket instance.
#[derive(Debug, Clone)]
struct Bucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl Bucket {
    fn new(capacity: u64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity as f64,
            capacity: capacity as f64,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    /// Try to consume `count` tokens. Returns the result.
    fn try_consume(&mut self, count: u64) -> RateLimitResult {
        self.refill();
        let count_f = count as f64;
        if self.tokens >= count_f {
            self.tokens -= count_f;
            RateLimitResult::Allowed {
                remaining: self.tokens as u64,
            }
        } else {
            let deficit = count_f - self.tokens;
            let wait_secs = deficit / self.refill_rate;
            RateLimitResult::Denied {
                retry_after: Duration::from_secs_f64(wait_secs),
            }
        }
    }

    /// Get remaining tokens without consuming.
    fn remaining(&mut self) -> u64 {
        self.refill();
        self.tokens as u64
    }
}

/// Token bucket rate limiter — per-key buckets with configurable capacity.
pub struct TokenBucketLimiter {
    buckets: Arc<RwLock<HashMap<String, Bucket>>>,
    capacity: u64,
    refill_rate: f64,
}

impl TokenBucketLimiter {
    /// Create a new token bucket limiter.
    ///
    /// - `capacity`: Maximum tokens per bucket.
    /// - `refill_per_minute`: How many tokens to add per minute.
    pub fn new(capacity: u64, refill_per_minute: u64) -> Self {
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            capacity,
            refill_rate: refill_per_minute as f64 / 60.0,
        }
    }

    /// Check and consume tokens for a given key.
    pub fn check(&self, key: &str, tokens: u64) -> RateLimitResult {
        let mut buckets = match self.buckets.write() {
            Ok(b) => b,
            Err(_) => {
                return RateLimitResult::Allowed { remaining: 0 };
            }
        };
        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| Bucket::new(self.capacity, self.refill_rate));
        bucket.try_consume(tokens)
    }

    /// Get remaining tokens for a key without consuming.
    pub fn remaining(&self, key: &str) -> u64 {
        let mut buckets = match self.buckets.write() {
            Ok(b) => b,
            Err(_) => return 0,
        };
        if let Some(bucket) = buckets.get_mut(key) {
            bucket.remaining()
        } else {
            self.capacity
        }
    }

    /// Reset a specific key's bucket.
    pub fn reset(&self, key: &str) {
        if let Ok(mut buckets) = self.buckets.write() {
            buckets.remove(key);
        }
    }

    /// Reset all buckets.
    pub fn reset_all(&self) {
        if let Ok(mut buckets) = self.buckets.write() {
            buckets.clear();
        }
    }

    /// Get the number of tracked keys.
    pub fn key_count(&self) -> usize {
        self.buckets.read().map(|b| b.len()).unwrap_or(0)
    }
}

impl Clone for TokenBucketLimiter {
    fn clone(&self) -> Self {
        Self {
            buckets: Arc::clone(&self.buckets),
            capacity: self.capacity,
            refill_rate: self.refill_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding Window Counter
// ---------------------------------------------------------------------------

/// A sliding window counter entry.
#[derive(Debug, Clone)]
struct WindowEntry {
    /// Count in the previous window.
    prev_count: u64,
    /// Count in the current window.
    curr_count: u64,
    /// When the current window started.
    window_start: Instant,
    /// Window duration.
    window_size: Duration,
}

impl WindowEntry {
    fn new(window_size: Duration) -> Self {
        Self {
            prev_count: 0,
            curr_count: 0,
            window_start: Instant::now(),
            window_size,
        }
    }

    /// Slide the window forward if needed.
    fn maybe_slide(&mut self) {
        let elapsed = self.window_start.elapsed();
        if elapsed >= self.window_size * 2 {
            // More than 2 windows passed, reset everything
            self.prev_count = 0;
            self.curr_count = 0;
            self.window_start = Instant::now();
        } else if elapsed >= self.window_size {
            // Slide: current becomes previous
            self.prev_count = self.curr_count;
            self.curr_count = 0;
            self.window_start = Instant::now();
        }
    }

    /// Get the estimated count using weighted sliding window.
    fn estimated_count(&mut self) -> f64 {
        self.maybe_slide();
        let elapsed_ratio = self.window_start.elapsed().as_secs_f64()
            / self.window_size.as_secs_f64();
        let prev_weight = 1.0 - elapsed_ratio.min(1.0);
        self.prev_count as f64 * prev_weight + self.curr_count as f64
    }

    /// Increment and check against limit.
    fn try_increment(&mut self, count: u64, limit: u64) -> RateLimitResult {
        self.maybe_slide();
        let elapsed_ratio = self.window_start.elapsed().as_secs_f64()
            / self.window_size.as_secs_f64();
        let prev_weight = 1.0 - elapsed_ratio.min(1.0);
        let estimated = self.prev_count as f64 * prev_weight + self.curr_count as f64 + count as f64;

        if estimated <= limit as f64 {
            self.curr_count += count;
            let remaining = (limit as f64 - estimated).max(0.0) as u64;
            RateLimitResult::Allowed { remaining }
        } else {
            let remaining_window = self.window_size.as_secs_f64() * (1.0 - elapsed_ratio);
            RateLimitResult::Denied {
                retry_after: Duration::from_secs_f64(remaining_window.max(0.1)),
            }
        }
    }
}

/// Sliding window rate limiter — per-key with configurable window and limit.
pub struct SlidingWindowLimiter {
    entries: Arc<RwLock<HashMap<String, WindowEntry>>>,
    limit: u64,
    window_size: Duration,
}

impl SlidingWindowLimiter {
    /// Create a new sliding window limiter.
    ///
    /// - `limit`: Maximum count per window.
    /// - `window_size`: Duration of the window.
    pub fn new(limit: u64, window_size: Duration) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            limit,
            window_size,
        }
    }

    /// Check and increment count for a given key.
    pub fn check(&self, key: &str, count: u64) -> RateLimitResult {
        let mut entries = match self.entries.write() {
            Ok(e) => e,
            Err(_) => return RateLimitResult::Allowed { remaining: 0 },
        };
        let entry = entries
            .entry(key.to_string())
            .or_insert_with(|| WindowEntry::new(self.window_size));
        entry.try_increment(count, self.limit)
    }

    /// Get the estimated current count for a key.
    pub fn current_count(&self, key: &str) -> f64 {
        let mut entries = match self.entries.write() {
            Ok(e) => e,
            Err(_) => return 0.0,
        };
        if let Some(entry) = entries.get_mut(key) {
            entry.estimated_count()
        } else {
            0.0
        }
    }

    /// Reset a specific key.
    pub fn reset(&self, key: &str) {
        if let Ok(mut entries) = self.entries.write() {
            entries.remove(key);
        }
    }
}

impl Clone for SlidingWindowLimiter {
    fn clone(&self) -> Self {
        Self {
            entries: Arc::clone(&self.entries),
            limit: self.limit,
            window_size: self.window_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Budget Tracker
// ---------------------------------------------------------------------------

/// Daily spend budget tracker.
pub struct BudgetTracker {
    /// Per-key spend tracking: key -> (total_spend, last_reset).
    spend: Arc<RwLock<HashMap<String, (f64, Instant)>>>,
    /// Daily budget limit in dollars (0 = unlimited).
    daily_limit: f64,
    /// Reset interval (default: 24 hours).
    reset_interval: Duration,
}

impl BudgetTracker {
    /// Create a new budget tracker with a daily limit.
    pub fn new(daily_limit: f64) -> Self {
        Self {
            spend: Arc::new(RwLock::new(HashMap::new())),
            daily_limit,
            reset_interval: Duration::from_secs(86400),
        }
    }

    /// Create a budget tracker with a custom reset interval (for testing).
    pub fn with_reset_interval(daily_limit: f64, reset_interval: Duration) -> Self {
        Self {
            spend: Arc::new(RwLock::new(HashMap::new())),
            daily_limit,
            reset_interval,
        }
    }

    /// Check if spending `amount` would exceed the budget.
    pub fn check(&self, key: &str, amount: f64) -> RateLimitResult {
        if self.daily_limit <= 0.0 {
            return RateLimitResult::Allowed { remaining: u64::MAX };
        }

        let mut spend = match self.spend.write() {
            Ok(s) => s,
            Err(_) => return RateLimitResult::Allowed { remaining: 0 },
        };

        let (current, last_reset) = spend
            .entry(key.to_string())
            .or_insert((0.0, Instant::now()));

        // Reset if interval has passed
        if last_reset.elapsed() > self.reset_interval {
            *current = 0.0;
            *last_reset = Instant::now();
        }

        if *current + amount <= self.daily_limit {
            *current += amount;
            let remaining_dollars = self.daily_limit - *current;
            RateLimitResult::Allowed {
                remaining: (remaining_dollars * 100.0) as u64, // cents
            }
        } else {
            let time_until_reset = self.reset_interval
                .checked_sub(last_reset.elapsed())
                .unwrap_or(Duration::from_secs(0));
            RateLimitResult::Denied {
                retry_after: time_until_reset,
            }
        }
    }

    /// Get current spend for a key.
    pub fn current_spend(&self, key: &str) -> f64 {
        self.spend
            .read()
            .ok()
            .and_then(|s| s.get(key).map(|(amount, _)| *amount))
            .unwrap_or(0.0)
    }

    /// Get the daily limit.
    pub fn limit(&self) -> f64 {
        self.daily_limit
    }

    /// Reset spend for a specific key.
    pub fn reset(&self, key: &str) {
        if let Ok(mut spend) = self.spend.write() {
            spend.remove(key);
        }
    }
}

impl Clone for BudgetTracker {
    fn clone(&self) -> Self {
        Self {
            spend: Arc::clone(&self.spend),
            daily_limit: self.daily_limit,
            reset_interval: self.reset_interval,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Token Bucket Tests ---

    #[test]
    fn test_token_bucket_allows_within_capacity() {
        let limiter = TokenBucketLimiter::new(10, 600); // 10 capacity, 10/sec refill
        let result = limiter.check("user1", 5);
        assert!(result.is_allowed());
        if let RateLimitResult::Allowed { remaining } = result {
            assert_eq!(remaining, 5);
        }
    }

    #[test]
    fn test_token_bucket_denies_over_capacity() {
        let limiter = TokenBucketLimiter::new(5, 60);
        let result = limiter.check("user1", 10);
        assert!(!result.is_allowed());
    }

    #[test]
    fn test_token_bucket_depletes() {
        let limiter = TokenBucketLimiter::new(3, 60);
        assert!(limiter.check("user1", 1).is_allowed());
        assert!(limiter.check("user1", 1).is_allowed());
        assert!(limiter.check("user1", 1).is_allowed());
        assert!(!limiter.check("user1", 1).is_allowed());
    }

    #[test]
    fn test_token_bucket_refills() {
        let limiter = TokenBucketLimiter::new(1, 6000); // 100/sec refill
        assert!(limiter.check("user1", 1).is_allowed());
        assert!(!limiter.check("user1", 1).is_allowed());
        std::thread::sleep(Duration::from_millis(20));
        assert!(limiter.check("user1", 1).is_allowed());
    }

    #[test]
    fn test_token_bucket_per_key_isolation() {
        let limiter = TokenBucketLimiter::new(2, 60);
        assert!(limiter.check("user1", 2).is_allowed());
        assert!(limiter.check("user2", 2).is_allowed()); // different key, full bucket
    }

    #[test]
    fn test_token_bucket_remaining() {
        let limiter = TokenBucketLimiter::new(10, 60);
        assert_eq!(limiter.remaining("user1"), 10);
        limiter.check("user1", 3);
        assert_eq!(limiter.remaining("user1"), 7);
    }

    #[test]
    fn test_token_bucket_reset() {
        let limiter = TokenBucketLimiter::new(5, 60);
        limiter.check("user1", 5);
        assert!(!limiter.check("user1", 1).is_allowed());
        limiter.reset("user1");
        assert!(limiter.check("user1", 5).is_allowed());
    }

    #[test]
    fn test_token_bucket_reset_all() {
        let limiter = TokenBucketLimiter::new(1, 60);
        limiter.check("a", 1);
        limiter.check("b", 1);
        assert_eq!(limiter.key_count(), 2);
        limiter.reset_all();
        assert_eq!(limiter.key_count(), 0);
    }

    #[test]
    fn test_token_bucket_retry_after() {
        let limiter = TokenBucketLimiter::new(0, 60); // 0 capacity
        let result = limiter.check("user1", 1);
        if let RateLimitResult::Denied { retry_after } = result {
            assert!(retry_after > Duration::from_secs(0));
        } else {
            panic!("expected denied");
        }
    }

    #[test]
    fn test_token_bucket_clone_shares_state() {
        let limiter1 = TokenBucketLimiter::new(5, 60);
        let limiter2 = limiter1.clone();
        limiter1.check("user1", 5);
        assert!(!limiter2.check("user1", 1).is_allowed());
    }

    // --- Sliding Window Tests ---

    #[test]
    fn test_sliding_window_allows_within_limit() {
        let limiter = SlidingWindowLimiter::new(10, Duration::from_secs(60));
        let result = limiter.check("user1", 5);
        assert!(result.is_allowed());
    }

    #[test]
    fn test_sliding_window_denies_over_limit() {
        let limiter = SlidingWindowLimiter::new(5, Duration::from_secs(60));
        assert!(limiter.check("user1", 5).is_allowed());
        assert!(!limiter.check("user1", 1).is_allowed());
    }

    #[test]
    fn test_sliding_window_per_key() {
        let limiter = SlidingWindowLimiter::new(5, Duration::from_secs(60));
        assert!(limiter.check("user1", 5).is_allowed());
        assert!(limiter.check("user2", 5).is_allowed());
    }

    #[test]
    fn test_sliding_window_current_count() {
        let limiter = SlidingWindowLimiter::new(100, Duration::from_secs(60));
        limiter.check("user1", 7);
        let count = limiter.current_count("user1");
        assert!((count - 7.0).abs() < 1.0);
    }

    #[test]
    fn test_sliding_window_reset() {
        let limiter = SlidingWindowLimiter::new(1, Duration::from_secs(60));
        limiter.check("user1", 1);
        assert!(!limiter.check("user1", 1).is_allowed());
        limiter.reset("user1");
        assert!(limiter.check("user1", 1).is_allowed());
    }

    #[test]
    fn test_sliding_window_unknown_key_zero() {
        let limiter = SlidingWindowLimiter::new(10, Duration::from_secs(60));
        assert_eq!(limiter.current_count("unknown"), 0.0);
    }

    // --- Budget Tracker Tests ---

    #[test]
    fn test_budget_allows_within_limit() {
        let tracker = BudgetTracker::new(10.0);
        let result = tracker.check("user1", 5.0);
        assert!(result.is_allowed());
    }

    #[test]
    fn test_budget_denies_over_limit() {
        let tracker = BudgetTracker::new(10.0);
        tracker.check("user1", 8.0);
        let result = tracker.check("user1", 5.0);
        assert!(!result.is_allowed());
    }

    #[test]
    fn test_budget_accumulates() {
        let tracker = BudgetTracker::new(10.0);
        tracker.check("user1", 3.0);
        tracker.check("user1", 4.0);
        assert!((tracker.current_spend("user1") - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_budget_per_key() {
        let tracker = BudgetTracker::new(10.0);
        tracker.check("user1", 8.0);
        assert!(tracker.check("user2", 8.0).is_allowed());
    }

    #[test]
    fn test_budget_unlimited_when_zero() {
        let tracker = BudgetTracker::new(0.0);
        assert!(tracker.check("user1", 1000.0).is_allowed());
    }

    #[test]
    fn test_budget_reset() {
        let tracker = BudgetTracker::new(5.0);
        tracker.check("user1", 5.0);
        assert!(!tracker.check("user1", 1.0).is_allowed());
        tracker.reset("user1");
        assert!(tracker.check("user1", 5.0).is_allowed());
    }

    #[test]
    fn test_budget_resets_after_interval() {
        let tracker = BudgetTracker::with_reset_interval(5.0, Duration::from_millis(10));
        tracker.check("user1", 5.0);
        assert!(!tracker.check("user1", 1.0).is_allowed());
        std::thread::sleep(Duration::from_millis(15));
        assert!(tracker.check("user1", 5.0).is_allowed());
    }

    #[test]
    fn test_budget_limit() {
        let tracker = BudgetTracker::new(42.0);
        assert_eq!(tracker.limit(), 42.0);
    }

    #[test]
    fn test_budget_current_spend_unknown_key() {
        let tracker = BudgetTracker::new(10.0);
        assert_eq!(tracker.current_spend("unknown"), 0.0);
    }

    #[test]
    fn test_budget_clone_shares_state() {
        let tracker1 = BudgetTracker::new(10.0);
        let tracker2 = tracker1.clone();
        tracker1.check("user1", 8.0);
        assert!(!tracker2.check("user1", 5.0).is_allowed());
    }
}
