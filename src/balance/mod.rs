//! Load balancing module — route requests across multiple upstream providers.
//!
//! Implements round-robin, weighted, P2C (power-of-two-choices),
//! and least-connections strategies. Includes provider health tracking.

use rand::Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Health state of an upstream provider.
#[derive(Debug, Clone)]
pub struct ProviderHealth {
    /// Provider identifier (name).
    pub name: String,
    /// Total number of requests sent.
    pub total_requests: u64,
    /// Number of failed requests.
    pub failed_requests: u64,
    /// Number of currently in-flight requests.
    pub in_flight: u64,
    /// Exponentially weighted moving average of latency in milliseconds.
    pub latency_ewma_ms: f64,
    /// Whether the provider is considered healthy.
    pub healthy: bool,
    /// When the last successful response was received.
    pub last_success: Option<Instant>,
    /// When the last error occurred.
    pub last_error: Option<Instant>,
    /// Weight for weighted balancing (0-100).
    pub weight: u32,
}

impl ProviderHealth {
    pub fn new(name: &str, weight: u32) -> Self {
        Self {
            name: name.to_string(),
            total_requests: 0,
            failed_requests: 0,
            in_flight: 0,
            latency_ewma_ms: 0.0,
            healthy: true,
            last_success: None,
            last_error: None,
            weight,
        }
    }

    /// Error rate as a fraction (0.0 - 1.0).
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.failed_requests as f64 / self.total_requests as f64
    }

    /// Record a successful request with latency.
    pub fn record_success(&mut self, latency: Duration) {
        self.total_requests += 1;
        self.in_flight = self.in_flight.saturating_sub(1);
        self.last_success = Some(Instant::now());

        // EWMA with alpha = 0.3 (recent observations weighted more)
        let latency_ms = latency.as_secs_f64() * 1000.0;
        if self.latency_ewma_ms == 0.0 {
            self.latency_ewma_ms = latency_ms;
        } else {
            self.latency_ewma_ms = 0.3 * latency_ms + 0.7 * self.latency_ewma_ms;
        }

        // Mark healthy if error rate drops below threshold
        if self.error_rate() < 0.5 {
            self.healthy = true;
        }
    }

    /// Record a failed request.
    pub fn record_failure(&mut self) {
        self.total_requests += 1;
        self.failed_requests += 1;
        self.in_flight = self.in_flight.saturating_sub(1);
        self.last_error = Some(Instant::now());

        // Mark unhealthy if error rate exceeds threshold
        if self.error_rate() > 0.5 && self.total_requests >= 3 {
            self.healthy = false;
        }
    }

    /// Record that a request is being started (in-flight count).
    pub fn record_start(&mut self) {
        self.in_flight += 1;
    }
}

/// Load balancing strategy trait.
pub trait BalanceStrategy: Send + Sync {
    /// Select the next provider index from the available list.
    /// Returns `None` if no provider is available.
    fn select(&self, providers: &[ProviderHealth]) -> Option<usize>;

    /// Name of the strategy for logging.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Round Robin
// ---------------------------------------------------------------------------

/// Round-robin balancer — cycles through providers sequentially.
pub struct RoundRobin {
    counter: AtomicU64,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl BalanceStrategy for RoundRobin {
    fn select(&self, providers: &[ProviderHealth]) -> Option<usize> {
        let healthy: Vec<usize> = providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.healthy)
            .map(|(i, _)| i)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let idx = self.counter.fetch_add(1, Ordering::Relaxed) as usize;
        Some(healthy[idx % healthy.len()])
    }

    fn name(&self) -> &str {
        "round_robin"
    }
}

// ---------------------------------------------------------------------------
// Weighted
// ---------------------------------------------------------------------------

/// Weighted balancer — selects providers based on their configured weights.
pub struct Weighted;

impl Weighted {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Weighted {
    fn default() -> Self {
        Self::new()
    }
}

impl BalanceStrategy for Weighted {
    fn select(&self, providers: &[ProviderHealth]) -> Option<usize> {
        let healthy: Vec<(usize, u32)> = providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.healthy && p.weight > 0)
            .map(|(i, p)| (i, p.weight))
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let total_weight: u32 = healthy.iter().map(|(_, w)| w).sum();
        if total_weight == 0 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut target = rng.gen_range(0..total_weight);

        for (idx, weight) in &healthy {
            if target < *weight {
                return Some(*idx);
            }
            target -= weight;
        }

        Some(healthy.last().unwrap().0)
    }

    fn name(&self) -> &str {
        "weighted"
    }
}

// ---------------------------------------------------------------------------
// P2C (Power of Two Choices)
// ---------------------------------------------------------------------------

/// P2C balancer — randomly pick 2 backends, choose the one with fewer in-flight.
pub struct P2C;

impl P2C {
    pub fn new() -> Self {
        Self
    }
}

impl Default for P2C {
    fn default() -> Self {
        Self::new()
    }
}

impl BalanceStrategy for P2C {
    fn select(&self, providers: &[ProviderHealth]) -> Option<usize> {
        let healthy: Vec<usize> = providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.healthy)
            .map(|(i, _)| i)
            .collect();

        match healthy.len() {
            0 => None,
            1 => Some(healthy[0]),
            _ => {
                let mut rng = rand::thread_rng();
                let a = healthy[rng.gen_range(0..healthy.len())];
                let mut b = a;
                // Ensure b != a
                while b == a {
                    b = healthy[rng.gen_range(0..healthy.len())];
                }

                // Choose the one with fewer in-flight requests
                let pa = &providers[a];
                let pb = &providers[b];

                if pa.in_flight < pb.in_flight {
                    Some(a)
                } else if pb.in_flight < pa.in_flight {
                    Some(b)
                } else {
                    // Tie-break on latency
                    if pa.latency_ewma_ms <= pb.latency_ewma_ms {
                        Some(a)
                    } else {
                        Some(b)
                    }
                }
            }
        }
    }

    fn name(&self) -> &str {
        "p2c"
    }
}

// ---------------------------------------------------------------------------
// Least Connections
// ---------------------------------------------------------------------------

/// Least connections balancer — choose the provider with fewest in-flight requests.
pub struct LeastConnections;

impl LeastConnections {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LeastConnections {
    fn default() -> Self {
        Self::new()
    }
}

impl BalanceStrategy for LeastConnections {
    fn select(&self, providers: &[ProviderHealth]) -> Option<usize> {
        providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.healthy)
            .min_by_key(|(_, p)| p.in_flight)
            .map(|(i, _)| i)
    }

    fn name(&self) -> &str {
        "least_connections"
    }
}

// ---------------------------------------------------------------------------
// Load Balancer (composite)
// ---------------------------------------------------------------------------

/// The main load balancer that wraps a strategy and health state.
pub struct LoadBalancer {
    providers: Arc<RwLock<Vec<ProviderHealth>>>,
    strategy: Box<dyn BalanceStrategy>,
    /// Provider name -> index mapping.
    index_map: Arc<RwLock<HashMap<String, usize>>>,
}

impl LoadBalancer {
    /// Create a new load balancer with the given strategy and providers.
    pub fn new(strategy: Box<dyn BalanceStrategy>, provider_configs: Vec<(&str, u32)>) -> Self {
        let providers: Vec<ProviderHealth> = provider_configs
            .iter()
            .map(|(name, weight)| ProviderHealth::new(name, *weight))
            .collect();

        let index_map: HashMap<String, usize> = providers
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.clone(), i))
            .collect();

        Self {
            providers: Arc::new(RwLock::new(providers)),
            strategy,
            index_map: Arc::new(RwLock::new(index_map)),
        }
    }

    /// Select the next provider to route to.
    pub fn select(&self) -> Option<String> {
        let providers = self.providers.read().ok()?;
        let idx = self.strategy.select(&providers)?;
        Some(providers[idx].name.clone())
    }

    /// Record that a request is starting for a provider.
    pub fn record_start(&self, provider_name: &str) {
        if let (Ok(index_map), Ok(mut providers)) =
            (self.index_map.read(), self.providers.write())
        {
            if let Some(&idx) = index_map.get(provider_name) {
                if idx < providers.len() {
                    providers[idx].record_start();
                }
            }
        }
    }

    /// Record a successful response for a provider.
    pub fn record_success(&self, provider_name: &str, latency: Duration) {
        if let (Ok(index_map), Ok(mut providers)) =
            (self.index_map.read(), self.providers.write())
        {
            if let Some(&idx) = index_map.get(provider_name) {
                if idx < providers.len() {
                    providers[idx].record_success(latency);
                }
            }
        }
    }

    /// Record a failed response for a provider.
    pub fn record_failure(&self, provider_name: &str) {
        if let (Ok(index_map), Ok(mut providers)) =
            (self.index_map.read(), self.providers.write())
        {
            if let Some(&idx) = index_map.get(provider_name) {
                if idx < providers.len() {
                    providers[idx].record_failure();
                }
            }
        }
    }

    /// Get health info for all providers.
    pub fn health_snapshot(&self) -> Vec<ProviderHealth> {
        self.providers
            .read()
            .map(|p| p.clone())
            .unwrap_or_default()
    }

    /// Get the strategy name.
    pub fn strategy_name(&self) -> &str {
        self.strategy.name()
    }

    /// Get healthy provider count.
    pub fn healthy_count(&self) -> usize {
        self.providers
            .read()
            .map(|p| p.iter().filter(|h| h.healthy).count())
            .unwrap_or(0)
    }

    /// Get total provider count.
    pub fn total_count(&self) -> usize {
        self.providers.read().map(|p| p.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_providers(n: usize) -> Vec<ProviderHealth> {
        (0..n)
            .map(|i| ProviderHealth::new(&format!("provider-{}", i), 100))
            .collect()
    }

    // --- ProviderHealth Tests ---

    #[test]
    fn test_provider_health_new() {
        let h = ProviderHealth::new("test", 80);
        assert_eq!(h.name, "test");
        assert_eq!(h.weight, 80);
        assert!(h.healthy);
        assert_eq!(h.total_requests, 0);
        assert_eq!(h.error_rate(), 0.0);
    }

    #[test]
    fn test_provider_health_success() {
        let mut h = ProviderHealth::new("test", 100);
        h.record_start();
        assert_eq!(h.in_flight, 1);
        h.record_success(Duration::from_millis(50));
        assert_eq!(h.in_flight, 0);
        assert_eq!(h.total_requests, 1);
        assert!(h.latency_ewma_ms > 0.0);
        assert!(h.last_success.is_some());
    }

    #[test]
    fn test_provider_health_failure() {
        let mut h = ProviderHealth::new("test", 100);
        h.record_start();
        h.record_failure();
        assert_eq!(h.failed_requests, 1);
        assert!(h.last_error.is_some());
    }

    #[test]
    fn test_provider_health_marks_unhealthy() {
        let mut h = ProviderHealth::new("test", 100);
        // Need 3+ requests with >50% error rate
        for _ in 0..4 {
            h.record_failure();
        }
        assert!(!h.healthy);
    }

    #[test]
    fn test_provider_health_recovers() {
        let mut h = ProviderHealth::new("test", 100);
        for _ in 0..4 {
            h.record_failure();
        }
        assert!(!h.healthy);
        // Many successes bring error rate below 50%
        for _ in 0..10 {
            h.record_success(Duration::from_millis(10));
        }
        assert!(h.healthy);
    }

    #[test]
    fn test_provider_latency_ewma() {
        let mut h = ProviderHealth::new("test", 100);
        h.record_success(Duration::from_millis(100));
        assert!((h.latency_ewma_ms - 100.0).abs() < 1.0);
        h.record_success(Duration::from_millis(200));
        // EWMA: 0.3 * 200 + 0.7 * 100 = 130
        assert!((h.latency_ewma_ms - 130.0).abs() < 1.0);
    }

    #[test]
    fn test_provider_error_rate() {
        let mut h = ProviderHealth::new("test", 100);
        h.record_success(Duration::from_millis(10));
        h.record_failure();
        assert!((h.error_rate() - 0.5).abs() < 0.001);
    }

    // --- Round Robin Tests ---

    #[test]
    fn test_round_robin_cycles() {
        let rr = RoundRobin::new();
        let providers = make_providers(3);
        let mut selections = Vec::new();
        for _ in 0..6 {
            selections.push(rr.select(&providers).unwrap());
        }
        assert_eq!(selections, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_round_robin_skips_unhealthy() {
        let rr = RoundRobin::new();
        let mut providers = make_providers(3);
        providers[1].healthy = false;
        let mut selections = Vec::new();
        for _ in 0..4 {
            selections.push(rr.select(&providers).unwrap());
        }
        assert_eq!(selections, vec![0, 2, 0, 2]);
    }

    #[test]
    fn test_round_robin_none_when_all_unhealthy() {
        let rr = RoundRobin::new();
        let mut providers = make_providers(2);
        providers[0].healthy = false;
        providers[1].healthy = false;
        assert!(rr.select(&providers).is_none());
    }

    #[test]
    fn test_round_robin_empty() {
        let rr = RoundRobin::new();
        assert!(rr.select(&[]).is_none());
    }

    // --- Weighted Tests ---

    #[test]
    fn test_weighted_respects_weights() {
        let w = Weighted::new();
        let mut providers = make_providers(2);
        providers[0].weight = 100;
        providers[1].weight = 0; // should never be selected

        for _ in 0..10 {
            assert_eq!(w.select(&providers).unwrap(), 0);
        }
    }

    #[test]
    fn test_weighted_skips_unhealthy() {
        let w = Weighted::new();
        let mut providers = make_providers(2);
        providers[0].healthy = false;
        providers[0].weight = 100;
        providers[1].weight = 50;

        for _ in 0..10 {
            assert_eq!(w.select(&providers).unwrap(), 1);
        }
    }

    #[test]
    fn test_weighted_none_when_no_weight() {
        let w = Weighted::new();
        let mut providers = make_providers(2);
        providers[0].weight = 0;
        providers[1].weight = 0;
        assert!(w.select(&providers).is_none());
    }

    // --- P2C Tests ---

    #[test]
    fn test_p2c_selects_one_provider() {
        let p2c = P2C::new();
        let providers = make_providers(1);
        assert_eq!(p2c.select(&providers).unwrap(), 0);
    }

    #[test]
    fn test_p2c_prefers_less_loaded() {
        let p2c = P2C::new();
        let mut providers = make_providers(2);
        providers[0].in_flight = 10;
        providers[1].in_flight = 1;

        // Run many times — should almost always pick provider 1
        let mut count_1 = 0;
        for _ in 0..100 {
            if p2c.select(&providers).unwrap() == 1 {
                count_1 += 1;
            }
        }
        assert!(count_1 > 90, "P2C should strongly prefer less-loaded provider");
    }

    #[test]
    fn test_p2c_skips_unhealthy() {
        let p2c = P2C::new();
        let mut providers = make_providers(3);
        providers[0].healthy = false;
        providers[2].healthy = false;
        // Only provider 1 is healthy
        for _ in 0..10 {
            assert_eq!(p2c.select(&providers).unwrap(), 1);
        }
    }

    #[test]
    fn test_p2c_empty() {
        let p2c = P2C::new();
        assert!(p2c.select(&[]).is_none());
    }

    // --- Least Connections Tests ---

    #[test]
    fn test_least_connections_picks_least() {
        let lc = LeastConnections::new();
        let mut providers = make_providers(3);
        providers[0].in_flight = 5;
        providers[1].in_flight = 2;
        providers[2].in_flight = 8;
        assert_eq!(lc.select(&providers).unwrap(), 1);
    }

    #[test]
    fn test_least_connections_skips_unhealthy() {
        let lc = LeastConnections::new();
        let mut providers = make_providers(2);
        providers[0].in_flight = 0;
        providers[0].healthy = false;
        providers[1].in_flight = 5;
        assert_eq!(lc.select(&providers).unwrap(), 1);
    }

    // --- LoadBalancer Tests ---

    #[test]
    fn test_load_balancer_select() {
        let lb = LoadBalancer::new(
            Box::new(RoundRobin::new()),
            vec![("a", 100), ("b", 100)],
        );
        let first = lb.select().unwrap();
        let second = lb.select().unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn test_load_balancer_record_success() {
        let lb = LoadBalancer::new(
            Box::new(RoundRobin::new()),
            vec![("a", 100)],
        );
        lb.record_start("a");
        lb.record_success("a", Duration::from_millis(50));
        let health = lb.health_snapshot();
        assert_eq!(health[0].total_requests, 1);
        assert_eq!(health[0].in_flight, 0);
    }

    #[test]
    fn test_load_balancer_record_failure() {
        let lb = LoadBalancer::new(
            Box::new(RoundRobin::new()),
            vec![("a", 100)],
        );
        lb.record_failure("a");
        let health = lb.health_snapshot();
        assert_eq!(health[0].failed_requests, 1);
    }

    #[test]
    fn test_load_balancer_healthy_count() {
        let lb = LoadBalancer::new(
            Box::new(RoundRobin::new()),
            vec![("a", 100), ("b", 100), ("c", 100)],
        );
        assert_eq!(lb.healthy_count(), 3);
        assert_eq!(lb.total_count(), 3);
    }

    #[test]
    fn test_load_balancer_strategy_name() {
        let lb = LoadBalancer::new(
            Box::new(P2C::new()),
            vec![("a", 100)],
        );
        assert_eq!(lb.strategy_name(), "p2c");
    }

    #[test]
    fn test_load_balancer_unknown_provider() {
        let lb = LoadBalancer::new(
            Box::new(RoundRobin::new()),
            vec![("a", 100)],
        );
        // Should not panic on unknown provider
        lb.record_start("unknown");
        lb.record_success("unknown", Duration::from_millis(10));
        lb.record_failure("unknown");
    }
}
