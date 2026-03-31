//! Cost tracking module — token counting and spend aggregation.
//!
//! Includes an embedded pricing database for common LLM models,
//! per-request cost computation, and per-key spend accumulation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Token usage extracted from an LLM response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Cost breakdown for a single request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestCost {
    /// Cost for input/prompt tokens in dollars.
    pub input_cost: f64,
    /// Cost for output/completion tokens in dollars.
    pub output_cost: f64,
    /// Total cost in dollars.
    pub total_cost: f64,
    /// Model used.
    pub model: String,
    /// Token usage.
    pub usage: TokenUsage,
}

/// Model pricing entry — cost per million tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Model name/identifier.
    pub model: String,
    /// Provider name.
    pub provider: String,
    /// Cost per million input tokens.
    pub input_cost_per_million: f64,
    /// Cost per million output tokens.
    pub output_cost_per_million: f64,
    /// Cache read cost multiplier (e.g., 0.1 = 10% of normal cost).
    #[serde(default = "default_cache_read_multiplier")]
    pub cache_read_multiplier: f64,
    /// Context window size (optional).
    pub context_window: Option<u64>,
}

fn default_cache_read_multiplier() -> f64 {
    0.1
}

/// Build the embedded pricing database.
fn default_pricing_db() -> HashMap<String, ModelPricing> {
    let models = vec![
        // OpenAI
        ("gpt-4o", "openai", 2.50, 10.0, 128000),
        ("gpt-4o-mini", "openai", 0.15, 0.60, 128000),
        ("gpt-4-turbo", "openai", 10.0, 30.0, 128000),
        ("gpt-4", "openai", 30.0, 60.0, 8192),
        ("gpt-3.5-turbo", "openai", 0.50, 1.50, 16385),
        ("o1", "openai", 15.0, 60.0, 200000),
        ("o1-mini", "openai", 3.0, 12.0, 128000),
        ("o3-mini", "openai", 1.10, 4.40, 200000),
        // Anthropic
        ("claude-3-5-sonnet", "anthropic", 3.0, 15.0, 200000),
        ("claude-3-5-haiku", "anthropic", 0.80, 4.0, 200000),
        ("claude-3-opus", "anthropic", 15.0, 75.0, 200000),
        ("claude-sonnet-4", "anthropic", 3.0, 15.0, 200000),
        // Google
        ("gemini-2.0-flash", "google", 0.10, 0.40, 1048576),
        ("gemini-1.5-pro", "google", 1.25, 5.0, 2097152),
        ("gemini-1.5-flash", "google", 0.075, 0.30, 1048576),
        // Meta (via providers)
        ("llama-3.1-405b", "meta", 3.0, 3.0, 128000),
        ("llama-3.1-70b", "meta", 0.70, 0.80, 128000),
        ("llama-3.1-8b", "meta", 0.05, 0.08, 128000),
        // DeepSeek
        ("deepseek-v3", "deepseek", 0.27, 1.10, 128000),
        ("deepseek-r1", "deepseek", 0.55, 2.19, 128000),
    ];

    let mut db = HashMap::new();
    for (model, provider, input, output, ctx) in models {
        db.insert(
            model.to_string(),
            ModelPricing {
                model: model.to_string(),
                provider: provider.to_string(),
                input_cost_per_million: input,
                output_cost_per_million: output,
                cache_read_multiplier: 0.1,
                context_window: Some(ctx),
            },
        );
    }
    db
}

/// Cost calculator — computes per-request costs using the pricing database.
pub struct CostCalculator {
    /// Pricing database: model_name -> pricing.
    pricing_db: HashMap<String, ModelPricing>,
}

impl CostCalculator {
    /// Create a calculator with the default embedded pricing database.
    pub fn new() -> Self {
        Self {
            pricing_db: default_pricing_db(),
        }
    }

    /// Create a calculator with custom pricing overrides merged into defaults.
    pub fn with_overrides(overrides: HashMap<String, (f64, f64)>) -> Self {
        let mut db = default_pricing_db();
        for (model, (input_cost, output_cost)) in overrides {
            db.insert(
                model.clone(),
                ModelPricing {
                    model: model.clone(),
                    provider: "custom".to_string(),
                    input_cost_per_million: input_cost,
                    output_cost_per_million: output_cost,
                    cache_read_multiplier: 0.1,
                    context_window: None,
                },
            );
        }
        Self { pricing_db: db }
    }

    /// Calculate cost for a request given model and usage.
    pub fn calculate(&self, model: &str, usage: &TokenUsage) -> RequestCost {
        let pricing = self.pricing_db.get(model);

        let (input_rate, output_rate) = match pricing {
            Some(p) => (p.input_cost_per_million, p.output_cost_per_million),
            None => (0.0, 0.0), // Unknown model — zero cost
        };

        let input_cost = usage.prompt_tokens as f64 * input_rate / 1_000_000.0;
        let output_cost = usage.completion_tokens as f64 * output_rate / 1_000_000.0;

        RequestCost {
            input_cost,
            output_cost,
            total_cost: input_cost + output_cost,
            model: model.to_string(),
            usage: usage.clone(),
        }
    }

    /// Calculate cost for a cached response (reduced rate).
    pub fn calculate_cached(&self, model: &str, usage: &TokenUsage) -> RequestCost {
        let pricing = self.pricing_db.get(model);
        let multiplier = pricing
            .map(|p| p.cache_read_multiplier)
            .unwrap_or(0.1);

        let mut cost = self.calculate(model, usage);
        cost.input_cost *= multiplier;
        cost.output_cost *= multiplier;
        cost.total_cost = cost.input_cost + cost.output_cost;
        cost
    }

    /// Look up pricing for a model.
    pub fn get_pricing(&self, model: &str) -> Option<&ModelPricing> {
        self.pricing_db.get(model)
    }

    /// List all known models.
    pub fn known_models(&self) -> Vec<&str> {
        self.pricing_db.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a model is in the pricing database.
    pub fn is_known_model(&self, model: &str) -> bool {
        self.pricing_db.contains_key(model)
    }
}

impl Default for CostCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Spend aggregator — tracks cumulative spend per key.
pub struct SpendTracker {
    /// Per-key spend: key -> total_dollars.
    spend: Arc<RwLock<HashMap<String, f64>>>,
    /// Per-key per-model breakdown: key -> { model -> dollars }.
    breakdown: Arc<RwLock<HashMap<String, HashMap<String, f64>>>>,
    /// Total requests tracked.
    request_count: Arc<RwLock<u64>>,
}

impl SpendTracker {
    pub fn new() -> Self {
        Self {
            spend: Arc::new(RwLock::new(HashMap::new())),
            breakdown: Arc::new(RwLock::new(HashMap::new())),
            request_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Record a cost for a key.
    pub fn record(&self, key: &str, cost: &RequestCost) {
        if let Ok(mut spend) = self.spend.write() {
            *spend.entry(key.to_string()).or_insert(0.0) += cost.total_cost;
        }
        if let Ok(mut breakdown) = self.breakdown.write() {
            let model_spend = breakdown.entry(key.to_string()).or_default();
            *model_spend.entry(cost.model.clone()).or_insert(0.0) += cost.total_cost;
        }
        if let Ok(mut count) = self.request_count.write() {
            *count += 1;
        }
    }

    /// Get total spend for a key.
    pub fn total_spend(&self, key: &str) -> f64 {
        self.spend
            .read()
            .ok()
            .and_then(|s| s.get(key).copied())
            .unwrap_or(0.0)
    }

    /// Get per-model breakdown for a key.
    pub fn model_breakdown(&self, key: &str) -> HashMap<String, f64> {
        self.breakdown
            .read()
            .ok()
            .and_then(|b| b.get(key).cloned())
            .unwrap_or_default()
    }

    /// Get total spend across all keys.
    pub fn global_spend(&self) -> f64 {
        self.spend
            .read()
            .ok()
            .map(|s| s.values().sum())
            .unwrap_or(0.0)
    }

    /// Get total request count.
    pub fn request_count(&self) -> u64 {
        self.request_count.read().ok().map(|c| *c).unwrap_or(0)
    }

    /// Get spend summary for all keys.
    pub fn summary(&self) -> HashMap<String, f64> {
        self.spend.read().ok().map(|s| s.clone()).unwrap_or_default()
    }

    /// Reset all tracking.
    pub fn reset(&self) {
        if let Ok(mut spend) = self.spend.write() {
            spend.clear();
        }
        if let Ok(mut breakdown) = self.breakdown.write() {
            breakdown.clear();
        }
        if let Ok(mut count) = self.request_count.write() {
            *count = 0;
        }
    }

    /// Reset a specific key.
    pub fn reset_key(&self, key: &str) {
        if let Ok(mut spend) = self.spend.write() {
            spend.remove(key);
        }
        if let Ok(mut breakdown) = self.breakdown.write() {
            breakdown.remove(key);
        }
    }
}

impl Default for SpendTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SpendTracker {
    fn clone(&self) -> Self {
        Self {
            spend: Arc::clone(&self.spend),
            breakdown: Arc::clone(&self.breakdown),
            request_count: Arc::clone(&self.request_count),
        }
    }
}

/// Extract token usage from an OpenAI-compatible response JSON.
pub fn extract_usage(response: &serde_json::Value) -> Option<TokenUsage> {
    let usage = response.get("usage")?;
    Some(TokenUsage {
        prompt_tokens: usage.get("prompt_tokens")?.as_u64()?,
        completion_tokens: usage.get("completion_tokens")?.as_u64().unwrap_or(0),
        total_tokens: usage.get("total_tokens")?.as_u64().unwrap_or(0),
    })
}

/// Extract model name from an OpenAI-compatible response JSON.
pub fn extract_model(response: &serde_json::Value) -> Option<String> {
    response.get("model")?.as_str().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CostCalculator Tests ---

    #[test]
    fn test_calculator_known_model() {
        let calc = CostCalculator::new();
        assert!(calc.is_known_model("gpt-4o"));
        assert!(calc.is_known_model("claude-3-5-sonnet"));
        assert!(!calc.is_known_model("nonexistent-model"));
    }

    #[test]
    fn test_calculator_gpt4o_cost() {
        let calc = CostCalculator::new();
        let usage = TokenUsage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };
        let cost = calc.calculate("gpt-4o", &usage);
        // gpt-4o: $2.50/M input, $10.0/M output
        let expected_input = 1000.0 * 2.50 / 1_000_000.0;
        let expected_output = 500.0 * 10.0 / 1_000_000.0;
        assert!((cost.input_cost - expected_input).abs() < 1e-10);
        assert!((cost.output_cost - expected_output).abs() < 1e-10);
        assert!((cost.total_cost - (expected_input + expected_output)).abs() < 1e-10);
    }

    #[test]
    fn test_calculator_unknown_model_zero_cost() {
        let calc = CostCalculator::new();
        let usage = TokenUsage {
            prompt_tokens: 1000,
            completion_tokens: 1000,
            total_tokens: 2000,
        };
        let cost = calc.calculate("unknown-model", &usage);
        assert_eq!(cost.total_cost, 0.0);
    }

    #[test]
    fn test_calculator_cached_cost() {
        let calc = CostCalculator::new();
        let usage = TokenUsage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };
        let normal = calc.calculate("gpt-4o", &usage);
        let cached = calc.calculate_cached("gpt-4o", &usage);
        assert!((cached.total_cost - normal.total_cost * 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_calculator_with_overrides() {
        let mut overrides = HashMap::new();
        overrides.insert("custom-model".to_string(), (1.0, 2.0));
        let calc = CostCalculator::with_overrides(overrides);

        assert!(calc.is_known_model("custom-model"));
        let usage = TokenUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
        };
        let cost = calc.calculate("custom-model", &usage);
        assert!((cost.input_cost - 1.0).abs() < 1e-10);
        assert!((cost.output_cost - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculator_get_pricing() {
        let calc = CostCalculator::new();
        let pricing = calc.get_pricing("gpt-4o").unwrap();
        assert_eq!(pricing.provider, "openai");
        assert!(pricing.context_window.is_some());
    }

    #[test]
    fn test_calculator_known_models_count() {
        let calc = CostCalculator::new();
        assert!(calc.known_models().len() >= 15);
    }

    #[test]
    fn test_calculator_zero_tokens() {
        let calc = CostCalculator::new();
        let usage = TokenUsage::default();
        let cost = calc.calculate("gpt-4o", &usage);
        assert_eq!(cost.total_cost, 0.0);
    }

    #[test]
    fn test_calculator_large_token_count() {
        let calc = CostCalculator::new();
        let usage = TokenUsage {
            prompt_tokens: 10_000_000,
            completion_tokens: 5_000_000,
            total_tokens: 15_000_000,
        };
        let cost = calc.calculate("gpt-4o", &usage);
        // 10M * $2.50/M + 5M * $10.0/M = $25 + $50 = $75
        assert!((cost.total_cost - 75.0).abs() < 1e-6);
    }

    // --- SpendTracker Tests ---

    #[test]
    fn test_spend_tracker_record() {
        let tracker = SpendTracker::new();
        let cost = RequestCost {
            input_cost: 0.01,
            output_cost: 0.02,
            total_cost: 0.03,
            model: "gpt-4o".to_string(),
            usage: TokenUsage::default(),
        };
        tracker.record("user1", &cost);
        assert!((tracker.total_spend("user1") - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_spend_tracker_accumulates() {
        let tracker = SpendTracker::new();
        let cost = RequestCost {
            input_cost: 0.01,
            output_cost: 0.02,
            total_cost: 0.03,
            model: "gpt-4o".to_string(),
            usage: TokenUsage::default(),
        };
        tracker.record("user1", &cost);
        tracker.record("user1", &cost);
        assert!((tracker.total_spend("user1") - 0.06).abs() < 1e-10);
        assert_eq!(tracker.request_count(), 2);
    }

    #[test]
    fn test_spend_tracker_per_key() {
        let tracker = SpendTracker::new();
        let cost1 = RequestCost {
            total_cost: 1.0,
            model: "gpt-4o".to_string(),
            ..Default::default()
        };
        let cost2 = RequestCost {
            total_cost: 2.0,
            model: "gpt-4o".to_string(),
            ..Default::default()
        };
        tracker.record("user1", &cost1);
        tracker.record("user2", &cost2);
        assert!((tracker.total_spend("user1") - 1.0).abs() < 1e-10);
        assert!((tracker.total_spend("user2") - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_spend_tracker_model_breakdown() {
        let tracker = SpendTracker::new();
        tracker.record(
            "user1",
            &RequestCost {
                total_cost: 1.0,
                model: "gpt-4o".to_string(),
                ..Default::default()
            },
        );
        tracker.record(
            "user1",
            &RequestCost {
                total_cost: 2.0,
                model: "claude-3-5-sonnet".to_string(),
                ..Default::default()
            },
        );
        let breakdown = tracker.model_breakdown("user1");
        assert!((breakdown["gpt-4o"] - 1.0).abs() < 1e-10);
        assert!((breakdown["claude-3-5-sonnet"] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_spend_tracker_global_spend() {
        let tracker = SpendTracker::new();
        tracker.record(
            "a",
            &RequestCost {
                total_cost: 1.0,
                model: "x".to_string(),
                ..Default::default()
            },
        );
        tracker.record(
            "b",
            &RequestCost {
                total_cost: 2.0,
                model: "x".to_string(),
                ..Default::default()
            },
        );
        assert!((tracker.global_spend() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_spend_tracker_reset() {
        let tracker = SpendTracker::new();
        tracker.record(
            "user1",
            &RequestCost {
                total_cost: 5.0,
                model: "x".to_string(),
                ..Default::default()
            },
        );
        tracker.reset();
        assert_eq!(tracker.total_spend("user1"), 0.0);
        assert_eq!(tracker.request_count(), 0);
    }

    #[test]
    fn test_spend_tracker_reset_key() {
        let tracker = SpendTracker::new();
        tracker.record(
            "user1",
            &RequestCost {
                total_cost: 1.0,
                model: "x".to_string(),
                ..Default::default()
            },
        );
        tracker.record(
            "user2",
            &RequestCost {
                total_cost: 2.0,
                model: "x".to_string(),
                ..Default::default()
            },
        );
        tracker.reset_key("user1");
        assert_eq!(tracker.total_spend("user1"), 0.0);
        assert!((tracker.total_spend("user2") - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_spend_tracker_unknown_key() {
        let tracker = SpendTracker::new();
        assert_eq!(tracker.total_spend("unknown"), 0.0);
    }

    #[test]
    fn test_spend_tracker_clone_shares_state() {
        let t1 = SpendTracker::new();
        let t2 = t1.clone();
        t1.record(
            "user1",
            &RequestCost {
                total_cost: 5.0,
                model: "x".to_string(),
                ..Default::default()
            },
        );
        assert!((t2.total_spend("user1") - 5.0).abs() < 1e-10);
    }

    // --- Usage Extraction Tests ---

    #[test]
    fn test_extract_usage_openai_format() {
        let response = serde_json::json!({
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        });
        let usage = extract_usage(&response).unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_extract_usage_missing() {
        let response = serde_json::json!({"id": "123"});
        assert!(extract_usage(&response).is_none());
    }

    #[test]
    fn test_extract_model() {
        let response = serde_json::json!({"model": "gpt-4o-2024-05-13"});
        assert_eq!(extract_model(&response).unwrap(), "gpt-4o-2024-05-13");
    }

    #[test]
    fn test_extract_model_missing() {
        let response = serde_json::json!({"id": "123"});
        assert!(extract_model(&response).is_none());
    }

    #[test]
    fn test_spend_tracker_summary() {
        let tracker = SpendTracker::new();
        tracker.record(
            "a",
            &RequestCost {
                total_cost: 1.0,
                model: "x".to_string(),
                ..Default::default()
            },
        );
        tracker.record(
            "b",
            &RequestCost {
                total_cost: 2.0,
                model: "x".to_string(),
                ..Default::default()
            },
        );
        let summary = tracker.summary();
        assert_eq!(summary.len(), 2);
        assert!((summary["a"] - 1.0).abs() < 1e-10);
    }
}
