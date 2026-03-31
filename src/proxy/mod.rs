//! Proxy module — HTTP reverse proxy engine for LLM API requests.
//!
//! Accepts incoming OpenAI-format requests, routes them through the
//! middleware pipeline (rate limit → cache → balance → forward),
//! and returns responses in OpenAI-compatible format.

use crate::balance::LoadBalancer;
use crate::cache::{CacheKeyComponents, CachedResponse, CachedUsage, MemoryCache};
use crate::config::GatewayConfig;
use crate::cost::{CostCalculator, RequestCost, SpendTracker, TokenUsage};
use crate::providers::{self, ChatRequest, Provider};
use crate::ratelimit::{BudgetTracker, RateLimitResult, SlidingWindowLimiter, TokenBucketLimiter};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Gateway error types.
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("rate limited: retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },
    #[error("budget exceeded for key")]
    BudgetExceeded,
    #[error("unknown model: {0}")]
    UnknownModel(String),
    #[error("no healthy provider available")]
    NoProvider,
    #[error("upstream error: {status} {body}")]
    UpstreamError { status: u16, body: String },
    #[error("request error: {0}")]
    RequestError(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

/// Gateway response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayResponse {
    /// The response body (OpenAI-compatible JSON).
    pub body: serde_json::Value,
    /// HTTP status code.
    pub status: u16,
    /// Whether this was a cache hit.
    pub cache_hit: bool,
    /// Cost of this request (if tracking enabled).
    pub cost: Option<RequestCost>,
    /// Which provider handled the request.
    pub provider: String,
    /// Latency in milliseconds.
    pub latency_ms: u64,
}

/// Shared gateway state accessible by all request handlers.
pub struct GatewayState {
    pub config: GatewayConfig,
    pub cache: MemoryCache,
    pub rate_limiter: TokenBucketLimiter,
    pub token_limiter: SlidingWindowLimiter,
    pub budget_tracker: BudgetTracker,
    pub load_balancer: LoadBalancer,
    pub cost_calculator: CostCalculator,
    pub spend_tracker: SpendTracker,
    pub http_client: reqwest::Client,
}

impl GatewayState {
    /// Create a new gateway state from config.
    pub fn from_config(config: GatewayConfig) -> Arc<Self> {
        let cache = MemoryCache::new(config.cache.max_entries, config.cache.ttl_seconds);

        let rate_limiter = TokenBucketLimiter::new(
            config.rate_limit.requests_per_minute,
            config.rate_limit.requests_per_minute,
        );

        let token_limiter = SlidingWindowLimiter::new(
            config.rate_limit.tokens_per_minute,
            std::time::Duration::from_secs(60),
        );

        let budget_tracker = BudgetTracker::new(config.rate_limit.max_spend_per_day);

        let provider_configs: Vec<(&str, u32)> = config
            .providers
            .iter()
            .filter(|p| p.enabled)
            .map(|p| (p.name.as_str(), p.weight))
            .collect();

        let strategy: Box<dyn crate::balance::BalanceStrategy> =
            match config.balance.strategy {
                crate::config::BalanceStrategy::RoundRobin => {
                    Box::new(crate::balance::RoundRobin::new())
                }
                crate::config::BalanceStrategy::P2c => Box::new(crate::balance::P2C::new()),
                crate::config::BalanceStrategy::Weighted => {
                    Box::new(crate::balance::Weighted::new())
                }
                crate::config::BalanceStrategy::LeastConnections => {
                    Box::new(crate::balance::LeastConnections::new())
                }
            };

        let load_balancer = LoadBalancer::new(strategy, provider_configs);

        let cost_overrides = config
            .cost
            .pricing_overrides
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    (v.input_cost_per_million, v.output_cost_per_million),
                )
            })
            .collect();
        let cost_calculator = CostCalculator::with_overrides(cost_overrides);

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default();

        Arc::new(Self {
            config,
            cache,
            rate_limiter,
            token_limiter,
            budget_tracker,
            load_balancer,
            cost_calculator,
            spend_tracker: SpendTracker::new(),
            http_client,
        })
    }
}

/// Process a chat completion request through the gateway pipeline.
///
/// Pipeline: rate limit → cache check → load balance → proxy → cache store → cost track
pub async fn handle_chat_request(
    state: &Arc<GatewayState>,
    request_body: &str,
    api_key: &str,
) -> Result<GatewayResponse, GatewayError> {
    let start = Instant::now();

    // 1. Parse request
    let chat_request: ChatRequest = serde_json::from_str(request_body)
        .map_err(|e| GatewayError::InvalidRequest(e.to_string()))?;

    // 2. Rate limit check (RPM)
    if state.config.rate_limit.enabled {
        let result = state.rate_limiter.check(api_key, 1);
        if let RateLimitResult::Denied { retry_after } = result {
            return Err(GatewayError::RateLimited {
                retry_after_ms: retry_after.as_millis() as u64,
            });
        }
    }

    // 3. Cache check
    if state.config.cache.enabled {
        let cache_key = build_cache_key(&chat_request, &state.config.cache.ignore_keys);
        if let Some(cached) = state.cache.get(&cache_key) {
            let body: serde_json::Value = serde_json::from_slice(&cached.body).unwrap_or_default();

            let cost = if state.config.cost.enabled {
                let usage = cached.usage.map(|u| TokenUsage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                });
                let cost = if let Some(u) = &usage {
                    state.cost_calculator.calculate_cached(&cached.model, u)
                } else {
                    RequestCost::default()
                };
                state.spend_tracker.record(api_key, &cost);
                Some(cost)
            } else {
                None
            };

            return Ok(GatewayResponse {
                body,
                status: cached.status,
                cache_hit: true,
                cost,
                provider: "cache".to_string(),
                latency_ms: start.elapsed().as_millis() as u64,
            });
        }
    }

    // 4. Resolve model to provider
    let (model_str, model_id) = parse_model_string(&chat_request.model);
    let provider_config = if let Some(provider_name) = model_str {
        state
            .config
            .providers
            .iter()
            .find(|p| p.name == provider_name && p.enabled)
    } else {
        // Use load balancer to pick provider
        let selected = state
            .load_balancer
            .select()
            .ok_or(GatewayError::NoProvider)?;
        state.config.providers.iter().find(|p| p.name == selected)
    };

    let provider_config = provider_config.ok_or_else(|| {
        GatewayError::UnknownModel(chat_request.model.clone())
    })?;

    let provider = Provider::from_name(&provider_config.name);
    let provider_name = provider_config.name.clone();
    let base_url = provider_config.base_url.clone();
    let provider_api_key = provider_config.resolve_api_key().unwrap_or_default();

    // 5. Transform request for provider
    let mut transformed_request = chat_request.clone();
    transformed_request.model = model_id.to_string();
    let request_body = providers::transform_request(&provider, &transformed_request);
    let url = providers::build_url(&base_url, &provider, model_id);

    // 6. Forward to upstream
    state.load_balancer.record_start(&provider_name);

    let mut req = state
        .http_client
        .post(&url)
        .header(
            provider.auth_header(),
            provider.auth_value(&provider_api_key),
        )
        .json(&request_body);

    for (header, value) in provider.extra_headers() {
        req = req.header(header, value);
    }

    let upstream_response = req
        .send()
        .await
        .map_err(|e| GatewayError::RequestError(e.to_string()))?;

    let status = upstream_response.status().as_u16();
    let response_body = upstream_response
        .text()
        .await
        .map_err(|e| GatewayError::RequestError(e.to_string()))?;

    let latency = start.elapsed();

    if status >= 400 {
        state.load_balancer.record_failure(&provider_name);
        return Err(GatewayError::UpstreamError {
            status,
            body: response_body,
        });
    }

    state
        .load_balancer
        .record_success(&provider_name, latency);

    // 7. Parse and transform response
    let response_json: serde_json::Value =
        serde_json::from_str(&response_body).unwrap_or_default();
    let normalized = providers::transform_response(&provider, &response_json, model_id);

    // 8. Cost tracking
    let cost = if state.config.cost.enabled {
        let usage = crate::cost::extract_usage(&normalized).unwrap_or_default();
        let cost = state.cost_calculator.calculate(model_id, &usage);
        state.spend_tracker.record(api_key, &cost);

        // Token-per-minute tracking (post-request — record tokens used)
        if state.config.rate_limit.enabled && usage.total_tokens > 0 {
            state.token_limiter.check(api_key, usage.total_tokens);
        }

        // Budget check (post-request — record spend, warn if exceeded)
        if state.config.rate_limit.max_spend_per_day > 0.0 {
            state
                .budget_tracker
                .check(api_key, cost.total_cost);
        }

        Some(cost)
    } else {
        None
    };

    // 9. Cache store
    if state.config.cache.enabled {
        let cache_key = build_cache_key(&chat_request, &state.config.cache.ignore_keys);
        let usage_for_cache = crate::cost::extract_usage(&normalized).map(|u| CachedUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });
        let cached = CachedResponse {
            status,
            headers: vec![],
            body: serde_json::to_vec(&normalized).unwrap_or_default(),
            model: model_id.to_string(),
            usage: usage_for_cache,
        };
        state.cache.put(cache_key, cached);
    }

    Ok(GatewayResponse {
        body: normalized,
        status,
        cache_hit: false,
        cost,
        provider: provider_name,
        latency_ms: latency.as_millis() as u64,
    })
}

/// Parse a model string like "openai/gpt-4o" into (Some("openai"), "gpt-4o").
/// If no prefix, returns (None, model_string).
fn parse_model_string(model: &str) -> (Option<&str>, &str) {
    if let Some((prefix, id)) = model.split_once('/') {
        (Some(prefix), id)
    } else {
        (None, model)
    }
}

/// Build a cache key from a chat request.
fn build_cache_key(request: &ChatRequest, ignore_keys: &[String]) -> String {
    let components = CacheKeyComponents {
        model: request.model.clone(),
        messages: serde_json::to_value(&request.messages).unwrap_or_default(),
        temperature: request.temperature,
        seed: request.seed,
        cache_seed: None,
        extra_params: request
            .extra
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<BTreeMap<_, _>>(),
    };
    crate::cache::compute_cache_key(&components, ignore_keys)
}

/// Gateway health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub providers: Vec<ProviderStatus>,
    pub cache_entries: usize,
    pub total_requests: u64,
    pub total_spend: f64,
}

/// Provider status in health check.
#[derive(Debug, Serialize)]
pub struct ProviderStatus {
    pub name: String,
    pub healthy: bool,
    pub latency_ms: f64,
    pub error_rate: f64,
}

/// Build a health check response.
pub fn health_check(state: &Arc<GatewayState>) -> HealthResponse {
    let health = state.load_balancer.health_snapshot();
    let providers: Vec<ProviderStatus> = health
        .iter()
        .map(|h| ProviderStatus {
            name: h.name.clone(),
            healthy: h.healthy,
            latency_ms: h.latency_ewma_ms,
            error_rate: h.error_rate(),
        })
        .collect();

    HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        providers,
        cache_entries: state.cache.len(),
        total_requests: state.spend_tracker.request_count(),
        total_spend: state.spend_tracker.global_spend(),
    }
}

/// Stats for the /stats endpoint.
#[derive(Debug, Serialize)]
pub struct GatewayStats {
    pub cache_entries: usize,
    pub cache_capacity: usize,
    pub total_requests: u64,
    pub total_spend: f64,
    pub healthy_providers: usize,
    pub total_providers: usize,
    pub balance_strategy: String,
    pub spend_by_key: std::collections::HashMap<String, f64>,
}

/// Build stats response.
pub fn gateway_stats(state: &Arc<GatewayState>) -> GatewayStats {
    let cache_stats = state.cache.stats();
    GatewayStats {
        cache_entries: cache_stats.entries,
        cache_capacity: cache_stats.capacity,
        total_requests: state.spend_tracker.request_count(),
        total_spend: state.spend_tracker.global_spend(),
        healthy_providers: state.load_balancer.healthy_count(),
        total_providers: state.load_balancer.total_count(),
        balance_strategy: state.load_balancer.strategy_name().to_string(),
        spend_by_key: state.spend_tracker.summary(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_with_prefix() {
        let (prefix, id) = parse_model_string("openai/gpt-4o");
        assert_eq!(prefix, Some("openai"));
        assert_eq!(id, "gpt-4o");
    }

    #[test]
    fn test_parse_model_without_prefix() {
        let (prefix, id) = parse_model_string("gpt-4o");
        assert_eq!(prefix, None);
        assert_eq!(id, "gpt-4o");
    }

    #[test]
    fn test_parse_model_multiple_slashes() {
        let (prefix, id) = parse_model_string("azure/deployments/gpt-4o");
        assert_eq!(prefix, Some("azure"));
        assert_eq!(id, "deployments/gpt-4o");
    }

    #[test]
    fn test_build_cache_key_consistency() {
        let req = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![crate::providers::ChatMessage {
                role: "user".to_string(),
                content: serde_json::json!("hello"),
            }],
            temperature: Some(0.7),
            max_tokens: None,
            stream: None,
            top_p: None,
            seed: None,
            extra: std::collections::HashMap::new(),
        };
        let key1 = build_cache_key(&req, &[]);
        let key2 = build_cache_key(&req, &[]);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_build_cache_key_different_models() {
        let mut req = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            stream: None,
            top_p: None,
            seed: None,
            extra: std::collections::HashMap::new(),
        };
        let key1 = build_cache_key(&req, &[]);
        req.model = "gpt-4o-mini".to_string();
        let key2 = build_cache_key(&req, &[]);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_gateway_state_from_config() {
        let config = crate::config::GatewayConfig::from_str(
            r#"
providers:
  - name: openai
    base_url: https://api.openai.com
    weight: 100
    models:
      - id: gpt-4o
rate_limit:
  enabled: true
  requests_per_minute: 60
  tokens_per_minute: 100000
"#,
        )
        .unwrap();
        let state = GatewayState::from_config(config);
        assert_eq!(state.load_balancer.total_count(), 1);
        // Token limiter should be initialized and tracking no usage
        assert_eq!(state.token_limiter.current_count("any-key"), 0.0);
    }

    #[test]
    fn test_health_check() {
        let config = crate::config::GatewayConfig::from_str(
            r#"
providers:
  - name: openai
    base_url: https://api.openai.com
  - name: anthropic
    base_url: https://api.anthropic.com
"#,
        )
        .unwrap();
        let state = GatewayState::from_config(config);
        let health = health_check(&state);
        assert_eq!(health.status, "ok");
        assert_eq!(health.providers.len(), 2);
        assert_eq!(health.version, "1.0.0");
    }

    #[test]
    fn test_gateway_stats() {
        let config = crate::config::GatewayConfig::from_str(
            r#"
providers:
  - name: openai
    base_url: https://api.openai.com
balance:
  strategy: p2c
"#,
        )
        .unwrap();
        let state = GatewayState::from_config(config);
        let stats = gateway_stats(&state);
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.balance_strategy, "p2c");
        assert_eq!(stats.healthy_providers, 1);
    }

    #[test]
    fn test_gateway_error_display() {
        let err = GatewayError::RateLimited {
            retry_after_ms: 5000,
        };
        assert!(err.to_string().contains("5000"));

        let err = GatewayError::UnknownModel("bad-model".to_string());
        assert!(err.to_string().contains("bad-model"));
    }

    #[test]
    fn test_gateway_response_serialize() {
        let resp = GatewayResponse {
            body: serde_json::json!({"id": "test"}),
            status: 200,
            cache_hit: false,
            cost: None,
            provider: "openai".to_string(),
            latency_ms: 150,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"status\":200"));
        assert!(json.contains("\"cache_hit\":false"));
    }
}
