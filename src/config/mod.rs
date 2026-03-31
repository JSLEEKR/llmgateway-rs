//! Configuration module — YAML-based gateway configuration.
//!
//! Parses provider definitions, rate limits, cache settings,
//! load balancing strategy, and server listen address from a YAML file.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Expand environment variable references in a string.
///
/// Supports `${VAR_NAME}` and `$VAR_NAME` syntax.
/// Returns the original string if no env var pattern is found.
pub fn expand_env(value: &str) -> String {
    if let Some(var_name) = value.strip_prefix("${").and_then(|s| s.strip_suffix('}')) {
        std::env::var(var_name).unwrap_or_else(|_| value.to_string())
    } else if let Some(var_name) = value.strip_prefix('$') {
        if !var_name.is_empty() && var_name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            std::env::var(var_name).unwrap_or_else(|_| value.to_string())
        } else {
            value.to_string()
        }
    } else {
        value.to_string()
    }
}

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("validation error: {0}")]
    Validation(String),
}

/// Top-level gateway configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Server listen address (e.g., "0.0.0.0:8080").
    #[serde(default = "default_listen")]
    pub listen: String,

    /// Upstream LLM provider definitions.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,

    /// Cache settings.
    #[serde(default)]
    pub cache: CacheConfig,

    /// Global rate limit settings.
    #[serde(default)]
    pub rate_limit: RateLimitConfig,

    /// Load balancing strategy.
    #[serde(default)]
    pub balance: BalanceConfig,

    /// Cost tracking settings.
    #[serde(default)]
    pub cost: CostConfig,
}

fn default_listen() -> String {
    "0.0.0.0:8080".to_string()
}

/// Provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider name (e.g., "openai", "anthropic").
    pub name: String,

    /// Base URL for the provider API.
    pub base_url: String,

    /// API key for authentication (can reference env var with `$ENV_VAR`).
    #[serde(default)]
    pub api_key: Option<String>,

    /// Authentication method.
    #[serde(default = "default_auth_method")]
    pub auth_method: AuthMethod,

    /// Models available from this provider.
    #[serde(default)]
    pub models: Vec<ModelConfig>,

    /// Weight for weighted load balancing (0-100).
    #[serde(default = "default_weight")]
    pub weight: u32,

    /// Whether this provider is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl ProviderConfig {
    /// Resolve the API key, expanding environment variable references.
    ///
    /// Supports `${VAR_NAME}` and `$VAR_NAME` syntax.
    pub fn resolve_api_key(&self) -> Option<String> {
        self.api_key.as_ref().map(|key| expand_env(key))
    }
}

fn default_auth_method() -> AuthMethod {
    AuthMethod::Bearer
}

fn default_weight() -> u32 {
    100
}

fn default_true() -> bool {
    true
}

/// Authentication method for a provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    Bearer,
    ApiKey,
    Custom(String),
    None,
}

/// Model-specific configuration within a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
    pub id: String,

    /// Cost per input token (per million tokens).
    #[serde(default)]
    pub input_cost_per_million: f64,

    /// Cost per output token (per million tokens).
    #[serde(default)]
    pub output_cost_per_million: f64,

    /// Context window size.
    #[serde(default)]
    pub context_window: Option<u64>,
}

/// Cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether caching is enabled by default.
    #[serde(default)]
    pub enabled: bool,

    /// Default TTL in seconds.
    #[serde(default = "default_ttl")]
    pub ttl_seconds: u64,

    /// Maximum number of cached entries.
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,

    /// Fields to ignore when computing cache key.
    #[serde(default)]
    pub ignore_keys: Vec<String>,
}

fn default_ttl() -> u64 {
    86400 // 24 hours
}

fn default_max_entries() -> usize {
    10000
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ttl_seconds: default_ttl(),
            max_entries: default_max_entries(),
            ignore_keys: Vec::new(),
        }
    }
}

/// Rate limit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Whether rate limiting is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// Default requests per minute per key.
    #[serde(default = "default_rpm")]
    pub requests_per_minute: u64,

    /// Default tokens per minute per key.
    #[serde(default = "default_tpm")]
    pub tokens_per_minute: u64,

    /// Maximum daily spend in dollars (0 = unlimited).
    #[serde(default)]
    pub max_spend_per_day: f64,

    /// Rate limit scope.
    #[serde(default)]
    pub scope: RateLimitScope,
}

fn default_rpm() -> u64 {
    60
}

fn default_tpm() -> u64 {
    100_000
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_minute: default_rpm(),
            tokens_per_minute: default_tpm(),
            max_spend_per_day: 0.0,
            scope: RateLimitScope::PerKey,
        }
    }
}

/// Rate limit scope — what dimension to limit on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitScope {
    #[default]
    PerKey,
    PerUser,
    Global,
}

/// Load balancing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceConfig {
    /// Load balancing strategy.
    #[serde(default)]
    pub strategy: BalanceStrategy,
}

impl Default for BalanceConfig {
    fn default() -> Self {
        Self {
            strategy: BalanceStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BalanceStrategy {
    #[default]
    RoundRobin,
    P2c,
    Weighted,
    LeastConnections,
}

/// Cost tracking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    /// Whether cost tracking is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Custom model pricing overrides (model_id -> (input_cost, output_cost) per million).
    #[serde(default)]
    pub pricing_overrides: HashMap<String, PricingOverride>,
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pricing_overrides: HashMap::new(),
        }
    }
}

/// Pricing override for a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingOverride {
    pub input_cost_per_million: f64,
    pub output_cost_per_million: f64,
}

impl GatewayConfig {
    /// Load configuration from a YAML file.
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// Parse configuration from a YAML string.
    pub fn from_str(yaml: &str) -> Result<Self, ConfigError> {
        let config: GatewayConfig = serde_yaml::from_str(yaml)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration for logical consistency.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.providers.is_empty() {
            return Err(ConfigError::Validation(
                "at least one provider must be configured".to_string(),
            ));
        }

        for provider in &self.providers {
            if provider.name.is_empty() {
                return Err(ConfigError::Validation(
                    "provider name cannot be empty".to_string(),
                ));
            }
            if provider.base_url.is_empty() {
                return Err(ConfigError::Validation(format!(
                    "provider '{}' must have a base_url",
                    provider.name
                )));
            }
        }

        if self.cache.max_entries == 0 {
            return Err(ConfigError::Validation(
                "cache max_entries must be > 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Resolve a model string like "openai/gpt-4o" to (provider_name, model_id).
    pub fn resolve_model<'a>(&'a self, model_str: &'a str) -> Option<(&'a ProviderConfig, &'a str)> {
        if let Some((provider_name, model_id)) = model_str.split_once('/') {
            self.providers
                .iter()
                .find(|p| p.name == provider_name && p.enabled)
                .map(|p| (p, model_id))
        } else {
            // No prefix — find first provider that has this model
            for provider in &self.providers {
                if !provider.enabled {
                    continue;
                }
                if provider.models.iter().any(|m| m.id == model_str) {
                    return Some((provider, model_str));
                }
            }
            None
        }
    }

    /// Get all enabled providers that support a given model.
    pub fn providers_for_model(&self, model_id: &str) -> Vec<&ProviderConfig> {
        self.providers
            .iter()
            .filter(|p| p.enabled && p.models.iter().any(|m| m.id == model_id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_yaml() -> &'static str {
        r#"
listen: "127.0.0.1:9090"
providers:
  - name: openai
    base_url: https://api.openai.com
    api_key: sk-test
    auth_method: bearer
    models:
      - id: gpt-4o
        input_cost_per_million: 5.0
        output_cost_per_million: 15.0
        context_window: 128000
      - id: gpt-4o-mini
        input_cost_per_million: 0.15
        output_cost_per_million: 0.6
    weight: 80
  - name: anthropic
    base_url: https://api.anthropic.com
    api_key: sk-ant-test
    auth_method: api_key
    models:
      - id: claude-3-5-sonnet
        input_cost_per_million: 3.0
        output_cost_per_million: 15.0
    weight: 60
cache:
  enabled: true
  ttl_seconds: 3600
  max_entries: 5000
rate_limit:
  enabled: true
  requests_per_minute: 100
  tokens_per_minute: 50000
  max_spend_per_day: 10.0
  scope: per_key
balance:
  strategy: p2c
cost:
  enabled: true
"#
    }

    #[test]
    fn test_parse_full_config() {
        let config = GatewayConfig::from_str(sample_yaml()).unwrap();
        assert_eq!(config.listen, "127.0.0.1:9090");
        assert_eq!(config.providers.len(), 2);
        assert_eq!(config.providers[0].name, "openai");
        assert_eq!(config.providers[1].name, "anthropic");
        assert!(config.cache.enabled);
        assert_eq!(config.cache.ttl_seconds, 3600);
        assert!(config.rate_limit.enabled);
        assert_eq!(config.rate_limit.requests_per_minute, 100);
        assert_eq!(config.balance.strategy, BalanceStrategy::P2c);
    }

    #[test]
    fn test_parse_provider_models() {
        let config = GatewayConfig::from_str(sample_yaml()).unwrap();
        let openai = &config.providers[0];
        assert_eq!(openai.models.len(), 2);
        assert_eq!(openai.models[0].id, "gpt-4o");
        assert_eq!(openai.models[0].input_cost_per_million, 5.0);
        assert_eq!(openai.models[0].context_window, Some(128000));
        assert_eq!(openai.weight, 80);
    }

    #[test]
    fn test_parse_auth_methods() {
        let config = GatewayConfig::from_str(sample_yaml()).unwrap();
        assert_eq!(config.providers[0].auth_method, AuthMethod::Bearer);
        assert_eq!(config.providers[1].auth_method, AuthMethod::ApiKey);
    }

    #[test]
    fn test_validation_no_providers() {
        let yaml = r#"
listen: "0.0.0.0:8080"
providers: []
"#;
        let err = GatewayConfig::from_str(yaml).unwrap_err();
        assert!(err.to_string().contains("at least one provider"));
    }

    #[test]
    fn test_validation_empty_provider_name() {
        let yaml = r#"
providers:
  - name: ""
    base_url: https://example.com
"#;
        let err = GatewayConfig::from_str(yaml).unwrap_err();
        assert!(err.to_string().contains("provider name cannot be empty"));
    }

    #[test]
    fn test_validation_empty_base_url() {
        let yaml = r#"
providers:
  - name: test
    base_url: ""
"#;
        let err = GatewayConfig::from_str(yaml).unwrap_err();
        assert!(err.to_string().contains("must have a base_url"));
    }

    #[test]
    fn test_defaults() {
        let yaml = r#"
providers:
  - name: test
    base_url: https://example.com
"#;
        let config = GatewayConfig::from_str(yaml).unwrap();
        assert_eq!(config.listen, "0.0.0.0:8080");
        assert!(!config.cache.enabled);
        assert_eq!(config.cache.ttl_seconds, 86400);
        assert_eq!(config.cache.max_entries, 10000);
        assert!(!config.rate_limit.enabled);
        assert_eq!(config.balance.strategy, BalanceStrategy::RoundRobin);
        assert!(config.cost.enabled);
    }

    #[test]
    fn test_resolve_model_with_prefix() {
        let config = GatewayConfig::from_str(sample_yaml()).unwrap();
        let (provider, model_id) = config.resolve_model("openai/gpt-4o").unwrap();
        assert_eq!(provider.name, "openai");
        assert_eq!(model_id, "gpt-4o");
    }

    #[test]
    fn test_resolve_model_without_prefix() {
        let config = GatewayConfig::from_str(sample_yaml()).unwrap();
        let (provider, model_id) = config.resolve_model("gpt-4o").unwrap();
        assert_eq!(provider.name, "openai");
        assert_eq!(model_id, "gpt-4o");
    }

    #[test]
    fn test_resolve_model_unknown() {
        let config = GatewayConfig::from_str(sample_yaml()).unwrap();
        assert!(config.resolve_model("unknown/model").is_none());
    }

    #[test]
    fn test_providers_for_model() {
        let config = GatewayConfig::from_str(sample_yaml()).unwrap();
        let providers = config.providers_for_model("gpt-4o");
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].name, "openai");
    }

    #[test]
    fn test_disabled_provider_skipped() {
        let yaml = r#"
providers:
  - name: openai
    base_url: https://api.openai.com
    enabled: false
    models:
      - id: gpt-4o
  - name: backup
    base_url: https://backup.com
    models:
      - id: gpt-4o
"#;
        let config = GatewayConfig::from_str(yaml).unwrap();
        let (provider, _) = config.resolve_model("gpt-4o").unwrap();
        assert_eq!(provider.name, "backup");
    }

    #[test]
    fn test_rate_limit_scope_parsing() {
        let yaml = r#"
providers:
  - name: test
    base_url: https://example.com
rate_limit:
  scope: global
"#;
        let config = GatewayConfig::from_str(yaml).unwrap();
        assert_eq!(config.rate_limit.scope, RateLimitScope::Global);
    }

    #[test]
    fn test_cost_pricing_overrides() {
        let yaml = r#"
providers:
  - name: test
    base_url: https://example.com
cost:
  enabled: true
  pricing_overrides:
    custom-model:
      input_cost_per_million: 1.0
      output_cost_per_million: 2.0
"#;
        let config = GatewayConfig::from_str(yaml).unwrap();
        let override_ = config.cost.pricing_overrides.get("custom-model").unwrap();
        assert_eq!(override_.input_cost_per_million, 1.0);
        assert_eq!(override_.output_cost_per_million, 2.0);
    }

    #[test]
    fn test_cache_zero_max_entries_invalid() {
        let yaml = r#"
providers:
  - name: test
    base_url: https://example.com
cache:
  max_entries: 0
"#;
        let err = GatewayConfig::from_str(yaml).unwrap_err();
        assert!(err.to_string().contains("max_entries must be > 0"));
    }

    #[test]
    fn test_expand_env_with_braces() {
        std::env::set_var("LLMGW_TEST_KEY_1", "sk-secret-123");
        assert_eq!(expand_env("${LLMGW_TEST_KEY_1}"), "sk-secret-123");
        std::env::remove_var("LLMGW_TEST_KEY_1");
    }

    #[test]
    fn test_expand_env_without_braces() {
        std::env::set_var("LLMGW_TEST_KEY_2", "sk-secret-456");
        assert_eq!(expand_env("$LLMGW_TEST_KEY_2"), "sk-secret-456");
        std::env::remove_var("LLMGW_TEST_KEY_2");
    }

    #[test]
    fn test_expand_env_missing_var() {
        assert_eq!(
            expand_env("${LLMGW_NONEXISTENT_VAR}"),
            "${LLMGW_NONEXISTENT_VAR}"
        );
    }

    #[test]
    fn test_expand_env_literal_value() {
        assert_eq!(expand_env("sk-plain-key"), "sk-plain-key");
    }

    #[test]
    fn test_resolve_api_key() {
        std::env::set_var("LLMGW_TEST_KEY_3", "resolved-key");
        let config = GatewayConfig::from_str(
            r#"
providers:
  - name: test
    base_url: https://example.com
    api_key: "${LLMGW_TEST_KEY_3}"
"#,
        )
        .unwrap();
        assert_eq!(
            config.providers[0].resolve_api_key().unwrap(),
            "resolved-key"
        );
        std::env::remove_var("LLMGW_TEST_KEY_3");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = GatewayConfig::from_str(sample_yaml()).unwrap();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let config2 = GatewayConfig::from_str(&yaml).unwrap();
        assert_eq!(config.listen, config2.listen);
        assert_eq!(config.providers.len(), config2.providers.len());
    }
}
