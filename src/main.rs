//! llmgateway-rs — High-performance LLM API gateway proxy.
//!
//! A single-binary reverse proxy that routes requests to multiple LLM providers
//! (OpenAI, Anthropic, Google, etc.) with caching, rate limiting, load balancing,
//! and cost tracking.

use axum::{
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use llmgateway_rs::config::GatewayConfig;
use llmgateway_rs::proxy::{self, GatewayState};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

/// LLM Gateway — route, cache, rate-limit, and track costs for LLM API calls.
#[derive(Parser, Debug)]
#[command(name = "llmgateway-rs", version, about)]
struct Cli {
    /// Path to the configuration YAML file.
    #[arg(short, long, default_value = "config.yaml")]
    config: PathBuf,

    /// Override listen address (e.g., "0.0.0.0:8080").
    #[arg(short, long)]
    listen: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Load configuration
    let config = GatewayConfig::from_file(&cli.config)?;
    let listen_addr = cli
        .listen
        .unwrap_or_else(|| config.listen.clone());

    tracing::info!(
        "Starting llmgateway-rs v{} on {}",
        env!("CARGO_PKG_VERSION"),
        listen_addr
    );
    tracing::info!(
        "Providers: {} configured, Balance: {:?}",
        config.providers.len(),
        config.balance.strategy
    );
    tracing::info!(
        "Cache: {}, Rate limit: {}, Cost tracking: {}",
        if config.cache.enabled { "ON" } else { "OFF" },
        if config.rate_limit.enabled { "ON" } else { "OFF" },
        if config.cost.enabled { "ON" } else { "OFF" },
    );

    // Build shared state
    let state = GatewayState::from_config(config);

    // Build router (10MB body limit to prevent memory exhaustion)
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/health", get(handle_health))
        .route("/stats", get(handle_stats))
        .route("/cache/purge", post(handle_cache_purge))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .with_state(state);

    // Start server
    let addr: SocketAddr = listen_addr.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Gateway ready — listening on {}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Gateway shut down gracefully");
    Ok(())
}

/// Handle POST /v1/chat/completions
async fn handle_chat(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    body: String,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    // Extract API key from Authorization header
    let api_key = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim_start_matches("Bearer ").to_string())
        .unwrap_or_else(|| "anonymous".to_string());

    match proxy::handle_chat_request(&state, &body, &api_key).await {
        Ok(response) => {
            let mut result = response.body;
            // Add gateway metadata headers as JSON fields (optional)
            if let Some(obj) = result.as_object_mut() {
                let mut gateway_meta = serde_json::Map::new();
                gateway_meta.insert(
                    "cache_hit".to_string(),
                    serde_json::Value::Bool(response.cache_hit),
                );
                gateway_meta.insert(
                    "provider".to_string(),
                    serde_json::Value::String(response.provider),
                );
                gateway_meta.insert(
                    "latency_ms".to_string(),
                    serde_json::json!(response.latency_ms),
                );
                if let Some(cost) = &response.cost {
                    gateway_meta.insert(
                        "cost_usd".to_string(),
                        serde_json::json!(cost.total_cost),
                    );
                }
                obj.insert(
                    "_gateway".to_string(),
                    serde_json::Value::Object(gateway_meta),
                );
            }
            Ok(Json(result))
        }
        Err(e) => {
            let (status, message) = match &e {
                proxy::GatewayError::RateLimited { .. } => {
                    (StatusCode::TOO_MANY_REQUESTS, e.to_string())
                }
                proxy::GatewayError::BudgetExceeded => {
                    (StatusCode::PAYMENT_REQUIRED, e.to_string())
                }
                proxy::GatewayError::UnknownModel(_) => {
                    (StatusCode::BAD_REQUEST, e.to_string())
                }
                proxy::GatewayError::NoProvider => {
                    (StatusCode::SERVICE_UNAVAILABLE, e.to_string())
                }
                proxy::GatewayError::InvalidRequest(_) => {
                    (StatusCode::BAD_REQUEST, e.to_string())
                }
                proxy::GatewayError::UpstreamError { status, .. } => {
                    (StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY), e.to_string())
                }
                proxy::GatewayError::RequestError(_) => {
                    (StatusCode::BAD_GATEWAY, e.to_string())
                }
            };
            Err((
                status,
                Json(serde_json::json!({
                    "error": {
                        "message": message,
                        "type": "gateway_error",
                    }
                })),
            ))
        }
    }
}

/// Handle GET /health
async fn handle_health(
    State(state): State<Arc<GatewayState>>,
) -> Json<serde_json::Value> {
    let health = proxy::health_check(&state);
    Json(serde_json::to_value(health).unwrap_or_default())
}

/// Handle GET /stats
async fn handle_stats(
    State(state): State<Arc<GatewayState>>,
) -> Json<serde_json::Value> {
    let stats = proxy::gateway_stats(&state);
    Json(serde_json::to_value(stats).unwrap_or_default())
}

/// Handle POST /cache/purge
async fn handle_cache_purge(
    State(state): State<Arc<GatewayState>>,
) -> Json<serde_json::Value> {
    let purged = state.cache.purge_expired();
    Json(serde_json::json!({
        "purged": purged,
        "remaining": state.cache.len(),
    }))
}

/// Graceful shutdown signal handler.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
    tracing::info!("Received shutdown signal");
}
