# llmgateway-rs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.94+-orange.svg?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/Tests-151-brightgreen.svg?style=for-the-badge)](#tests)

**High-performance LLM API gateway proxy in Rust.**

Route, cache, rate-limit, and track costs across multiple LLM providers through a single unified OpenAI-compatible API endpoint. Inspired by [Helicone's ai-gateway](https://github.com/Helicone/helicone).

---

## Why This Exists

Managing multiple LLM providers is painful. Each has different:
- API formats (OpenAI, Anthropic Messages, Google Gemini)
- Auth patterns (Bearer tokens, API keys, custom headers)
- Pricing models (per-token with tiered thresholds)
- Rate limits (RPM, TPM, spend caps)

**llmgateway-rs** solves this by sitting between your app and LLM providers as a transparent reverse proxy. Point your OpenAI SDK at the gateway, prefix your model name with the provider, and everything just works.

```
Your App (OpenAI SDK)
  --> llmgateway-rs (single endpoint)
    --> OpenAI / Anthropic / Google / DeepSeek / Any OpenAI-compatible
```

---

## Features

### Core Gateway
- **Unified API**: Single OpenAI-compatible endpoint for all providers
- **Provider Translation**: Automatic request/response format conversion
- **Graceful Shutdown**: Clean SIGINT handling with in-flight request completion

### Caching
- **Exact-Match Cache**: SHA-256 hash of (model + messages + params) as cache key
- **In-Memory LRU**: No external dependencies (no Redis required)
- **TTL Support**: Per-entry expiration with configurable defaults
- **Field Ignore**: Exclude specific fields (e.g., `stream`) from cache key
- **JSON Canonicalization**: Deterministic keys regardless of JSON key ordering

### Rate Limiting
- **Token Bucket**: Per-key rate limiting with configurable capacity and refill
- **Sliding Window**: Weighted sliding window counters for smooth rate enforcement
- **Budget Tracking**: Daily dollar-spend caps per API key
- **Per-Key Isolation**: Each API key gets independent limits

### Load Balancing
- **Round Robin**: Simple sequential rotation across healthy providers
- **P2C (Power of Two Choices)**: Pick 2 random backends, choose less-loaded one
- **Weighted**: Configurable weight-based distribution
- **Least Connections**: Route to provider with fewest in-flight requests
- **Health Tracking**: EWMA latency, error rates, automatic unhealthy marking

### Cost Tracking
- **Embedded Pricing DB**: 20+ models with current per-token pricing
- **Per-Request Cost**: Automatic cost calculation from response usage data
- **Spend Aggregation**: Per-key, per-model, and global spend tracking
- **Cache Cost Multiplier**: Reduced cost for cache hits (10% of normal)
- **Custom Pricing Overrides**: Override or add model pricing via config

### Provider Support
| Provider | Format | Auth | Status |
|----------|--------|------|--------|
| OpenAI | Native | Bearer | Full |
| Anthropic | Messages API | x-api-key | Full |
| Google Gemini | generateContent | Bearer | Full |
| Azure OpenAI | OpenAI-compatible | Bearer | Full |
| DeepSeek | OpenAI-compatible | Bearer | Full |
| Any OpenAI-compatible | Passthrough | Bearer | Full |

---

## Installation

### From Source

```bash
git clone https://github.com/JSLEEKR/llmgateway-rs.git
cd llmgateway-rs
cargo build --release
```

The binary is at `target/release/llmgateway-rs` (~30MB standalone).

### Requirements

- Rust 1.94+ (2021 edition)
- No external runtime dependencies

---

## Quick Start

### 1. Create Configuration

```bash
cp config/default.yaml config.yaml
```

Edit `config.yaml` with your provider API keys:

```yaml
listen: "0.0.0.0:8080"

providers:
  - name: openai
    base_url: https://api.openai.com
    api_key: sk-your-key-here
    models:
      - id: gpt-4o
        input_cost_per_million: 2.50
        output_cost_per_million: 10.0

  - name: anthropic
    base_url: https://api.anthropic.com
    api_key: sk-ant-your-key-here
    auth_method: api_key
    models:
      - id: claude-3-5-sonnet
        input_cost_per_million: 3.0
        output_cost_per_million: 15.0

cache:
  enabled: true
  ttl_seconds: 86400
  max_entries: 10000

rate_limit:
  enabled: true
  requests_per_minute: 60
  max_spend_per_day: 50.0

balance:
  strategy: p2c

cost:
  enabled: true
```

### 2. Start the Gateway

```bash
./target/release/llmgateway-rs --config config.yaml
```

### 3. Send Requests

Use any OpenAI SDK — just change the base URL and prefix the model name:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080",
    api_key="your-gateway-key",
)

# Route to OpenAI
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Route to Anthropic (automatic format translation)
response = client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Route to Google Gemini
response = client.chat.completions.create(
    model="google/gemini-2.0-flash",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

```bash
# Or with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## API Endpoints

### POST /v1/chat/completions

Main proxy endpoint. Accepts OpenAI-format chat completion requests.

**Request**: Standard OpenAI chat completion body with provider-prefixed model name.

**Response**: OpenAI-format response with additional `_gateway` metadata:

```json
{
  "id": "chatcmpl-abc123",
  "model": "gpt-4o",
  "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
  "_gateway": {
    "cache_hit": false,
    "provider": "openai",
    "latency_ms": 450,
    "cost_usd": 0.000075
  }
}
```

### GET /health

Health check endpoint with provider status:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "providers": [
    {"name": "openai", "healthy": true, "latency_ms": 120.5, "error_rate": 0.001},
    {"name": "anthropic", "healthy": true, "latency_ms": 95.2, "error_rate": 0.0}
  ],
  "cache_entries": 1234,
  "total_requests": 50000,
  "total_spend": 42.50
}
```

### GET /stats

Detailed gateway statistics:

```json
{
  "cache_entries": 1234,
  "cache_capacity": 10000,
  "total_requests": 50000,
  "total_spend": 42.50,
  "healthy_providers": 3,
  "total_providers": 3,
  "balance_strategy": "p2c",
  "spend_by_key": {"key1": 20.0, "key2": 22.50}
}
```

### POST /cache/purge

Purge expired cache entries:

```json
{"purged": 150, "remaining": 1084}
```

---

## Configuration Reference

### Top-Level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `listen` | string | `0.0.0.0:8080` | Server listen address |
| `providers` | array | required | Upstream LLM provider configs |
| `cache` | object | disabled | Cache settings |
| `rate_limit` | object | disabled | Rate limiting settings |
| `balance` | object | round_robin | Load balancing settings |
| `cost` | object | enabled | Cost tracking settings |

### Provider Config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Provider identifier |
| `base_url` | string | required | Provider API base URL |
| `api_key` | string | none | API key for auth |
| `auth_method` | enum | `bearer` | `bearer`, `api_key`, `none` |
| `models` | array | `[]` | Available models |
| `weight` | u32 | `100` | Weight for weighted balancing |
| `enabled` | bool | `true` | Whether provider is active |
| `rate_limits` | object | none | Provider-specific rate limits |

### Cache Config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable caching |
| `ttl_seconds` | u64 | `86400` | Default TTL (24h) |
| `max_entries` | usize | `10000` | Max cache size |
| `ignore_keys` | array | `[]` | JSON fields to exclude from cache key |

### Rate Limit Config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable rate limiting |
| `requests_per_minute` | u64 | `60` | RPM per key |
| `tokens_per_minute` | u64 | `100000` | TPM per key |
| `max_spend_per_day` | f64 | `0.0` | Daily budget cap ($) |
| `scope` | enum | `per_key` | `per_key`, `per_user`, `global` |

### Balance Config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | enum | `round_robin` | `round_robin`, `p2c`, `weighted`, `least_connections` |

---

## Architecture

```
src/
  lib.rs              # Crate root — module declarations
  main.rs             # Server startup, axum routes, graceful shutdown
  config/mod.rs       # YAML config parsing and validation
  cache/mod.rs        # LRU cache with TTL, SHA-256 key generation
  ratelimit/mod.rs    # Token bucket, sliding window, budget tracker
  balance/mod.rs      # Load balancing strategies and health tracking
  cost/mod.rs         # Token pricing DB, cost calculator, spend tracker
  providers/mod.rs    # Provider detection, request/response translation
  proxy/mod.rs        # Gateway pipeline orchestration, error handling
```

### Request Pipeline

```
1. Parse Request (OpenAI-format JSON)
2. Rate Limit Check (token bucket, per API key)
3. Cache Lookup (SHA-256 hash → LRU)
   └─ HIT → return cached response (10% cost)
4. Resolve Provider (model prefix → provider config)
5. Load Balance (P2C/RR/Weighted → select backend)
6. Transform Request (OpenAI → provider-native format)
7. Forward to Upstream (reqwest HTTP client)
8. Transform Response (provider-native → OpenAI format)
9. Track Cost (usage tokens × model pricing)
10. Store in Cache (for future hits)
11. Return Response (with _gateway metadata)
```

---

## Tests

151 tests across all modules:

```
Module          Tests   Coverage
─────────────────────────────────
config          22      Config parsing, validation, model resolution, defaults, env expansion
cache           22      Key generation, LRU eviction, TTL, JSON canonicalization
ratelimit       27      Token bucket, sliding window, budget tracking
balance         26      Round robin, P2C, weighted, least connections, health
cost            22      Pricing DB, cost calculation, spend tracking, extraction
providers       21      Provider detection, auth, request/response transforms
proxy           11      Model parsing, cache keys, gateway state, health/stats
─────────────────────────────────
Total           151
```

Run all tests:

```bash
cargo test
```

Run tests for a specific module:

```bash
cargo test cache::
cargo test ratelimit::
cargo test balance::
```

---

## Load Balancing Strategies

### P2C (Power of Two Choices)

The recommended strategy for production. Randomly picks 2 providers and routes to the one with fewer in-flight requests. Avoids thundering herd problems that plague simple round-robin. Used in production by Envoy, linkerd, and Helicone.

### Round Robin

Simple sequential rotation. Good for evenly-matched providers with similar latency.

### Weighted

Routes based on configured provider weights. Use when providers have different capacities or you want to gradually shift traffic.

### Least Connections

Always routes to the provider with the fewest active requests. Optimal for heterogeneous latency profiles but can concentrate traffic.

---

## Comparison with Helicone ai-gateway

| Feature | Helicone ai-gateway | llmgateway-rs |
|---------|-------------------|---------------|
| Language | Rust | Rust |
| Cache Backend | Redis / S3 | In-Memory LRU |
| External Deps | Redis, S3, Supabase | None |
| Config | UI + env vars | Single YAML |
| Providers | 100+ | OpenAI, Anthropic, Google, Generic |
| Load Balancing | P2C, PeakEWMA | P2C, RR, Weighted, LeastConn |
| Cost Tracking | DB-backed | In-memory, embedded pricing |
| Tests | Minimal | 151 |
| Deployment | Docker + infra | Single binary |
| Binary Size | ~30MB | ~30MB |
| Use Case | Full observability | Lightweight gateway |

**Our advantages**:
- Zero external dependencies (no Redis, no database)
- Single YAML configuration file
- Embedded pricing database (20+ models, no external lookup)
- Comprehensive test suite (151 tests)
- Simpler deployment (just the binary + config)

**Helicone advantages**:
- Full observability dashboard
- 100+ provider support
- Production-proven at scale (2B+ requests)
- Persistent cache (Redis/S3)
- Advanced analytics (ClickHouse)

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure `cargo test` passes (151+ tests)
4. Ensure `cargo build` has zero warnings
5. Submit a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.
