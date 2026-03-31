//! Cache module — exact-match request caching with LRU eviction.
//!
//! Implements SHA-256 hash-based cache keying, in-memory LRU storage
//! with TTL, and optional field-ignore for flexible cache key generation.

use lru::LruCache;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Cached response entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response headers (key-value pairs).
    pub headers: Vec<(String, String)>,
    /// Response body bytes.
    pub body: Vec<u8>,
    /// Model name that produced this response.
    pub model: String,
    /// Token usage from the response.
    pub usage: Option<CachedUsage>,
}

/// Cached token usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Internal cache entry with TTL tracking.
#[derive(Debug, Clone)]
struct CacheEntry {
    response: CachedResponse,
    created_at: Instant,
    ttl: Duration,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Cache key components used for hashing.
#[derive(Debug, Clone)]
pub struct CacheKeyComponents {
    /// Model name (e.g., "gpt-4o").
    pub model: String,
    /// Request messages as JSON value.
    pub messages: serde_json::Value,
    /// Temperature parameter.
    pub temperature: Option<f64>,
    /// Max tokens parameter.
    pub max_tokens: Option<u64>,
    /// Top-p (nucleus sampling) parameter.
    pub top_p: Option<f64>,
    /// Seed parameter for deterministic outputs.
    pub seed: Option<u64>,
    /// Optional cache namespace.
    pub cache_seed: Option<String>,
    /// Additional parameters that affect output.
    pub extra_params: BTreeMap<String, serde_json::Value>,
}

/// Generate a deterministic cache key from components.
///
/// Uses canonical JSON serialization (BTreeMap for key ordering)
/// followed by SHA-256 hashing for a fixed-size key.
pub fn compute_cache_key(components: &CacheKeyComponents, ignore_keys: &[String]) -> String {
    let mut canonical = BTreeMap::new();

    canonical.insert(
        "model".to_string(),
        serde_json::Value::String(components.model.clone()),
    );

    // Canonicalize messages
    let canonical_messages = canonicalize_json(&components.messages, ignore_keys);
    canonical.insert("messages".to_string(), canonical_messages);

    if let Some(temp) = components.temperature {
        canonical.insert(
            "temperature".to_string(),
            serde_json::json!(format!("{:.6}", temp)),
        );
    }

    if let Some(max_tokens) = components.max_tokens {
        canonical.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
    }

    if let Some(top_p) = components.top_p {
        canonical.insert(
            "top_p".to_string(),
            serde_json::json!(format!("{:.6}", top_p)),
        );
    }

    if let Some(seed) = components.seed {
        canonical.insert("seed".to_string(), serde_json::json!(seed));
    }

    if let Some(ref cache_seed) = components.cache_seed {
        canonical.insert(
            "cache_seed".to_string(),
            serde_json::Value::String(cache_seed.clone()),
        );
    }

    for (k, v) in &components.extra_params {
        if !ignore_keys.contains(k) {
            canonical.insert(k.clone(), canonicalize_json(v, ignore_keys));
        }
    }

    let json_bytes = serde_json::to_vec(&canonical).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(&json_bytes);
    let result = hasher.finalize();
    hex::encode(result)
}

/// Recursively canonicalize a JSON value by sorting object keys
/// and removing ignored keys.
fn canonicalize_json(value: &serde_json::Value, ignore_keys: &[String]) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (k, v) in map {
                if !ignore_keys.contains(k) {
                    sorted.insert(k.clone(), canonicalize_json(v, ignore_keys));
                }
            }
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(|v| canonicalize_json(v, ignore_keys)).collect())
        }
        other => other.clone(),
    }
}

/// Hex encoding utility (to avoid adding a `hex` dependency).
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

/// In-memory LRU cache with TTL support.
pub struct MemoryCache {
    store: Arc<RwLock<LruCache<String, CacheEntry>>>,
    default_ttl: Duration,
}

impl MemoryCache {
    /// Create a new cache with the given capacity and default TTL.
    pub fn new(max_entries: usize, default_ttl_seconds: u64) -> Self {
        let cap = NonZeroUsize::new(max_entries).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            store: Arc::new(RwLock::new(LruCache::new(cap))),
            default_ttl: Duration::from_secs(default_ttl_seconds),
        }
    }

    /// Look up a cached response by key.
    /// Returns `None` if the key is missing or expired.
    pub fn get(&self, key: &str) -> Option<CachedResponse> {
        let mut store = self.store.write().ok()?;
        if let Some(entry) = store.get(key) {
            if entry.is_expired() {
                store.pop(key);
                return None;
            }
            return Some(entry.response.clone());
        }
        None
    }

    /// Store a response in the cache with the default TTL.
    pub fn put(&self, key: String, response: CachedResponse) {
        self.put_with_ttl(key, response, self.default_ttl);
    }

    /// Store a response in the cache with a custom TTL.
    pub fn put_with_ttl(&self, key: String, response: CachedResponse, ttl: Duration) {
        if let Ok(mut store) = self.store.write() {
            store.put(
                key,
                CacheEntry {
                    response,
                    created_at: Instant::now(),
                    ttl,
                },
            );
        }
    }

    /// Remove a specific entry from the cache.
    pub fn remove(&self, key: &str) -> Option<CachedResponse> {
        self.store
            .write()
            .ok()
            .and_then(|mut store| store.pop(key).map(|e| e.response))
    }

    /// Get the current number of entries (including potentially expired ones).
    pub fn len(&self) -> usize {
        self.store.read().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        if let Ok(mut store) = self.store.write() {
            store.clear();
        }
    }

    /// Purge expired entries.
    pub fn purge_expired(&self) -> usize {
        let mut store = match self.store.write() {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let mut expired_keys = Vec::new();
        // Collect expired keys
        for (key, entry) in store.iter() {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }
        let count = expired_keys.len();
        for key in expired_keys {
            store.pop(&key);
        }
        count
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let len = self.len();
        CacheStats {
            entries: len,
            capacity: self
                .store
                .read()
                .map(|s| s.cap().get())
                .unwrap_or(0),
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub capacity: usize,
}

impl Clone for MemoryCache {
    fn clone(&self) -> Self {
        Self {
            store: Arc::clone(&self.store),
            default_ttl: self.default_ttl,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_response(body: &str) -> CachedResponse {
        CachedResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: body.as_bytes().to_vec(),
            model: "gpt-4o".to_string(),
            usage: Some(CachedUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        }
    }

    fn make_key_components(model: &str, msg: &str) -> CacheKeyComponents {
        CacheKeyComponents {
            model: model.to_string(),
            messages: serde_json::json!([{"role": "user", "content": msg}]),
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            seed: None,
            cache_seed: None,
            extra_params: BTreeMap::new(),
        }
    }

    #[test]
    fn test_cache_key_deterministic() {
        let comp = make_key_components("gpt-4o", "hello");
        let key1 = compute_cache_key(&comp, &[]);
        let key2 = compute_cache_key(&comp, &[]);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_different_models() {
        let comp1 = make_key_components("gpt-4o", "hello");
        let comp2 = make_key_components("gpt-4o-mini", "hello");
        assert_ne!(compute_cache_key(&comp1, &[]), compute_cache_key(&comp2, &[]));
    }

    #[test]
    fn test_cache_key_different_messages() {
        let comp1 = make_key_components("gpt-4o", "hello");
        let comp2 = make_key_components("gpt-4o", "goodbye");
        assert_ne!(compute_cache_key(&comp1, &[]), compute_cache_key(&comp2, &[]));
    }

    #[test]
    fn test_cache_key_with_seed() {
        let mut comp1 = make_key_components("gpt-4o", "hello");
        let mut comp2 = make_key_components("gpt-4o", "hello");
        comp1.seed = Some(42);
        comp2.seed = Some(43);
        assert_ne!(compute_cache_key(&comp1, &[]), compute_cache_key(&comp2, &[]));
    }

    #[test]
    fn test_cache_key_with_cache_seed() {
        let mut comp1 = make_key_components("gpt-4o", "hello");
        let mut comp2 = make_key_components("gpt-4o", "hello");
        comp1.cache_seed = Some("ns1".to_string());
        comp2.cache_seed = Some("ns2".to_string());
        assert_ne!(compute_cache_key(&comp1, &[]), compute_cache_key(&comp2, &[]));
    }

    #[test]
    fn test_cache_key_ignore_keys() {
        let mut comp1 = make_key_components("gpt-4o", "hello");
        let mut comp2 = make_key_components("gpt-4o", "hello");
        comp1.extra_params.insert("stream".to_string(), serde_json::json!(true));
        comp2.extra_params.insert("stream".to_string(), serde_json::json!(false));

        // Without ignore: different keys
        assert_ne!(compute_cache_key(&comp1, &[]), compute_cache_key(&comp2, &[]));

        // With ignore: same keys
        let ignore = vec!["stream".to_string()];
        assert_eq!(
            compute_cache_key(&comp1, &ignore),
            compute_cache_key(&comp2, &ignore)
        );
    }

    #[test]
    fn test_cache_key_temperature_precision() {
        let mut comp1 = make_key_components("gpt-4o", "hello");
        let mut comp2 = make_key_components("gpt-4o", "hello");
        comp1.temperature = Some(0.7);
        comp2.temperature = Some(0.70);
        assert_eq!(compute_cache_key(&comp1, &[]), compute_cache_key(&comp2, &[]));
    }

    #[test]
    fn test_cache_key_json_key_order_irrelevant() {
        let comp1 = CacheKeyComponents {
            model: "gpt-4o".to_string(),
            messages: serde_json::json!({"a": 1, "b": 2}),
            temperature: None,
            max_tokens: None,
            top_p: None,
            seed: None,
            cache_seed: None,
            extra_params: BTreeMap::new(),
        };
        let comp2 = CacheKeyComponents {
            model: "gpt-4o".to_string(),
            messages: serde_json::json!({"b": 2, "a": 1}),
            temperature: None,
            max_tokens: None,
            top_p: None,
            seed: None,
            cache_seed: None,
            extra_params: BTreeMap::new(),
        };
        assert_eq!(compute_cache_key(&comp1, &[]), compute_cache_key(&comp2, &[]));
    }

    #[test]
    fn test_memory_cache_put_get() {
        let cache = MemoryCache::new(100, 3600);
        let resp = make_response("test body");
        cache.put("key1".to_string(), resp.clone());
        let got = cache.get("key1").unwrap();
        assert_eq!(got.body, resp.body);
        assert_eq!(got.status, 200);
    }

    #[test]
    fn test_memory_cache_miss() {
        let cache = MemoryCache::new(100, 3600);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_memory_cache_ttl_expiration() {
        let cache = MemoryCache::new(100, 0); // 0 second TTL
        cache.put_with_ttl(
            "key1".to_string(),
            make_response("expired"),
            Duration::from_millis(1),
        );
        thread::sleep(Duration::from_millis(10));
        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_memory_cache_lru_eviction() {
        let cache = MemoryCache::new(2, 3600);
        cache.put("a".to_string(), make_response("a"));
        cache.put("b".to_string(), make_response("b"));
        cache.put("c".to_string(), make_response("c")); // evicts "a"
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_memory_cache_remove() {
        let cache = MemoryCache::new(100, 3600);
        cache.put("key1".to_string(), make_response("test"));
        assert!(cache.remove("key1").is_some());
        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_memory_cache_clear() {
        let cache = MemoryCache::new(100, 3600);
        cache.put("a".to_string(), make_response("a"));
        cache.put("b".to_string(), make_response("b"));
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_memory_cache_purge_expired() {
        let cache = MemoryCache::new(100, 3600);
        cache.put_with_ttl(
            "expired1".to_string(),
            make_response("e1"),
            Duration::from_millis(1),
        );
        cache.put_with_ttl(
            "expired2".to_string(),
            make_response("e2"),
            Duration::from_millis(1),
        );
        cache.put("alive".to_string(), make_response("alive"));
        thread::sleep(Duration::from_millis(10));
        let purged = cache.purge_expired();
        assert_eq!(purged, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_memory_cache_stats() {
        let cache = MemoryCache::new(50, 3600);
        cache.put("a".to_string(), make_response("a"));
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.capacity, 50);
    }

    #[test]
    fn test_memory_cache_clone_shares_state() {
        let cache1 = MemoryCache::new(100, 3600);
        let cache2 = cache1.clone();
        cache1.put("shared".to_string(), make_response("shared"));
        assert!(cache2.get("shared").is_some());
    }

    #[test]
    fn test_memory_cache_overwrite() {
        let cache = MemoryCache::new(100, 3600);
        cache.put("key".to_string(), make_response("first"));
        cache.put("key".to_string(), make_response("second"));
        let resp = cache.get("key").unwrap();
        assert_eq!(String::from_utf8_lossy(&resp.body), "second");
    }

    #[test]
    fn test_cached_response_usage() {
        let resp = make_response("test");
        let usage = resp.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex::encode([0xde, 0xad, 0xbe, 0xef]), "deadbeef");
        assert_eq!(hex::encode([0x00, 0xff]), "00ff");
    }

    #[test]
    fn test_canonicalize_nested_json() {
        let val = serde_json::json!({"z": {"b": 2, "a": 1}, "a": [3, 1]});
        let canonical = canonicalize_json(&val, &[]);
        let expected = serde_json::json!({"a": [3, 1], "z": {"a": 1, "b": 2}});
        assert_eq!(canonical, expected);
    }

    #[test]
    fn test_canonicalize_ignores_keys() {
        let val = serde_json::json!({"keep": 1, "ignore_me": 2, "also_keep": 3});
        let canonical = canonicalize_json(&val, &["ignore_me".to_string()]);
        assert!(canonical.get("ignore_me").is_none());
        assert!(canonical.get("keep").is_some());
    }
}
