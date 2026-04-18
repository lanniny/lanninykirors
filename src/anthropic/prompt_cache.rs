//! 模拟 Prompt Cache
//!
//! 这不是上游 Kiro 的真实计费缓存，而是 Anthropic 兼容层上的本地模拟：
//! - 支持显式 `cache_control: { "type": "ephemeral" }` 断点
//! - 同时自动为 system / message / block 前缀建立分层缓存
//! - 命中时减少返回给客户端的 `input_tokens`
//! - 同时返回 `cache_creation_input_tokens` / `cache_read_input_tokens`
//! - 冷启动首次写入不再对外暴露 `cache_creation_input_tokens`，避免仅写不读时放大计费

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use chrono::Utc;
use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::token;

use super::converter::map_model;
use super::types::{Message, MessagesRequest, SystemMessage};

/// 缓存条目 TTL
const PROMPT_CACHE_TTL: Duration = Duration::from_secs(60 * 60 * 24);
/// 命中热条目时，间隔一段时间刷新一次时间戳，避免活跃缓存被当成旧条目淘汰。
const PROMPT_CACHE_TOUCH_INTERVAL: Duration = Duration::from_secs(60 * 10);
/// 内存中最多保留的 prompt cache 条目数
const MAX_PROMPT_CACHE_ENTRIES: usize = 16_384;
/// 大文本内部额外生成前缀断点时，至少要保留这么多 tokens，避免过短前缀污染缓存。
const MIN_TEXT_PREFIX_TOKENS: u64 = 128;
/// 当本地预估过小或与上游实际输入相差过大时，不放大 cache usage，避免把伪命中放大成大额命中。
const MIN_ESTIMATED_INPUT_TOKENS_FOR_UPWARD_SCALING: i32 = 128;
const MAX_UPWARD_USAGE_SCALE_FACTOR: f64 = 8.0;
/// 对大文本内部做“近尾部”前缀采样，以便尾巴变化时依然能高命中。
const TEXT_PREFIX_TAIL_DELTAS: &[u64] = &[
    16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
];

/// 模拟 prompt cache 统计
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptCacheUsage {
    /// 最终计费的输入 tokens（命中 cache 后已扣减）
    pub billed_input_tokens: i32,
    /// 本次写入 cache 的 tokens
    pub cache_creation_input_tokens: i32,
    /// 本次从 cache 读取命中的 tokens
    pub cache_read_input_tokens: i32,
}

impl PromptCacheUsage {
    pub fn with_billed_input_tokens(mut self, billed_input_tokens: i32) -> Self {
        self.billed_input_tokens = billed_input_tokens.max(1);
        self
    }

    pub fn billed_input_tokens_for(&self, input_tokens: i32) -> i32 {
        (input_tokens - self.cache_read_input_tokens).max(1)
    }

    pub fn with_actual_input_tokens(&self, input_tokens: i32) -> Self {
        let input_tokens = input_tokens.max(1);
        let estimated_total_input_tokens =
            (self.billed_input_tokens + self.cache_read_input_tokens).max(1);

        if input_tokens <= estimated_total_input_tokens {
            return self
                .clone()
                .with_billed_input_tokens(self.billed_input_tokens_for(input_tokens));
        }

        let scale = input_tokens as f64 / estimated_total_input_tokens as f64;
        if self.cache_read_input_tokens <= 0
            || estimated_total_input_tokens < MIN_ESTIMATED_INPUT_TOKENS_FOR_UPWARD_SCALING
            || scale > MAX_UPWARD_USAGE_SCALE_FACTOR
        {
            return self
                .clone()
                .with_billed_input_tokens(self.billed_input_tokens_for(input_tokens));
        }

        let cache_read_input_tokens =
            ((self.cache_read_input_tokens as f64) * scale).round() as i32;
        let cache_creation_input_tokens =
            ((self.cache_creation_input_tokens as f64) * scale).round() as i32;
        let cache_read_input_tokens = cache_read_input_tokens.clamp(0, input_tokens);
        let cache_creation_input_tokens = cache_creation_input_tokens
            .clamp(0, input_tokens.saturating_sub(cache_read_input_tokens));

        Self {
            billed_input_tokens: (input_tokens - cache_read_input_tokens).max(1),
            cache_creation_input_tokens,
            cache_read_input_tokens,
        }
    }

    pub fn hit_rate_for(&self, input_tokens: i32) -> f64 {
        if input_tokens <= 0 {
            return 0.0;
        }
        self.cache_read_input_tokens as f64 / input_tokens as f64
    }
}

#[derive(Debug, Clone)]
struct PromptCacheEntry {
    token_count: i32,
    cached_at: i64,
}

#[derive(Debug, Default)]
struct PersistenceWorkerState {
    pending: Option<HashMap<String, PromptCacheEntry>>,
    shutdown: bool,
}

#[derive(Debug)]
struct PromptCachePersistence {
    path: PathBuf,
    state: Arc<(Mutex<PersistenceWorkerState>, Condvar)>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl PromptCachePersistence {
    fn new(path: PathBuf) -> Self {
        let state = Arc::new((
            Mutex::new(PersistenceWorkerState::default()),
            Condvar::new(),
        ));
        let worker_state = Arc::clone(&state);
        let worker_path = path.clone();
        let worker = std::thread::Builder::new()
            .name("prompt-cache-persist".to_string())
            .spawn(move || Self::run(worker_path, worker_state))
            .ok();

        if worker.is_none() {
            tracing::warn!("启动 prompt cache 持久化线程失败，后续将仅在退出时尽力落盘");
        }

        Self {
            path,
            state,
            worker: Mutex::new(worker),
        }
    }

    fn schedule_save(&self, entries: HashMap<String, PromptCacheEntry>) {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock();
        state.pending = Some(entries);
        cvar.notify_one();
    }

    fn shutdown(&self, final_entries: HashMap<String, PromptCacheEntry>) {
        let fallback_entries = final_entries.clone();
        {
            let (lock, cvar) = &*self.state;
            let mut state = lock.lock();
            state.pending = Some(final_entries);
            state.shutdown = true;
            cvar.notify_one();
        }

        if let Some(worker) = self.worker.lock().take() {
            if let Err(error) = worker.join() {
                tracing::warn!("等待 prompt cache 持久化线程退出失败: {:?}", error);
            }
        } else {
            save_entries_to_path(&self.path, &fallback_entries);
        }
    }

    fn run(path: PathBuf, state: Arc<(Mutex<PersistenceWorkerState>, Condvar)>) {
        loop {
            let entries = {
                let (lock, cvar) = &*state;
                let mut worker_state = lock.lock();
                while worker_state.pending.is_none() && !worker_state.shutdown {
                    cvar.wait(&mut worker_state);
                }

                if worker_state.pending.is_none() && worker_state.shutdown {
                    return;
                }

                worker_state.pending.take()
            };

            if let Some(entries) = entries {
                save_entries_to_path(&path, &entries);
            }

            let should_exit = {
                let (lock, _) = &*state;
                let worker_state = lock.lock();
                worker_state.shutdown && worker_state.pending.is_none()
            };

            if should_exit {
                return;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedPromptCacheEntry {
    token_count: i32,
    cached_at: i64,
}

#[derive(Debug, Clone)]
struct PromptCacheCandidate {
    key: String,
    token_count: i32,
}

/// 本地 prompt cache 管理器
#[derive(Debug)]
pub struct PromptCache {
    entries: Mutex<HashMap<String, PromptCacheEntry>>,
    persistence: Option<PromptCachePersistence>,
}

impl Default for PromptCache {
    fn default() -> Self {
        Self::new(None)
    }
}

impl PromptCache {
    pub fn new(cache_path: Option<PathBuf>) -> Self {
        let entries = Self::load_entries(cache_path.as_deref());
        Self {
            entries: Mutex::new(entries),
            persistence: cache_path.map(PromptCachePersistence::new),
        }
    }

    /// 评估请求的 prompt cache 命中情况，并在需要时写入新的缓存断点。
    ///
    /// 规则：
    /// - 显式 `ephemeral` 断点优先保留
    /// - 自动为 system / message / block 前缀建立候选断点，以提高命中率
    /// - 读取：使用命中的最长前缀断点
    /// - 写入：将本次请求中所有候选断点写入缓存，以便下次命中更长前缀
    /// - 冷启动首次写入不计入 `cache_creation_input_tokens`
    /// - 计费：`input_tokens - cache_read_input_tokens`
    pub fn resolve_usage(&self, req: &MessagesRequest, base_input_tokens: i32) -> PromptCacheUsage {
        let candidates = collect_candidates(req);
        if candidates.is_empty() {
            return PromptCacheUsage::default().with_billed_input_tokens(base_input_tokens);
        }

        let (longest_hit, longest_written, snapshot_to_persist) = {
            let mut entries = self.entries.lock();
            let mut changed = prune_expired(&mut entries);
            let now = Utc::now().timestamp();

            let mut longest_hit = 0;
            for candidate in candidates.iter().rev() {
                let Some(entry) = entries.get_mut(&candidate.key) else {
                    continue;
                };

                longest_hit = candidate.token_count;
                changed |= refresh_entry_timestamp_if_stale(entry, now);
                break;
            }

            let mut longest_written = longest_hit;
            for candidate in &candidates {
                let already_cached = entries.contains_key(&candidate.key);
                if !already_cached {
                    longest_written = longest_written.max(candidate.token_count);
                    insert_entry(&mut entries, candidate.key.clone(), candidate.token_count);
                    changed = true;
                }
            }

            let snapshot_to_persist = changed.then(|| entries.clone());
            (longest_hit, longest_written, snapshot_to_persist)
        };

        if let Some(entries) = snapshot_to_persist {
            self.save_entries(entries);
        }

        let cache_creation_input_tokens = if longest_hit > 0 {
            (longest_written - longest_hit).max(0)
        } else {
            0
        };

        let cache_read_input_tokens = longest_hit.max(0);
        let billed_input_tokens = (base_input_tokens - cache_read_input_tokens).max(1);

        PromptCacheUsage {
            billed_input_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
        }
    }

    fn load_entries(cache_path: Option<&Path>) -> HashMap<String, PromptCacheEntry> {
        let Some(path) = cache_path else {
            return HashMap::new();
        };

        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(_) => return HashMap::new(),
        };

        let persisted: HashMap<String, PersistedPromptCacheEntry> =
            match serde_json::from_str(&content) {
                Ok(entries) => entries,
                Err(error) => {
                    tracing::warn!("解析 prompt cache 持久化文件失败，将忽略: {}", error);
                    return HashMap::new();
                }
            };

        let now = Utc::now().timestamp();
        persisted
            .into_iter()
            .filter_map(|(key, value)| {
                if now - value.cached_at >= PROMPT_CACHE_TTL.as_secs() as i64 {
                    return None;
                }

                Some((
                    key,
                    PromptCacheEntry {
                        token_count: value.token_count,
                        cached_at: value.cached_at,
                    },
                ))
            })
            .collect()
    }

    fn save_entries(&self, entries: HashMap<String, PromptCacheEntry>) {
        let Some(persistence) = &self.persistence else {
            return;
        };
        persistence.schedule_save(entries);
    }
}

impl Drop for PromptCache {
    fn drop(&mut self) {
        let Some(persistence) = &self.persistence else {
            return;
        };

        let entries = self.entries.lock().clone();
        persistence.shutdown(entries);
    }
}

fn insert_entry(entries: &mut HashMap<String, PromptCacheEntry>, key: String, token_count: i32) {
    entries.insert(
        key,
        PromptCacheEntry {
            token_count,
            cached_at: Utc::now().timestamp(),
        },
    );

    if entries.len() <= MAX_PROMPT_CACHE_ENTRIES {
        return;
    }

    if let Some(oldest_key) = entries
        .iter()
        .min_by_key(|(_, entry)| entry.cached_at)
        .map(|(key, _)| key.clone())
    {
        entries.remove(&oldest_key);
    }
}

fn prune_expired(entries: &mut HashMap<String, PromptCacheEntry>) -> bool {
    let before_len = entries.len();
    let now = Utc::now().timestamp();
    entries.retain(|_, entry| now - entry.cached_at < PROMPT_CACHE_TTL.as_secs() as i64);
    before_len != entries.len()
}

fn refresh_entry_timestamp_if_stale(entry: &mut PromptCacheEntry, now: i64) -> bool {
    if now - entry.cached_at < PROMPT_CACHE_TOUCH_INTERVAL.as_secs() as i64 {
        return false;
    }

    entry.cached_at = now;
    true
}

fn collect_candidates(req: &MessagesRequest) -> Vec<PromptCacheCandidate> {
    let mut candidates_by_key = HashMap::new();

    if let Some(system) = &req.system {
        for index in 0..system.len() {
            let partial_system = system[..=index].to_vec();
            record_candidate(
                &mut candidates_by_key,
                build_candidate(req, Some(partial_system), Vec::new()),
            );

            for text_prefix in collect_text_prefixes(&system[index].text) {
                let mut partial_system = system[..=index].to_vec();
                if let Some(current) = partial_system.last_mut() {
                    current.text = text_prefix;
                }
                record_candidate(
                    &mut candidates_by_key,
                    build_candidate(req, Some(partial_system), Vec::new()),
                );
            }
        }
    }

    for msg_index in 0..req.messages.len() {
        let msg = &req.messages[msg_index];
        let base_messages = req.messages[..msg_index].to_vec();

        match &msg.content {
            Value::Array(blocks) => {
                for block_index in 0..blocks.len() {
                    let mut partial_messages = base_messages.clone();
                    let partial_current = Message {
                        role: msg.role.clone(),
                        content: Value::Array(blocks[..=block_index].to_vec()),
                    };
                    partial_messages.push(partial_current);
                    record_candidate(
                        &mut candidates_by_key,
                        build_candidate(req, req.system.clone(), partial_messages),
                    );

                    if let Some(text) = blocks[block_index]
                        .get("text")
                        .and_then(|value| value.as_str())
                    {
                        for text_prefix in collect_text_prefixes(text) {
                            let mut partial_blocks = blocks[..=block_index].to_vec();
                            if let Some(last_block) = partial_blocks.last_mut() {
                                apply_text_prefix(last_block, text_prefix);
                            }

                            let mut partial_messages = base_messages.clone();
                            partial_messages.push(Message {
                                role: msg.role.clone(),
                                content: Value::Array(partial_blocks),
                            });
                            record_candidate(
                                &mut candidates_by_key,
                                build_candidate(req, req.system.clone(), partial_messages),
                            );
                        }
                    }
                }
            }
            Value::String(text) => {
                let mut partial_messages = base_messages;
                partial_messages.push(msg.clone());
                record_candidate(
                    &mut candidates_by_key,
                    build_candidate(req, req.system.clone(), partial_messages),
                );

                for text_prefix in collect_text_prefixes(text) {
                    let mut partial_messages = req.messages[..msg_index].to_vec();
                    partial_messages.push(Message {
                        role: msg.role.clone(),
                        content: Value::String(text_prefix),
                    });
                    record_candidate(
                        &mut candidates_by_key,
                        build_candidate(req, req.system.clone(), partial_messages),
                    );
                }
            }
            _ => {
                let mut partial_messages = base_messages;
                partial_messages.push(msg.clone());
                record_candidate(
                    &mut candidates_by_key,
                    build_candidate(req, req.system.clone(), partial_messages),
                );
            }
        }
    }

    let mut candidates = candidates_by_key.into_values().collect::<Vec<_>>();
    candidates.sort_by_key(|candidate| candidate.token_count);
    candidates
}

fn record_candidate(
    candidates_by_key: &mut HashMap<String, PromptCacheCandidate>,
    candidate: PromptCacheCandidate,
) {
    match candidates_by_key.get(&candidate.key) {
        Some(existing) if existing.token_count >= candidate.token_count => {}
        _ => {
            candidates_by_key.insert(candidate.key.clone(), candidate);
        }
    }
}

fn build_candidate(
    req: &MessagesRequest,
    system: Option<Vec<SystemMessage>>,
    messages: Vec<Message>,
) -> PromptCacheCandidate {
    let key_source = json!({
        "scope": prompt_cache_scope(req),
        "model": prompt_cache_model(req),
        "system": system,
        "messages": messages,
        "tools": req.tools,
    });
    let key = sha256_hex(&canonical_json(&key_source));
    let token_count = token::count_all_tokens_local(system, messages, req.tools.clone())
        .min(i32::MAX as u64) as i32;

    PromptCacheCandidate { key, token_count }
}

fn collect_text_prefixes(text: &str) -> Vec<String> {
    let full_tokens = token::count_tokens(text);
    if full_tokens < MIN_TEXT_PREFIX_TOKENS {
        return Vec::new();
    }

    let mut prefixes = Vec::new();
    let mut seen_lengths = HashSet::new();

    for delta in TEXT_PREFIX_TAIL_DELTAS {
        let target_tokens = full_tokens.saturating_sub(*delta);
        if target_tokens < MIN_TEXT_PREFIX_TOKENS || target_tokens >= full_tokens {
            continue;
        }

        if let Some(prefix) = truncate_text_to_tokens(text, target_tokens) {
            if prefix.len() < text.len() && seen_lengths.insert(prefix.len()) {
                prefixes.push(prefix);
            }
        }
    }

    prefixes
}

fn truncate_text_to_tokens(text: &str, target_tokens: u64) -> Option<String> {
    if text.is_empty() || target_tokens == 0 {
        return None;
    }

    let mut boundaries = text
        .char_indices()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    boundaries.push(text.len());

    let mut left = 0usize;
    let mut right = boundaries.len() - 1;
    let mut best_end = 0usize;

    while left <= right {
        let mid = left + (right - left) / 2;
        let end = boundaries[mid];
        let token_count = token::count_tokens(&text[..end]);

        if token_count <= target_tokens {
            best_end = end;
            left = mid.saturating_add(1);
        } else if mid == 0 {
            break;
        } else {
            right = mid - 1;
        }
    }

    (best_end > 0).then(|| text[..best_end].to_string())
}

fn apply_text_prefix(block: &mut Value, text_prefix: String) {
    if let Some(object) = block.as_object_mut() {
        object.insert("text".to_string(), Value::String(text_prefix));
    }
}

pub(crate) fn prompt_cache_scope(req: &MessagesRequest) -> String {
    let Some(raw_user_id) = req
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.user_id.as_ref())
    else {
        return "anonymous".to_string();
    };

    if let Ok(value) = serde_json::from_str::<Value>(raw_user_id) {
        if let Some(account_uuid) = value.get("account_uuid").and_then(|v| v.as_str()) {
            return format!("account:{account_uuid}");
        }
        if let Some(user_id) = value.get("user_id").and_then(|v| v.as_str()) {
            return format!("user:{user_id}");
        }
    }

    if let Some((prefix, _)) = raw_user_id.split_once("__session_") {
        return prefix.to_string();
    }

    raw_user_id.to_string()
}

pub(crate) fn prompt_cache_model(req: &MessagesRequest) -> String {
    map_model(&req.model).unwrap_or_else(|| req.model.clone())
}

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_string(value).unwrap_or_default()
        }
        Value::Array(items) => {
            let parts = items.iter().map(canonical_json).collect::<Vec<_>>();
            format!("[{}]", parts.join(","))
        }
        Value::Object(map) => canonical_object(map),
    }
}

fn canonical_object(map: &Map<String, Value>) -> String {
    let mut entries = map.iter().collect::<Vec<_>>();
    entries.sort_by(|(left, _), (right, _)| left.cmp(right));

    let parts = entries
        .into_iter()
        .map(|(key, value)| {
            format!(
                "{}:{}",
                serde_json::to_string(key).unwrap_or_default(),
                canonical_json(value)
            )
        })
        .collect::<Vec<_>>();

    format!("{{{}}}", parts.join(","))
}

fn save_entries_to_path(path: &Path, entries: &HashMap<String, PromptCacheEntry>) {
    if let Some(parent) = path.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            tracing::warn!("创建 prompt cache 目录失败: {}", error);
            return;
        }
    }

    let persisted = entries
        .iter()
        .map(|(key, value)| {
            (
                key.clone(),
                PersistedPromptCacheEntry {
                    token_count: value.token_count,
                    cached_at: value.cached_at,
                },
            )
        })
        .collect::<HashMap<_, _>>();

    match serde_json::to_string_pretty(&persisted) {
        Ok(json) => {
            let tmp_path = path.with_extension("json.tmp");
            if let Err(error) = std::fs::write(&tmp_path, json) {
                tracing::warn!("写入 prompt cache 持久化文件失败: {}", error);
                return;
            }

            if let Err(error) = replace_existing_file(&tmp_path, path) {
                tracing::warn!("替换 prompt cache 持久化文件失败: {}", error);
                let _ = std::fs::remove_file(&tmp_path);
            }
        }
        Err(error) => tracing::warn!("序列化 prompt cache 持久化数据失败: {}", error),
    }
}

fn replace_existing_file(tmp_path: &Path, path: &Path) -> std::io::Result<()> {
    match std::fs::rename(tmp_path, path) {
        Ok(()) => Ok(()),
        Err(rename_error) if path.exists() => {
            std::fs::remove_file(path)?;
            std::fs::rename(tmp_path, path).map_err(|second_error| {
                std::io::Error::new(
                    second_error.kind(),
                    format!(
                        "{}; fallback replace after remove failed: {}",
                        rename_error, second_error
                    ),
                )
            })
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::types::{CacheControl, Message, Metadata, SystemMessage};
    use std::path::PathBuf;

    fn ephemeral() -> CacheControl {
        CacheControl {
            cache_type: "ephemeral".to_string(),
        }
    }

    fn base_request() -> MessagesRequest {
        MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: json!([
                    {"type": "text", "text": "你是一个很强的 Rust 助手", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "请帮我修复这个 bug"}
                ]),
            }],
            stream: false,
            system: Some(vec![SystemMessage {
                text: "系统前缀".to_string(),
                cache_control: Some(ephemeral()),
            }]),
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: Some(Metadata {
                user_id: Some("user_a__session_11111111-1111-1111-1111-111111111111".to_string()),
            }),
        }
    }

    #[test]
    fn first_request_writes_cache_but_does_not_bill_cache_creation() {
        let cache = PromptCache::default();
        let req = base_request();
        let base_input_tokens = token::count_all_tokens_local(
            req.system.clone(),
            req.messages.clone(),
            req.tools.clone(),
        ) as i32;

        let usage = cache.resolve_usage(&req, base_input_tokens);

        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.billed_input_tokens, base_input_tokens);
    }

    #[test]
    fn second_request_hits_longest_cached_prefix() {
        let cache = PromptCache::default();
        let req = base_request();
        let base_input_tokens = token::count_all_tokens_local(
            req.system.clone(),
            req.messages.clone(),
            req.tools.clone(),
        ) as i32;

        let first = cache.resolve_usage(&req, base_input_tokens);
        let second = cache.resolve_usage(&req, base_input_tokens);

        assert_eq!(first.cache_creation_input_tokens, 0);
        assert_eq!(second.cache_creation_input_tokens, 0);
        assert!(second.cache_read_input_tokens > 0);
        assert!(second.billed_input_tokens < base_input_tokens);
    }

    #[test]
    fn longer_prompt_reads_shorter_prefix_and_writes_delta() {
        let cache = PromptCache::default();
        let first_req = base_request();
        let first_tokens = token::count_all_tokens_local(
            first_req.system.clone(),
            first_req.messages.clone(),
            first_req.tools.clone(),
        ) as i32;
        cache.resolve_usage(&first_req, first_tokens);

        let mut second_req = first_req;
        second_req.messages = vec![Message {
            role: "user".to_string(),
            content: json!([
                {"type": "text", "text": "你是一个很强的 Rust 助手", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "这是延长后的上下文", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "请输出完整修复方案"}
            ]),
        }];
        let second_tokens = token::count_all_tokens_local(
            second_req.system.clone(),
            second_req.messages.clone(),
            second_req.tools.clone(),
        ) as i32;

        let usage = cache.resolve_usage(&second_req, second_tokens);

        assert!(usage.cache_read_input_tokens > 0);
        assert!(usage.cache_creation_input_tokens > 0);
        assert!(usage.billed_input_tokens < second_tokens);
    }

    #[test]
    fn cache_scope_uses_metadata_user_id() {
        let cache = PromptCache::default();
        let req = base_request();
        let base_input_tokens = token::count_all_tokens_local(
            req.system.clone(),
            req.messages.clone(),
            req.tools.clone(),
        ) as i32;
        cache.resolve_usage(&req, base_input_tokens);

        let mut other_user_req = base_request();
        other_user_req.metadata = Some(Metadata {
            user_id: Some("user_b__session_22222222-2222-2222-2222-222222222222".to_string()),
        });

        let usage = cache.resolve_usage(&other_user_req, base_input_tokens);

        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.cache_creation_input_tokens, 0);
    }

    #[test]
    fn same_account_different_session_still_hits_cache() {
        let cache = PromptCache::default();
        let req = base_request();
        let base_input_tokens = token::count_all_tokens_local(
            req.system.clone(),
            req.messages.clone(),
            req.tools.clone(),
        ) as i32;
        cache.resolve_usage(&req, base_input_tokens);

        let mut next_session_req = base_request();
        next_session_req.metadata = Some(Metadata {
            user_id: Some("user_a__session_33333333-3333-3333-3333-333333333333".to_string()),
        });

        let usage = cache.resolve_usage(&next_session_req, base_input_tokens);

        assert!(usage.cache_read_input_tokens > 0);
        assert!(usage.billed_input_tokens < base_input_tokens);
    }

    #[test]
    fn model_aliases_share_prompt_cache() {
        let cache = PromptCache::default();
        let mut req = base_request();
        req.model = "claude-opus-4-7".to_string();
        let base_input_tokens = token::count_all_tokens_local(
            req.system.clone(),
            req.messages.clone(),
            req.tools.clone(),
        ) as i32;
        cache.resolve_usage(&req, base_input_tokens);

        let mut alias_req = base_request();
        alias_req.model = "claude-opus-4-6".to_string();

        let usage = cache.resolve_usage(&alias_req, base_input_tokens);

        assert!(
            usage.cache_read_input_tokens > 0,
            "expected alias model to reuse prompt cache, usage={usage:?}"
        );
        assert!(usage.billed_input_tokens < base_input_tokens);
    }

    #[test]
    fn automatic_prefix_cache_can_reach_over_ninety_percent_hit_rate() {
        let cache = PromptCache::default();
        let large_prefix = "稳定前缀上下文".repeat(500);
        let req1 = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: json!([
                    {"type": "text", "text": large_prefix},
                    {"type": "text", "text": "尾部问题 A"}
                ]),
            }],
            stream: false,
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: Some(Metadata {
                user_id: Some(
                    "user_cache__session_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa".to_string(),
                ),
            }),
        };
        let req1_tokens = token::count_all_tokens_local(
            req1.system.clone(),
            req1.messages.clone(),
            req1.tools.clone(),
        ) as i32;
        cache.resolve_usage(&req1, req1_tokens);

        let req2 = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: json!([
                    {"type": "text", "text": "稳定前缀上下文".repeat(500)},
                    {"type": "text", "text": "尾部问题 B"}
                ]),
            }],
            stream: false,
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: Some(Metadata {
                user_id: Some(
                    "user_cache__session_bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb".to_string(),
                ),
            }),
        };
        let req2_tokens = token::count_all_tokens_local(
            req2.system.clone(),
            req2.messages.clone(),
            req2.tools.clone(),
        ) as i32;

        let usage = cache.resolve_usage(&req2, req2_tokens);
        let hit_rate = usage.cache_read_input_tokens as f64 / req2_tokens as f64;

        assert!(
            hit_rate > 0.9,
            "hit_rate={hit_rate}, usage={usage:?}, req2_tokens={req2_tokens}"
        );
    }

    #[test]
    fn single_large_text_message_can_still_hit_over_ninety_percent() {
        let cache = PromptCache::default();
        let large_prefix = "共享上下文".repeat(1200);
        let req1 = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: Value::String(format!("{large_prefix}\n尾部问题 A")),
            }],
            stream: false,
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: Some(Metadata {
                user_id: Some("single_text__session_aaaa".to_string()),
            }),
        };
        let req1_tokens = token::count_all_tokens_local(
            req1.system.clone(),
            req1.messages.clone(),
            req1.tools.clone(),
        ) as i32;
        cache.resolve_usage(&req1, req1_tokens);

        let req2 = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: Value::String(format!("{large_prefix}\n尾部问题 B")),
            }],
            stream: false,
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: Some(Metadata {
                user_id: Some("single_text__session_bbbb".to_string()),
            }),
        };
        let req2_tokens = token::count_all_tokens_local(
            req2.system.clone(),
            req2.messages.clone(),
            req2.tools.clone(),
        ) as i32;

        let usage = cache.resolve_usage(&req2, req2_tokens);
        let hit_rate = usage.hit_rate_for(req2_tokens);

        assert!(
            hit_rate > 0.9,
            "single text message hit_rate={hit_rate}, usage={usage:?}, req2_tokens={req2_tokens}"
        );
    }

    #[test]
    fn medium_text_with_mid_tail_edit_can_hit_over_sixty_percent() {
        let cache = PromptCache::default();
        let stable_prefix = "共享上下文".repeat(350);
        let req1 = MessagesRequest {
            model: "claude-opus-4-7".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: Value::String(format!(
                    "{stable_prefix}{}{}",
                    "第一版扩展内容".repeat(120),
                    "结尾问题 A"
                )),
            }],
            stream: false,
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: Some(Metadata {
                user_id: Some(
                    "user_medium_tail__session_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa".to_string(),
                ),
            }),
        };
        let req1_tokens = token::count_all_tokens_local(
            req1.system.clone(),
            req1.messages.clone(),
            req1.tools.clone(),
        ) as i32;
        cache.resolve_usage(&req1, req1_tokens);

        let req2 = MessagesRequest {
            model: "claude-opus-4-6".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: Value::String(format!(
                    "{stable_prefix}{}{}",
                    "第二版扩展内容".repeat(120),
                    "结尾问题 B"
                )),
            }],
            stream: false,
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: Some(Metadata {
                user_id: Some(
                    "user_medium_tail__session_bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb".to_string(),
                ),
            }),
        };
        let req2_tokens = token::count_all_tokens_local(
            req2.system.clone(),
            req2.messages.clone(),
            req2.tools.clone(),
        ) as i32;

        let usage = cache.resolve_usage(&req2, req2_tokens);
        let hit_rate = usage.cache_read_input_tokens as f64 / req2_tokens as f64;

        assert!(
            hit_rate > 0.6,
            "medium tail edit hit_rate={hit_rate}, usage={usage:?}, req2_tokens={req2_tokens}"
        );
    }

    #[test]
    fn cache_persists_to_disk_and_can_be_reloaded() {
        let cache_path = unique_test_cache_path("prompt_cache_persist");
        let _ = std::fs::remove_file(&cache_path);

        let req = base_request();
        let base_input_tokens = token::count_all_tokens_local(
            req.system.clone(),
            req.messages.clone(),
            req.tools.clone(),
        ) as i32;

        {
            let cache = PromptCache::new(Some(cache_path.clone()));
            let usage = cache.resolve_usage(&req, base_input_tokens);
            assert_eq!(usage.cache_creation_input_tokens, 0);
        }

        let reloaded = PromptCache::new(Some(cache_path.clone()));
        let usage = reloaded.resolve_usage(&req, base_input_tokens);
        assert!(usage.cache_read_input_tokens > 0);

        let _ = std::fs::remove_file(cache_path);
    }

    #[test]
    fn cache_persistence_can_replace_existing_file() {
        let cache_path = unique_test_cache_path("prompt_cache_replace");
        std::fs::write(&cache_path, "{\"legacy\":true}").expect("should seed cache file");

        let req = base_request();
        let base_input_tokens = token::count_all_tokens_local(
            req.system.clone(),
            req.messages.clone(),
            req.tools.clone(),
        ) as i32;

        {
            let cache = PromptCache::new(Some(cache_path.clone()));
            let usage = cache.resolve_usage(&req, base_input_tokens);
            assert_eq!(usage.cache_creation_input_tokens, 0);
        }

        let content = std::fs::read_to_string(&cache_path).expect("cache file should exist");
        let persisted: HashMap<String, PersistedPromptCacheEntry> =
            serde_json::from_str(&content).expect("cache file should be valid json");
        assert!(
            !persisted.is_empty(),
            "persisted entries should not be empty"
        );

        let _ = std::fs::remove_file(cache_path);
    }

    #[test]
    fn actual_input_tokens_scales_cache_usage_upward() {
        let usage = PromptCacheUsage {
            billed_input_tokens: 900,
            cache_creation_input_tokens: 80,
            cache_read_input_tokens: 100,
        };

        let scaled = usage.with_actual_input_tokens(2500);

        assert_eq!(scaled.cache_read_input_tokens, 250);
        assert_eq!(scaled.cache_creation_input_tokens, 200);
        assert_eq!(scaled.billed_input_tokens, 2250);
    }

    #[test]
    fn actual_input_tokens_does_not_scale_cache_usage_downward() {
        let usage = PromptCacheUsage {
            billed_input_tokens: 900,
            cache_creation_input_tokens: 80,
            cache_read_input_tokens: 100,
        };

        let scaled = usage.with_actual_input_tokens(500);

        assert_eq!(scaled.cache_read_input_tokens, 100);
        assert_eq!(scaled.cache_creation_input_tokens, 80);
        assert_eq!(scaled.billed_input_tokens, 400);
    }

    #[test]
    fn actual_input_tokens_does_not_scale_tiny_estimates_upward() {
        let usage = PromptCacheUsage {
            billed_input_tokens: 1,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 1,
        };

        let scaled = usage.with_actual_input_tokens(2064);

        assert_eq!(scaled.cache_read_input_tokens, 1);
        assert_eq!(scaled.cache_creation_input_tokens, 0);
        assert_eq!(scaled.billed_input_tokens, 2063);
    }

    #[test]
    fn actual_input_tokens_does_not_scale_implausible_ratio_upward() {
        let usage = PromptCacheUsage {
            billed_input_tokens: 80,
            cache_creation_input_tokens: 20,
            cache_read_input_tokens: 160,
        };

        let scaled = usage.with_actual_input_tokens(5000);

        assert_eq!(scaled.cache_read_input_tokens, 160);
        assert_eq!(scaled.cache_creation_input_tokens, 20);
        assert_eq!(scaled.billed_input_tokens, 4840);
    }

    #[test]
    fn repeated_hit_refreshes_stale_cache_timestamp() {
        let cache = PromptCache::default();
        let req = base_request();
        let base_input_tokens = token::count_all_tokens_local(
            req.system.clone(),
            req.messages.clone(),
            req.tools.clone(),
        ) as i32;
        cache.resolve_usage(&req, base_input_tokens);

        let longest_candidate = collect_candidates(&req)
            .into_iter()
            .max_by_key(|candidate| candidate.token_count)
            .expect("should have prompt cache candidates");
        let stale_cached_at =
            Utc::now().timestamp() - PROMPT_CACHE_TOUCH_INTERVAL.as_secs() as i64 - 1;
        {
            let mut entries = cache.entries.lock();
            let entry = entries
                .get_mut(&longest_candidate.key)
                .expect("candidate should be cached");
            entry.cached_at = stale_cached_at;
        }

        cache.resolve_usage(&req, base_input_tokens);

        let refreshed_cached_at = cache
            .entries
            .lock()
            .get(&longest_candidate.key)
            .expect("candidate should still exist")
            .cached_at;
        assert!(refreshed_cached_at > stale_cached_at);
    }

    fn unique_test_cache_path(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "{}_{}_{}.json",
            prefix,
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ))
    }
}
