# Visitor Preference Memory — 设计文档

日期：2026-11-11 · 分支：`feat/shopping-agent-p0p1-20261111`
实现：`app/services/commerce/preferences.py` · ORM：`app/models/database/visitor_preference.py` ·
迁移：`migrations/versions/b9d4e6f2a1c8_visitor_preferences.py`（down_revision `b7f3d2c9a4e1`）
测试：`app/tests/unit/services/commerce/test_preferences.py`（12 例）、`app/tests/unit/deploy/test_visitor_preference_migration.py`（2 例）

## 定位（本相位：仅服务层）

按访客（`guest_visitors.id`）存取电商偏好记忆：尺寸偏好、品牌倾向、预算提示等**主观意愿**。本相位交付存储 + 服务 + 迁移；**不做** conversation.py / tools.py 接线（下一波，见文末集成点）。

## 决策记录

- **不实现 langgraph `BaseStore` 子类**：本地核对（`.venv/.../langgraph/store/base/__init__.py` L708 起）`BaseStore` 是面向通用 Op 代数（Get/Put/Search + 向量 + TTL 配置）的抽象，抽象方法为 `batch/abatch`。本需求只是单键值偏好表，实现完整 Op 协议是无消费者的表面积；任务约束的真实意图是「不加新依赖」（不引入 `langgraph-checkpoint-postgres` 等）——因此直接复用应用现有 SQLAlchemy async engine/sessionmaker（`app.api.database.async_session_maker`），零新依赖。若下一波 LangGraph 节点需要 store 注入，可在本服务外包一层薄 `BaseStore` 适配器。
- **audit 同库同事务**：每次写偏好必落一行审计（`visitor_preference_audit`，动作 `save`/`forget`，含 old/new 值）。选择独立审计表（int 自增 PK）而非同表 history：主表保持单行 upsert 语义，审计 append-only；审计表 **不设 FK**（telemetry 先例 `user_facts`），访客行被清理时审计可存活用于排查。
- **JSONB**：`pref_value` 在 PostgreSQL 上是 `JSONB`（`JSON().with_variant(JSONB, "postgresql")`，ORM 与迁移同一表达式，避免 autogenerate 漂移）；SQLite（测试）退化为通用 JSON。
- **source 语义**：`inferred`（对话推断）/ `confirmed`（访客明确表达）。**`confirmed` 不可被后续 `inferred` 静默降级**——对已 confirmed 的键做 inferred 写入是无操作，返回原 confirmed 条目（测试 `test_inferred_never_silently_overwrites_confirmed`）。
- **TTL**：`save(..., ttl_seconds=N)` 写 `expires_at = now + N`；`load_active` 过滤已过期行（不删除，保留审计可回溯）。时钟可注入（默认 `utcnow`），测试用 FixedClock。

## 红线：偏好 ≠ 权限（Preference / Permission separation）

本表**只存偏好**。配额/限额/权限/策略类配置（如 `quota.monthly_spend`、`limit.cart_lines`、`permission.admin_tools`、`policy.*`）**永远不进这张表**：

- 键空间保留前缀 `quota.` / `limit.` / `permission.` / `policy.`，`PreferenceMemoryService.save` 直接 `ValueError` 拒绝（测试 `test_quota_and_permission_keys_are_rejected`）。
- 理由：这是 agent 可写的存储；强制类配置（如 `COMMERCE_SPEND_LIMIT_*_CENTS`，见 `.env.example`）属于代码与配置的职权，放进来等于允许模型用「记忆」绕过限额。写入侧强制，读侧无需防御。

## 服务契约（`PreferenceMemoryService`）

```python
PreferenceMemoryService(async_session_maker, *, clock=utc_now)

await save(visitor_id, key, value, *, source="inferred", ttl_seconds=None) -> VisitorPreferenceEntry
await load_active(visitor_id) -> list[VisitorPreferenceEntry]   # 过滤过期，按 key 排序
await forget(visitor_id, key) -> bool                            # 删除 + 审计；不存在返回 False
```

- upsert：`(visitor_id, pref_key)` 唯一约束，重复 save 原地更新，值/来源/TTL 全量替换（除 confirmed 降级保护）。
- 校验：`pref_key` 1–64 字符；`source ∈ {inferred, confirmed}`；`ttl_seconds ≥ 1`；value 必须可 JSON 序列化（提前 `json.dumps` 快速失败）。
- 事务：save/forget 各自单事务（写主表 + 写审计原子完成）。

## 表结构

`visitor_preferences`：`id`(uuid36 PK) · `visitor_id`(FK guest_visitors.id ON DELETE CASCADE, 索引) · `pref_key`(≤64) · `pref_value`(JSONB) · `source`(check ∈ inferred/confirmed) · `expires_at`(nullable) · `created_at/updated_at`（TimestampMixin）· 唯一 `(visitor_id, pref_key)`。

`visitor_preference_audit`：`id`(int PK) · `visitor_id`(索引, 无 FK) · `pref_key` · `action`(check ∈ save/forget) · `old_value`/`new_value`(JSONB nullable) · `created_at/updated_at`。

## 下一波集成点（建议，不在本相位实现）

接线位置基于 `app/services/commerce/conversation.py`（2026-11-11 状态）：

1. **读取注入**：`CommerceConversation.respond()`（L728）构建 commerce 上下文处——在调用 planner/provider 前 `await preferences.load_active(visitor_id)`，把条目序列化进 `domain_frame_note()`（L88）同层的偏好框架段（有 confirmed 标记的条目优先、附带 source 说明）。只注入 active 条目，条数上界建议 ≤8，超出按 updated_at 取最新。
2. **写入工具**：仿 `_register_memory_tools()`（L649）的既有先例，新增 `commerce_remember_preference`（`source` 固定由调用方语义决定：用户明说 → confirmed，LLM 推断 → inferred）与 `commerce_forget_preference` 两个 `ToolDefinition`，参数 schema 收敛到本服务校验规则；`remember_unsupported_domain` 的会话级记忆与偏好级记忆保持分离，不合并。
3. **provider 侧工具面**（可选）：若走 MCP 面，`app/commerce_provider/mcp_server.py` 的工具面不暴露偏好读写——偏好属于应用库（PostgreSQL），不属于 provider 的 SQLite 领域事实，跨库混写会破坏 provider 单一事实源。
4. **visitor_id 获取**：从会话归属链取当前访客（guest identity 中间件注入的 `GuestVisitor.id`），严禁由模型参数提供（防跨访客写入；服务层已按 visitor_id 隔离，但 id 本身的来源必须可信）。

## 测试清单

- 保存/读取往返；TTL 过期过滤（FixedClock 推进 61s）
- upsert 单行 + 每写一行审计（2 次写 → 1 行主表 + 2 行审计）
- 审计 old/new 值正确（save→save→forget 三行）
- forget 幂等语义（True/False）；跨访客隔离（同 key 互不可见，forget 不误伤）
- 保留键拒绝（quota./limit./permission./policy.）；键长度/source/ttl 校验
- confirmed 不被 inferred 降级
- 迁移：建表/约束/FK/降级保留父表；`expires_at` 时区感知往返
