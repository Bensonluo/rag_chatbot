# rag_chatbot 按用户（per-user）架构设计

> 2026-10-01。回应 review 2026-09-26 #1 遗留项「检索 ACL 服务端注入」以及产品层缺口：
> 当前 demo 从未按用户设计。本文件给出完整的目标架构与分阶段落地路径。
> 状态：P0a 实施中；**用户决策（2026-10-01）：demo 以单用户共享形态为最终形态，
> 隐私问题不考虑；DEMO_MODE 开关即目标架构——开 = 公共单用户 demo，
> 关 = 每用户独立 session 的常规产品。据此 P0b（session UUID 不可枚举）与
> P1（访客层）从计划中移除，严格模式产品线保留 P2/P3。**

## 1. 现状盘点（证据）

| 维度 | 现状 | 证据 |
|------|------|------|
| 认证 | 真 JWT 底座：register/login/refresh 完整；无凭据 → `None`（匿名放行） | `app/api/v1/auth.py`、`app/api/deps/__init__.py:83` |
| 姿态开关 | `DEMO_MODE=true`（默认）：存在性检查；`false`：身份只取令牌，越权统一 404 | `app/api/deps/authorization.py`（d12a1cd） |
| 会话 | server-side 建 session 挂 `current_user.id`；匿名访客不能调 POST /sessions → 客户端自造 id → auto-create 挂**共享** demo-visitor（user 5） | `app/api/v1/sessions.py:73`、`app/services/chat/persistence.py`（d0d2b2a） |
| Session id | DB 自增整数、client-facing 直接暴露 → **可枚举**；demo 姿态 history 读只查存在 → **任何访客可枚举他人会话** | `app/api/v1/chat.py:500`（`ensure_session_access` demo 短路） |
| 知识库 | Document 零归属列；Qdrant payload 无 owner/visibility；严格模式写入 admin-gated，但读 = 全局共享 | `app/models/database/document.py`、`qdrant_client.py:162` |
| 检索过滤 | 业务 filter 来自 slot 抽取，白名单 `FILTERABLE_METADATA_KEYS` 收窄；**无 ACL 通道**；filter-miss 回退**丢弃全部 filter** | `vector_base.py`、`nodes.py:854` |
| L2 检索缓存 | key = (query, filters) 哈希，全局共享，无身份维度 | `retrieval_cache.py:123` |
| L0/L1 答案缓存 | 已有 personalization 门槛（used_history 不写）——按「内容是否个人化」防御，与身份无关 | `nodes.py:_maybe_put_semantic`（8c3873d） |
| 记忆/事实 | user_facts 按 user_id 键控；demo 匿名 user_id=0 → 空召回，无串扰 | `nodes.py:_recall_user_facts` |
| Feedback | 严格模式归属校验已上（148507d）；demo 存在性检查 | `feedback_repository.py:get_owned_message` |
| 限流 | 全局 per-IP 滑窗 + chat per-IP 预算；无 per-user 维度 | CLAUDE.md 已知问题 |
| 工具 | mock 订单库；工具执行已传 user_id（接口就绪，数据未分层） | `nodes.py:688` |

结论：**缺的不是某个补丁，是一条「身份贯穿」的主线**。认证底座已就绪，产品层没有消费它。

## 2. 目标身份模型：三层

```
┌─ Visitor（匿名访客，浏览器身份）
│   服务端签发 visitor 凭据（UUID，签名）
│   拥有：自己的 sessions、history、feedback 权限
│   不拥有：账号、私有文档、长期 facts（默认关）
│
├─ User（注册用户，JWT 账号）
│   拥有：sessions、history、feedback、personalization facts、
│         私有 KB 文档（P2）、per-user 配额（P3）
│
└─ Admin（is_admin）
│   在 User 之上：共享 KB 写入、（现状即有）
```

**核心决策 D1 —— 访客即 User 行。** 访客不引入平行的 `visitor_id` 概念，而是建
`User(role="visitor")` 行。理由：整条 per-user 主线（session.user_id、feedback
归属 join、facts 键、工具 user_id）已经全部以 user_id 为键——复用它，访客层
免费获得全部隔离属性；引入第二套 visitor_id 意味着每个消费点都要写两遍。

## 3. 各层设计决策

### 3.1 会话不可枚举（D2）

内部 int PK 保留，新增 `public_id: UUID`（唯一索引），**所有 client-facing
路由（chat、history、clear、feedback 引用）改用 public_id**。自增整数不再
离开服务端。迁移 + API 兼容期（int id 保留一个版本的只读兼容或直接切，见 P0b）。

### 3.2 Demo 姿态也做归属校验（D3）

现状 demo 的 `ensure_session_access` 只查存在。改后：访客凭据 → 匹配
session 归属（他人的和不存在的一样 404，同 BOLA 措辞原则）。demo 姿态的
语义从「完全公开」收窄为「访客只看自己的」——公共演示不受影响（一个浏览器
就是一个访客），但枚举他人会话被关闭。

### 3.3 KB 归属与检索 ACL（D4 + D5，即 review #1 的正式解法）

**Schema（P2）：** `Document` 增 `visibility ENUM(public|private)`（默认
public）+ `owner_user_id INT NULL`（private 时非空）。alembic migration。

**Payload 打标（P2）：** ingestion 组 payload 处（`ingestion.py:177`）写入
`metadata.visibility` / `metadata.owner`。存量 chunk 不回填——见下。

**ACL 语义 = 「排除 private」，不是「匹配 public」（D4 关键）：**
ACL 过滤表达为 *must_not(visibility=private 且 owner≠调用者)*。Qdrant 里
没有该字段的旧 chunk 不命中排除条件 → 自动视为 public，**无需全量重建索引**。
BM25 in-memory 索引同理（metadata 缺失 = public）。

**服务端注入通道（D5，P0a 本次落地，先于 schema）：**

```python
@dataclass
class VectorSearchRequest:
    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None      # 业务 filter：slot 抽取，白名单收窄
    acl_filters: dict[str, Any] | None = None  # ACL：仅服务端身份上下文构造
```

不变式（逐条对应 review 原文「ACL 由服务端注入，不能放进可取消的业务筛选条件中」）：
1. `acl_filters` 只在服务端由身份上下文构造（当前调用者 → scope 表达式），
   永不出自请求体或 slot 抽取；
2. `intersect_metadata_filters` 白名单**不适用**于它（白名单是业务 filter 的
   契约，ACL 键如 `visibility`/`owner` 不在 chunk 契约白名单内，正因此必须
   独立通道）；
3. nodes.py 的 filter-miss 回退（`nodes.py:854`）**丢业务 filter、保 ACL**——
   召回保护不得变成越权通道；
4. hybrid search 两腿把业务 filter 与 ACL 合并（AND 语义）进有效过滤；
5. L2 缓存 key 纳入 ACL scope（否则 A 用户的私有命中会 replay 给 B）。

**今日注入什么？** 全部文档皆 public → 所有调用者的 scope 相同
（`must_not private`，当前匹配空集）。seam 落地后系统行为零变化，但从此
「加私有文档」只是把 scope 表达式从常量换成 per-user，不动管线。
这是先建不可剥离的接缝、再灌数据——顺序反过来（先有私有数据后有 ACL）
才是事故。

### 3.4 缓存体系在多用户下（D6）

- **L2 检索缓存**：key 加 scope 摘要（D5.5）。公开 scope 的条目天然共享，
  私有 scope 各自隔离。
- **L0/L1 答案缓存**：现有 `generation_used_history` 门槛防的是「内容个人化」；
  新增同构门槛 `retrieval_user_scoped`——本 turn 检索带非公开 scope 时，
  L0/L1 写入拒绝（答案可能引用了只有该用户能看到的文档，跨用户 replay 即泄漏）。
- FAQ / fact_store：curated 共享内容，不设归属（明确不做）。

### 3.5 记忆与个性化（D7）

facts 按 user_id 键控，身份真实后自动正确。访客层默认**关闭** facts 写入
（匿名流量不做 LLM 抽取的长期画像），`USER_FACTS_FOR_VISITORS=false` 配置。

### 3.6 配额与滥用（D8）

严格模式：per-user bucket 叠加在现有 per-IP 滑窗之上（Redis，复用
EndpointRateLimiter 骨架，key 从 IP 换 user_id）。Demo：per-visitor +
per-IP 双层不变。配置：`USER_RATE_LIMIT_REQUESTS_PER_MINUTE`。

### 3.7 工具与订单数据（D9）

工具执行签名已带 `user_id`（`nodes.py:688`、agent 工具同）。真实订单系统
接入时：查询/退款工具按 user_id 过滤订单（服务端注入，同样不信任模型给的
order_id 归属——查到的订单必须属于调用者，否则 404）。mock 阶段无 per-user
数据，仅记录该契约。

## 4. 分阶段落地

| 阶段 | 内容 | 交付物 | 依赖 |
|------|------|--------|------|
| **P0a** | ACL seam：`acl_filters` 通道 + hybrid 两腿合并 + 回退保 ACL + L2 scope key + 全套不变式测试 | commit（无 schema 变更，行为零变化） | 无 —— review #1 关闭 |
| **P0b** | Session `public_id` UUID 迁移 + client-facing 路由切换 | migration + API 变更 | 无 |
| **P1** | 访客层：首聊签发访客凭据（HttpOnly cookie，Bearer 优先）；server-side 建 per-visitor User 行 + session；demo auto-create 从共享 demo-visitor 改为 per-visitor；demo posture 归属校验（D3） | 访客隔离闭环 | P0b（session 引用方式） |
| **P2** | 私有文档：schema migration（visibility/owner）+ ingestion 打标 + admin 上传带归属 + per-user scope 注入 + L0/L1 `retrieval_user_scoped` 门槛 | 私有 KB 端到端 | P0a（seam） |
| **P3** | per-user 配额、访客 facts 策略开关、访客→注册账号数据合并 | 运营层完善 | P1/P2 |

每阶段独立可交付、demo 公共体验不变（P1 起 demo 从「共享一切」变为「访客间互不可见」，
这是收紧不是破坏）。

## 5. 明确不做的

- **Collection-per-tenant（Qdrant 按租户分集合）**：当前规模下 payload-filter
  ACL 是正确粒度；分集合是远期分片决策，且让混合检索/重排复杂化。
- **完整 RBAC / SSO / OAuth**：`is_admin` 布尔足够当前产品语义。
- **GraphRAG 的 ACL**：图检索当前默认关闭（`GRAPH_RAG_ENABLED=false`）；启用
  私有文档后再按同一「排除 private」语义补图侧（单独 finding）。
- **访客跨设备漫游**：访客凭据绑定浏览器，不提供找回。

## 6. 风险与回退

- P0a 无行为变化，回退 = revert 单 commit。
- P0b API 引用方式变更是 breaking change（demo 前端同步改）；保留一个版本的
  int-id 只读兼容窗口。
- P2 migration 在共享服务器 DB 上执行需用户授权 + 备份（backup.sh 已就位）。
- 最大风险项：**P1 凭据签发与现有 deploy 的 nginx/cookie 路径交互**——上线前
  用 curl 验证 Set-Cookie 行为，不假设浏览器语义。
