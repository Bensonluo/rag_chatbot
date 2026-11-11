# Commerce Provider MCP Server — 设计文档

日期：2026-11-11 · 分支：`feat/shopping-agent-p0p1-20261111`
实现：`app/commerce_provider/mcp_server.py` · 测试：`app/tests/unit/commerce_provider/test_mcp_server.py`

## 定位

把内部虚拟电商 provider 的领域能力（`app/commerce_provider/core.py`）通过 Model Context Protocol 暴露为 MCP 工具面，供任意 MCP host（包括下一波的对话 agent 循环）驱动搜索与购买流程。与 HTTP facade（`app/commerce_provider/api.py`）**共用同一个 `CommerceProviderCore`**——本模块是又一个薄类型化外壳，两个面语义上逐字对齐，结构上互不依赖（不经过 HTTP 自调用）。

## 依据（先核实后编码，两源确认）

- MCP Python SDK：`pip install mcp` 当前为 **2.x（本仓库锁定 `mcp>=2.3.0,<3`，实测 2.3.0）**。v2 将 `FastMCP` 更名为 `mcp.server.MCPServer`；transport 配置移到 `run()`；确认机制使用 `Resolve`/`Elicit` 依赖注入。来源 1：官方 SDK 文档站 `py.sdk.modelcontextprotocol.io`（README / What's new in v2 / Servers）。来源 2：本仓库 `.venv` 内安装的 2.3.0 源码逐签名核对（`MCPServer.__init__`/`run_streamable_http_async`/`resolve.py`/`bearer_auth.py`）+ 进程内冒烟（`Client(mcp)` 走通 accept/decline 两臂）。
- Elicitation 语义：MCP 规范 2025-11-25《Elicitation》页（form 模式、accept/decline/cancel 三动作、扁平 primitive schema）。SDK 2.x 同时服务 2025-11-25（会话内 `elicitation/create`）与 2026-07-28（`InputRequiredResult` 多轮回合）两个协议代际：同一个工具体无需改动即可服务两类客户端。
- 鉴权：SDK `TokenVerifier` 协议（`mcp/server/auth/provider.py`）+ `BearerAuthBackend`/`RequireAuthMiddleware`（缺失/错误 token → HTTP 401 + `WWW-Authenticate`）。

## 工具面（8 个）

只读（模型可直接调用）：

| 工具 | 入参要点 | 返回 |
|---|---|---|
| `search_products` | `query`(≤500), `owner_user_id`(≥1), `page_size` **1–5**（默认 5，比 HTTP 的 1–50 收窄，控上下文体积）, `page_token`, `sort`, `filters` | `CatalogPage` |
| `get_product` | `sku_id`(1–64), `owner_user_id` | `CatalogItem` |
| `get_categories` | `category_ids` 过滤 ≤32 个 × ≤64 字符（与 HTTP 语义一致） | `CatalogTaxonomy`（全量 151 个权威类目） |
| `discover_catalog` | `query`, `owner_user_id` | 一次调用返回全量类目 + 该查询前 3 条商品（开场定位） |

写操作（确认后执行，全部幂等）：

| 工具 | 入参要点 | 返回 |
|---|---|---|
| `update_cart` | `lines` ≤50 行（整体替换语义）, `cart_id?`, `expected_cart_version?` | `CartSnapshot` |
| `prepare_quote` | `cart_id`, `expected_cart_version` | `QuoteSnapshot`（TTL 由 `COMMERCE_PROVIDER_QUOTE_TTL_SECONDS` 控制，默认 300s） |
| `create_order` | `quote_id` | `OrderSnapshot` |
| `sandbox_pay` | `payment_id` **或** `order_id`（二选一；给 `order_id` 时服务端先建支付单再支付，两步各自幂等） | `PaymentSnapshot` |

### 确认机制实现方式

每个写工具带一个 `confirmation: Annotated[ElicitationResult[WriteConfirmation], Resolve(ask)]` 参数（`WriteConfirmation = {confirmed: bool}`）。SDK 在调用时把 resolver 返回的 `Elicit("…确认…？", WriteConfirmation)` 按协商协议送达客户端（旧代际是真实 elicitation 请求，新代际是 input-required 回合），模型不可见、不可伪造该参数。执行分支：

- `AcceptedElicitation` 且 `confirmed=True` → 执行核心层调用；
- accept-but-false / decline / cancel → 抛 `ToolError("user_declined: …")`，**核心层零接触**（测试 `test_write_tool_declined_confirmation_changes_nothing` 钉死）。

### 幂等键（结构性派生，禁止随机 UUID）

同一逻辑操作 → 同一键 → 核心层 `_mutate` 回放原始回执（`operations` 表）：

- `update_cart`: `mcp:cart:{owner}:{cart_id|new}:v{version}:{sha256(canonical lines)[:16]}`（行按 sku 排序后取摘要，重排序不换键）
- `prepare_quote`: `mcp:quote:{owner}:{cart_id}:v{version}`
- `create_order`: `mcp:order:{owner}:{quote_id}`
- `sandbox_pay`: `mcp:pay-init:{owner}:{order_id}`（建单）+ `mcp:pay:{owner}:{payment_id}`（支付）

结果未确认（`ProviderUnknownError`，fault 注入可复现）时返回给模型的补救话术就是「用完全相同的参数重试本工具」——重试即回放，不会二次扣款（测试 `test_unknown_outcome_retry_replays_the_committed_result`）。

### 错误映射

`ProviderError` → `ToolError(f"[{code}] {message}")`（模型可见 `is_error` 结果，code 与 HTTP 面同词表：`not_found`/`rejected`/`conflict`/`auth`/`unknown`）。其余异常按 SDK v2 默认成为脱敏协议错误，不泄内部细节。

## 与 HTTP API 的映射表

| MCP 工具 | HTTP（`/commerce/*`） | 核心方法 |
|---|---|---|
| `search_products` | `POST /search` | `search_products` |
| `get_product` | `GET /products/{sku_id}` | `get_product` |
| `get_categories` | `GET /categories` | `catalog_categories` |
| `discover_catalog` | （组合：`GET /categories` + `POST /search`） | `catalog_categories` + `search_products` |
| `update_cart` | `POST /cart/update` | `update_cart_lines` |
| `prepare_quote` | `POST /checkout` | `prepare_checkout` |
| `create_order` | `POST /orders` | `create_order` |
| `sandbox_pay`(payment_id) | `POST /payments/{id}/sandbox-pay` | `mock_pay` |
| `sandbox_pay`(order_id) | `POST /payments` + `POST /payments/{id}/sandbox-pay` | `create_payment` + `mock_pay` |

MCP 面不暴露：动态类目铸造（`POST /categories/dynamic`）、测试故障注入（`/testing/*`）、operation 查询（未知结果统一引导走「同参重试」，见上）。

## 鉴权与配置

- 复用 `COMMERCE_PROVIDER_AUTH_TOKEN`（`StaticTokenVerifier`，`hmac.compare_digest` 比较；空 token 关闭鉴权，与 HTTP 面同规则）。401 带 `WWW-Authenticate`。
- 配置类 `CommerceMcpSettings`（`env_prefix="COMMERCE_PROVIDER_"`, `extra="ignore"`）：`COMMERCE_PROVIDER_MCP_HOST`(127.0.0.1) / `COMMERCE_PROVIDER_MCP_PORT`(8072) / `COMMERCE_PROVIDER_MCP_PATH`(/mcp)；`database_path`/`auth_token`/`quote_ttl_seconds` 与 HTTP 面共享同名环境变量（默认同一 SQLite 文件，WAL 支持双进程读写）。

## 启动

```bash
python -m app.commerce_provider.mcp_server            # 127.0.0.1:8072/mcp
python -m app.commerce_provider.mcp_server --port 9000 --token "$COMMERCE_PROVIDER_AUTH_TOKEN"
```

Streamable HTTP transport（`server.run(transport="streamable-http", host=…, port=…, streamable_http_path=…)`）。进程内测试用 SDK 官方 in-memory 模式 `async with Client(mcp) as client:`，不需要端口。

## 测试清单（`app/tests/unit/commerce_provider/test_mcp_server.py`，12 例）

- 握手：`list_tools` 恰好 8 个工具（协议握手冒烟）
- 只读四工具正常流 + `page_size=6` 被拒 + `get_product` 404 分支
- 写工具：decline 零副作用；同参重放 == 首调结果；不同参数 → 新版本；quote→order→pay 全链路 + pay 重放
- 幂等键纯函数派生（确定性、owner 隔离、内容区分、≤128 字符）
- 故障注入：unknown_after_commit 后同参重试回放已提交结果
- HTTP 面：无 token/错 token 401，正 token 非 401（Starlette ASGI 级）

## 已知边界（未做）

- 无独立健康检查路由（SDK app 由 `streamable_http_app()` 生成；如需 `/health` 探针需挂 ASGI mount，留给下一波）。
- 未接 conversation.py / tools.py（按分工明确不在本相位）。
- 多副本部署的 `request_state` 密钥共享（SDK `RequestStateSecurity`）未配置——当前单实例内网部署不涉及。
