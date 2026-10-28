# 公共访客与 Agent 纯文本模拟购买

日期：2026-10-29。本文描述当前代码目标：购物语义由 provider LLM Agent 独占，商品与交易事实由独立内部虚拟 commerce provider 提供。无真实扣款、真实配送或商业支付接入。

## 目标与边界

- 项目目标是验证 **Agent 自动导购编排链路**：多轮理解、工具选择、参数绑定、多商品购物车、交易状态推进、插问恢复和故障后原操作恢复。
- 商品、库存、限购、多行购物车、报价、订单和支付事实由独立 `commerce-provider` 服务拥有；它用确定性生成的 100,000,000 个可寻址 SKU 和 SQLite 事务模拟成熟外部 API，不把目录全量枚举进内存或聊天库。
- 全流程只使用公开聊天输入和助手文本；不提供购买按钮、数量输入、购买卡片、付款链接或独立购买操作入口。
- 下单与付款是两次独立、明确、新鲜的文字授权；模型不能自行绕过服务端授权门。
- `COMMERCE_PROVIDER_MODE=external` 是显式组合选择。HTTP provider 超时、拒绝或不可用时，购物显式失败/提示稍后查询，不会静默退回本地 planner 或本地 mock。

## 身份与数据归属

`POST /api/v1/guest/bootstrap {}` 签发 `rag_guest_session` Cookie，返回访客 UUID、到期时间、私人咨询 ID 和 `session_key`。浏览器持有 Cookie 即代表这位访客，不绑定现实个人；清除 Cookie 后不能凭旧咨询 ID 恢复身份。

Cookie 为 HttpOnly、Secure、SameSite=Lax、Path=/ 的 session cookie。服务端默认七天有效，可通过 `GUEST_SESSION_TTL_SECONDS` 调整；本地 HTTP 才可设置 `GUEST_COOKIE_SECURE=false`。数据库只保存随机令牌 SHA-256。

访客使用内部 `User.is_guest` 主体复用整数 owner，但不能成为管理员。所有请求使用 `credentials: include` 和 `X-Rag-Guest: 1`，写请求核对 Origin。`session_id + session_key` 只定位咨询，Cookie 才是身份；HTTP 与会话门闩内再次核对 owner、private 标记和当前 key。访客路径不采信 body/model 的 `user_id`。

凭据失效返回 401，前端清除旧上下文，由用户明确重新开始；身份存储故障返回 503，不签发替代访客。消息历史仍是 best-effort，不能以空历史判断交易不存在；购买状态由 provider 回执和本地 operation 记录核实。

## Agent 与 provider 边界

```mermaid
flowchart LR
    U[用户文字] --> G[Cookie 与会话授权]
    G --> C[CommerceConversation<br/>上下文与工具权限]
    C --> A[有界 AgentService]
    A --> T[请求专属导购工具]
    T --> P[PurchaseService<br/>授权、版本、幂等、恢复]
    P --> X[ExternalCommercePort]
    X --> S[commerce-provider<br/>FastAPI + SQLite]
    S --> V[Typed Catalog/Cart/Quote/Order/Payment]
    V --> R[完整助手文本]
    C --> H[持久聊天快照]
```

- `CommerceConversation` 在每轮构造请求局部工具 registry，不向旧共享 Demo 图注册购买工具。接受购物任务后先持久提升 automation epoch，再调用模型或工具；模型失败、请求取消或聊天保存失败不能复活旧退款提案。
- `AgentService` 负责理解、规划和实际工具调用：请求是否购物、买什么、约束是什么、候选页中“1号/第二款/便宜的那个”指哪个、如何拆成多行、是否追问、是否足以授权，均由模型判断。
- 常规工具：结构化商品搜索（价格/属性/库存过滤、排序、分页）、SKU 详情、多行购物车更新、试算、状态查询。
- 受控工具：`commit_checkout` 与 `mock_pay` 只在当前阶段匹配且最新展示版本有效时注册；handler 再核对 owner、purchase、version、confirmation/payment 和 generation。模型传入 owner、金额、付款成功标记或外来 idempotency key 无效。
- 一回合最多推进一个不可逆阶段。模糊确认、否定、疑问或组合的“确认下单并付款”均只澄清或查询，不写入。
- 助手文本由服务端 `CommerceView` 投影：目录页、SKU/规格、限购、多行小计、运费、总价、有效期、订单号、付款号和状态。模型自由文本不能宣布交易成功。

## Provider 事实与本地编排状态

外部 provider 独立拥有：

- 100,000,000 个稳定可寻址 SKU、价格、库存、限购和属性；
- 结构化搜索、过滤、排序和有界分页；
- 多行购物车和乐观版本；
- 报价、有效期、报价/购物车版本一致性；
- 订单创建、库存扣减、quote 消费；
- payment intent、模拟支付、processing/paid/failed 状态；
- 按稳定 `request_key` 的幂等回执和 operation lookup；
- 测试专用 reject、unknown-after-commit、processing 故障注入（仅在显式启用时暴露）。

本地 `commerce_purchases/operations/confirmations` 只保留非语义安全编排：owner 绑定、会话活动购买、展示版本、不可变报价确认、operation 幂等键、unknown 后 lookup 和审计。历史 `commerce_actions` 表保留生产数据但运行时不再读写。provider 的 SQLite 状态位于 Docker volume `commerce_provider_data`，与主 PostgreSQL 备份分开，必须单独备份；清空聊天不取消 provider 订单或付款。

## HTTP 与前端契约

|请求|用途|
|---|---|
|POST `/guest/bootstrap`|Cookie + visitor_id/expires_at/session{id,session_key}|
|POST `/guest/chat`、`/guest/chat/stream`|商品、购物车、报价、下单、付款、查询的唯一公开入口|
|GET、DELETE `/guest/history`|核对原 key；清空轮换 key，不清购买|
|POST `/guest/feedback`|绑定真实完成消息及当前咨询 key|
|GET `/guest/commerce/purchases/active`、`/{purchase_id}`|服务端/测试用只读状态核实，不向前端提供操作|

`commerce-provider` 只监听内部 Docker 网络，不发布宿主端口；API 通过 `ExternalCommercePort` 和 Bearer token 访问。`CommerceView` 仍是服务端事实投影，没有 actions/action_ref。前端不解析该结构，也不发起购买专用 POST。

SSE 输出文本帧、可选持久化 completion 和 DONE。completion 必须匹配原 session_key；没有有效 completion 时不显示普通“接收完成”，只提示聊天保存未确认、交易状态需查询。聊天保存失败不回滚业务，也不虚构 message ID。

## 千万变体边界

购物自然语言变体空间开放且千万级以上，不能穷举覆盖。新增一个短语不得要求新增正则、固定话术、候选映射或状态迁移；穷举规则和穷举测试不能证明语义完备。未见输入、错别字、口语、混合语言、多意图和长上下文由 LLM Agent 泛化、追问或安全拒绝；模糊时不得写交易。自动化测试只验证编排、类型、授权、幂等和恢复，真实模型质量需分层抽样和 held-out 评测估计。

## 验证

定向命令：

```sh
.venv/bin/pytest --no-cov -q app/tests/unit/commerce_provider \
  app/tests/unit/services/commerce \
  app/tests/evaluation/test_guest_commerce_flow.py \
  app/tests/evaluation/test_commerce_task_fencing.py \
  app/tests/evaluation/test_guest_conversations.py
```

覆盖 provider catalog/address space、多行 cart、报价/订单/支付、幂等冲突、unknown-after-commit、SQLite 重开持久化、owner 隔离、HTTP wire contract、Agent 工具循环、两次独立文字确认、重复确认幂等、跨访客隔离、报价过期、购物车变更、政策插问恢复、processing 和消息保存失败。完整门禁结果以最新部署记录为准。

本合同不宣称真实支付、真实供应商、生产并发/容量、跨副本 Cookie 拓扑或浏览器支付认证。原[接入评估](conversational-commerce-assessment.md)保留历史设计；其中的按钮/结算页建议已被本文纯文本合同取代。
