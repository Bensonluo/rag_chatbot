# 外部虚拟 Commerce Provider 部署记录

2026-10-29 08:07（Asia/Shanghai）。[公开 RAG 页面](https://benluo.art/projects/rag-chatbot/)已切换到独立内部 HTTP commerce provider；仍为模拟交易，无真实扣款、支付、库存或履约。

## 版本与边界

- 后端：`f35d776 feat(commerce): externalize facts to virtual provider`（`bugfix/refund-handoff-20261028` 与 `main` 均指向该提交）
- 前端：保持 `cadcb3b fix(chat): remove public workspace notice`，无前端发布
- API image manifest digest：`sha256:b7a32f16f6bddc2451c8ded269e655d5b701f3ad60912ff041d4e18e8c35a7c2`
- commerce-provider image manifest digest：`sha256:195dfa320b45fc7d0518b526f350b2736cf6a253498b55c542a0b60a5fdfb40f`
- `DEPLOYED_REVISION=f35d776`
- PostgreSQL Alembic：部署前后均为 `e75c9d1064a3 (head)`；无迁移、downgrade 或备份覆盖
- provider SQLite 卷：`rag_chatbot_commerce_provider_data -> /data/commerce.db`
- provider 不发布宿主端口；公网 `POST https://benluo.art/commerce/search` 返回 404
- 内部 Bearer token 已写入服务器 `.env`（长度 64 字符，未写入仓库或本文），API/provider 共用；无 token 访问内部 search 返回 401

部署前完整门禁：

- Ruff check：通过
- Ruff format check：513 个文件已格式化
- strict Mypy：513 个文件无问题
- pytest：**3772 passed / 3 configured skipped / 5 warnings**
- 核心 coverage：**93.58%**

新增契约覆盖 provider catalog/address space、SQLite 事务、幂等冲突、unknown-after-commit、重开持久化、owner 隔离、HTTP wire、`PurchaseService` over external provider、多行购物车、报价/下单/付款、Compose token/无宿主端口，以及外部 unknown 回执恢复。

## 部署前备份与配置

PostgreSQL 逻辑备份：

`/opt/rag_chatbot/backups/production/db_backup_20261029_075235_stage2.sql.gz`

- 大小：1,424,126 bytes
- SHA-256：`07299cdb1b28a66de23ea1514b16e61436462019b8c35a11cecd018466c7395b`
- 部署前 Alembic：`e75c9d1064a3 (head)`

服务器 `.env` 修改前已另存到 `backups/production/env_backup_20261029_075235`（0600）。本次只新增内部 `COMMERCE_PROVIDER_TOKEN`，不回显 secret。

## 部署后健康

|检查|结果|
|---|---|
|commerce-provider container|healthy / running|
|API container|healthy / running|
|API `/health`|200|
|API `/ready`|200|
|provider internal `/health`|200|
|provider 无宿主端口|`{"8000/tcp":null}`|
|provider unauthorized search|401|
|provider authorized search|200|
|公开 provider search|404|
|Alembic|`e75c9d1064a3 (head)`|

授权内部搜索返回 `恒锐 Astro A0000 显示器 [SKU-00000000] ¥861.00` 等生成目录事实，带库存、限购和下一页 token，不枚举 100,000,000 SKU 空间。

## 真实浏览器旅程

新 Cookie 访客在公开页面仅通过聊天输入发送：

1. `我想买办公室用的东西`
2. `两个黑色显示器，再加一个最便宜的键盘，先放进购物车`
3. `1号机械键盘来一个，另外帮我搜索黑色显示器`
4. `1号和2号显示器各一台，键盘保留，更新购物车`
5. `结算`
6. `确认下单`
7. `确认模拟付款`
8. `查询刚才的订单和付款`

结果：

- 第一轮 broad request 由 GLM 追问类别/预算/规格，未创建交易。
- Agent 搜索生成目录并按上一页候选解析 `1号机械键盘`。
- 多行购物车最终为：
  - `SKU-00000000` 恒锐 Astro A0000 黑色显示器 × 1，¥861.00
  - `SKU-00000004` 恒锐 Nova N0001 黑色显示器 × 1，¥921.00
  - `SKU-04000000` 恒锐 Astro A0000 黑色机械键盘 × 1，¥265.00
- 报价总价：**¥2047.00**，配送 ¥0.00，带有效期。
- `确认下单` 只创建一个订单和 payment intent，明确尚未付款。
- `确认模拟付款` 是独立授权；付款成功后仍明确无真实扣款。
- 状态查询返回同一订单/付款事实。
- 浏览器网络请求无 `/commerce/purchases/...` 或 provider URL；购买推进全部通过 `/api/v1/guest/chat/stream`。
- 页面 DOM 无购买按钮、数量输入或购买卡片。

交易事实：

- order：`o_8e0727222ae7462e94baaad847c05eb7`
- payment：`pay_582f6e60931648ad8f4121d10ad37d7e`
- quote：`q_6d68dfabd3a147b4a722b8399e50d341`
- total_minor：204,700（CNY）
- order/payment 状态：`paid`

## 双侧数据证据

本地 PostgreSQL 只保存安全编排引用/审计；本购买 operation 为：

|kind|status|generation|数量|
|---|---|---:|---:|
|update_cart|completed|1|1|
|update_cart|completed|3|1|
|quote|completed|5|1|
|order|completed|2|1|
|payment|completed|2|1|
|mock_pay|completed|2|1|

新会话 8 条 shopping assistant 消息全部为 `planner_mode=provider`，`simulated` 为 0。

独立 provider SQLite 事实：

|表|行数|
|---|---:|
|carts|1|
|cart_lines|3|
|quotes|1|
|orders|1|
|payments|1|
|operations|6|
|stock_deltas|3|

下单后 provider 权威库存变化：

|SKU|下单后 available_quantity|
|---|---:|
|SKU-00000000|6|
|SKU-00000004|12|
|SKU-04000000|54|

provider SQLite 在线备份：

`/opt/rag_chatbot/backups/commerce_provider/provider_backup_20261029_080705.sqlite3.gz`

- 大小：4,222 bytes
- SHA-256：`9de67bd3666a7cd31b636fd6d329035125b91f11218fe57d3457a2ae2cfa2feb`

备份通过 SQLite online backup 生成一致快照，未直接复制运行中的 db/wal 文件。

## 日志与已知非阻塞问题

commerce-provider 日志无 traceback/error。API 主链路无 provider fallback 或 simulated planner。仍有一个既有非购物路径警告：MiniMax light tier 余额不足（HTTP 402），session compression 因此按设计 best-effort 失败；GLM 购物 Agent 与外部 provider 不受影响。

## 未验证范围

- 真实商家、支付 mandate、真实扣款、退款或履约。
- 多副本 API/provider 拓扑、生产并发容量和负载测试。
- provider volume 的异地备份、恢复演练及长期保留策略。
- 真实模型的 held-out 大规模语义质量统计；自动化测试只验证合同、编排与安全边界。
