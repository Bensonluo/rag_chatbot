# 外部虚拟 Commerce Provider 部署记录

2026-10-29 08:07（Asia/Shanghai）。[公开 RAG 页面](https://benluo.art/projects/rag-chatbot/)已切换到独立内部 HTTP commerce provider；仍为模拟交易，无真实扣款、支付、库存或履约。

## 版本与边界

- 部署的后端运行时提交：`f35d776 feat(commerce): externalize facts to virtual provider`
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

## 2026-10-29 13:25 语义类目发现更新

- 后端运行时提交：`a372fd2 fix(commerce): disclose virtual storefront data source`（包含前一提交 `66b1f47 feat(commerce): let agent discover catalog taxonomy`）
- `DEPLOYED_REVISION=a372fd2`
- API image manifest list digest：`sha256:326fb9daf9c042a1df4adfe28452444a2a639b34d17db90982bbb0cb86a46188`
- commerce-provider image manifest list digest：`sha256:e12c9d1714991a05ee81adb2f0af3711d553d155f8ca6364ffea1fa9c0899663`
- 前端无发布；公开页面仍只用 `/api/v1/guest/chat/stream`，无购买按钮或购买 mutation 请求。
- PostgreSQL Alembic 保持 `e75c9d1064a3 (head)`；无迁移、downgrade、备份恢复或旧镜像回滚。provider SQLite volume 未清空。
- 部署后 provider health 为 healthy，API `/health` 为 200。

发布门禁：

- Ruff check / format：517 个文件通过
- strict Mypy：517 个文件无问题
- pytest：**3803 passed / 3 configured skipped / 5 warnings**，核心 coverage **93.53%**

生产浏览器验证：

1. 新访客发送 `我要把小书房改成远程办公角落，预算1500以内，帮我配一套必备装备`。
2. 回复先给出标注“尚未加购”的三套预算内组合，再附权威 SKU/价格/库存/限购事实；候选覆盖显示器、机械键盘、鼠标、耳机、摄像头、人体工学椅。
3. 消息 `id=1329` metadata：`planner_mode=provider`，1 次 `commerce_discover_catalog`，6 次 `commerce_search_products`，0 次模型自造 `max_price`。
4. 执行链从 Pipeline Entry / Intent 开始；浏览器网络无 `/commerce/purchases/...` 请求。
5. 复测 `我想看看耳机` 时，回复明确声明“购物事实来自独立内部虚拟商城API；下单与付款仅模拟，不产生真实订单或扣款”，并展示 typed SKU 事实。

本次发布前发现并修正了一个真实规划缺陷：消息 `id=1327` 中模型把总预算拆成各类目 `max_price`，导致显示器/摄像头/椅子被错误报告为无结果。修正后的系统提示和工具参数说明明确：总预算在组合层控制，类目搜索不得自行设置单项价格上限；带筛选空结果必须去筛选复搜。消息 `id=1329` 证明修正后无自造 `max_price`。

## 2026-10-29 14:00 候选上限与路由置信度更新

- 后端运行时提交：`606cceb fix(commerce): cap visible candidates and trace confidence`
- `DEPLOYED_REVISION=606cceb`
- API image manifest list digest：`sha256:16e6f3979e3b75af07459b10664e0515422f3c702c36d9eb8416c2a39e6c3501`
- commerce-provider image manifest list digest：`sha256:ea4e5a65129c4c5385e9a0fb6fb3a5d8fc93e80123cc4eb451f941d841bd265e`
- PostgreSQL Alembic 保持 `e75c9d1064a3 (head)`；无迁移、备份恢复或数据覆盖。前端无发布。
- 部署后 provider health 为 healthy，API `/health` 为 200。

变更：

- 每轮购物回复的权威 typed 候选硬上限为 5 个；多次类目搜索按各查询的 provider 排名轮转均衡，合并视图保留总量估计并提示可换一批。
- 模型组合建议只能引用当前合并候选块中仍展示的 SKU；交易事实仍由服务端投影。
- shopping/support 路由 LLM 输出 `confidence`，SSE 在业务执行前发出 `cs.intent detail:{intent,confidence,router,reason}`，现有 Execution Chain 恢复显示意图与置信度。
- 代表性验证：Ruff check/format 与 strict Mypy 通过；commerce/provider/evaluation 相关 139 项通过；针对最终断言的 39 项复测通过。按用户要求，本轮不重复运行 3803 项全量。

生产浏览器验证：

1. 新访客发送 `我想买一台电视，帮忙推荐`。
2. 权威候选正好 5 行：3 个显示器 SKU、2 个投影仪 SKU；整条回复唯一 SKU 引用数为 5。
3. 回复提示“目录还有更多匹配商品；可说明偏好或要求换一批，本轮最多展示5个候选”。
4. Execution Chain 顺序为 Pipeline Entry → Intent → Agent Tool Loop；Intent 卡片显示 `shopping` 与 `confidence 98%`。
5. 浏览器仅调用 `/api/v1/guest/chat/stream`，没有 `/commerce/purchases/...` mutation 请求。


## 日志与已知非阻塞问题

commerce-provider 日志无 traceback/error。API 主链路无 provider fallback 或 simulated planner。仍有一个既有非购物路径警告：MiniMax light tier 余额不足（HTTP 402），session compression 因此按设计 best-effort 失败；GLM 购物 Agent 与外部 provider 不受影响。

## 未验证范围

- 真实商家、支付 mandate、真实扣款、退款或履约。
- 多副本 API/provider 拓扑、生产并发容量和负载测试。
- provider volume 的异地备份、恢复演练及长期保留策略。
- 真实模型的 held-out 大规模语义质量统计；自动化测试只验证合同、编排与安全边界。
