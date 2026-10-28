# Agent 纯文本模拟购买部署记录

日期：2026-10-29 00:31（Asia/Shanghai）。[公开 RAG 页面](https://benluo.art/projects/rag-chatbot/)已切换为无购买按钮的纯文本聊天购买体验。

## 版本与边界

- 后端：`4094c52 feat(commerce): execute mock purchases through agent text`
- 前端：`cadcb3b fix(chat): remove public workspace notice`
- 后端镜像 digest：`sha256:8080184b856baa093255186f8d4041b1e9c96f0e88096ce17691c78351293edd`
- API 仅监听 `127.0.0.1:8000`，由宿主 HTTPS nginx 反代。
- 数据库保持在 `e75c9d1064a3 (head)`；本次无新迁移，未 downgrade，未用旧备份覆盖新数据。
- 所有商品、订单和付款仍为 SQL mock，无真实扣款、真实供应商或履约。

## 部署前保护

生产逻辑备份：

`/opt/rag_chatbot/backups/production/db_backup_20261029_002940_BZbPiB3L.sql.gz`

- gzip 完整性校验通过
- 大小：1,320,600 bytes
- SHA-256：`fcb298ac9ce73fec2e20099a759e15e1d10224a859ed4a9dc9ddb419adf1b50d`

部署前 Alembic：`e75c9d1064a3 (head)`。

部署前基线：

|对象|行数|
|---|---:|
|users|11|
|guest_visitors|6|
|chat_sessions|412|
|messages|1098|
|commerce_purchases|2|
|commerce_operations|0|
|commerce_confirmations|0|
|commerce_actions|8|
|commerce_mock_objects|3|
|commerce_mock_receipts|0|

## 部署后健康

- API 容器：running
- 本机 `/health`：200
- 本机 `/ready`：200
- 公网 `/health`：200
- 公开 RAG 页面：200
- Alembic 仍为 `e75c9d1064a3 (head)`
- 旧 `POST /guest/commerce/purchases/{id}/actions`：404

宿主 nginx 当前只反代公网 `/health` 与 `/api/`，不暴露公网 `/ready`；这不是应用 readiness 状态。

## 线上真实浏览器旅程

访客在公开页面仅通过输入框发送：

1. `我想买降噪耳机`
2. `第二款黑色，两副`
3. `就这些，帮我结算`
4. `确认下单`
5. `确认模拟付款`
6. `查询刚才的订单和付款`

结果：

- 目录显示 SKU、颜色、价格和库存。
- 购物车显示黑色耳机 × 2、小计 ¥598.00。
- 报价显示商品小计、配送、总价 ¥598.00 和有效期。
- `确认下单` 创建订单并明确“尚未付款”。
- `确认模拟付款` 后返回已付款状态，并明确无真实扣款。
- 查询返回同一订单和付款事实。
- 页面 DOM 无购买按钮、数量输入或购买卡片。
- 浏览器网络无 `/commerce/purchases/...` 请求。

本次验证产生的模拟订单：

- order：`mo_796bca87-057d-4c6c-8537-bd06c75f322f`
- payment：`mp_27582227-eb71-4ea4-8208-308928b25730`
- purchase 状态：`paid`

该购买下 operation 结果：

|operation|状态|数量|
|---|---|---:|
|set_cart|completed|1|
|quote|completed|1|
|order|completed|1|
|payment|completed|1|
|mock_pay|completed|1|

第二位全新 Cookie 访客查询“我的订单和付款”时只得到空购买提示，看不到第一位访客的订单或付款。

## 部署后数据对照

|对象|部署前|部署后|变化|
|---|---:|---:|---:|
|users|11|12|+1|
|guest_visitors|6|7|+1|
|chat_sessions|412|413|+1|
|messages|1098|1112|+14|
|commerce_purchases|2|3|+1|
|commerce_operations|0|5|+5|
|commerce_confirmations|0|1|+1|
|commerce_actions|8|8|0|
|commerce_mock_objects|3|7|+4|
|commerce_mock_receipts|0|5|+5|

变化来自两位线上验证访客和一笔已核实 mock 购买；既有 `commerce_actions` 历史行未变，新实现不再写入该表。

## 未验证范围

- 真实供应商、真实支付、真实履约或退款。
- 生产 PostgreSQL 并发竞争、容量与跨副本横向扩展。
- 真实支付认证、真实商家系统或商业开通。
- 多商家、多 SKU 购物车和真实地址配送。
