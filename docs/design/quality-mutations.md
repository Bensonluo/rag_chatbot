# 反馈与质检写入的会话及证据绑定

2026-10-28，第二十六轮本地实现。目标是让评分与对应回答实例绑定，让人工质检裁决与服务端审核提案绑定，避免清空、重建或新增问题后把旧判断写入新对象。会话 key、质检版本和管理员角色分别核对，任何一项都不能替代其它项。

## 实证与范围

在第25轮冻结源码上，以真实 JWT/FastAPI、外键 SQLite、编译图和正常 clear/delete/create 操作复现四项问题：

1. 清空后 key 已轮换、消息 ID 自然复用，旧评分仍返回 200，改变新回答的 rating，并增加评分计数及触发缓存淘汰调用。
2. 在旧 owner 的真实授权查询后仅加入时序暂停；原 owner 正常删除会话，再由另一客户新建并发送消息得到相同 session/message 数字 ID，暂停请求恢复后仍返回 200 并修改新 owner 的回答。顺序执行的旧 owner 对照为 404，说明单次先查归属不足以保护随后裸 ID 写入。
3. QA 队列只给数字 session ID。旧会话删除重建后，旧裁决可把另一客户的新会话标为 resolved，使其退出抽检队列并改变 resolved 统计。
4. 同一会话 key 不变，但客户已提出新问题，消息水位由 2 变为 4；针对旧问题的 QA 裁决仍能把当前会话标为 resolved。只绑定实例 key 不能证明审核证据仍相同。

未手工指定被复用对象的主键或伪造授权/业务结果。缓存边界只记录实际淘汰函数的入参，不证明远端 Redis 已删除哪些 key。SQLite 的自然 ID 复用不是 PostgreSQL 普通序列删除后复用的证明。本轮真实 PG、跨进程锁、浏览器和生产验收均未完成。

## 回答反馈

`POST /api/v1/feedback` 保留 `message_id`、`rating`、可选 `text`，新增可选格式字段 `expected_session_key`；服务端按目标的访问模式决定是否必需：

| 目标及请求 | 规则 |
| --- | --- |
| 私人会话，两种运行模式 | active owner 才可评分；缺 key 或 key 不符为 409 |
| 未标记的共享会话，Demo 模式 | 保留认证用户的共享评分及无 key 调用 |
| 未标记的共享会话，strict 模式 | 仍要求 owner；没有新增无 key 的跨 owner 许可 |
| 任一目标显式提供 key | 核对该 key；格式不合法为 422 |
| 未登录、外来 owner、缺失或不合格消息 | 原认证/归属拒绝及 404 语义保留；不能靠 key 获得访问权 |

只有 completed assistant 行可评分；user、system、pending、processing 和 failed 行不可用。已授权私人目标缺 key/不符 key 的检查发生在归属与资格核对后；非法字段的请求模型验证可能先返回 422。

一次请求依次执行：

1. 以真实认证用户查询授权且合格的目标，取得 message ID、父 session ID、服务器 key 和 private 标记的标量快照，不依赖 ORM identity map 中的旧对象。
2. 结束发现阶段的只读事务，再取得原 session 的 `SessionCoordinator` 门闩；与聊天、清空、删除及其它质量写入使用同一协调域。
3. 门闩内重新查询原父会话下的目标，重验 owner/shared 许可和消息资格；当前 key 必须匹配初次发现的 key，再检查客户端 key。共享无 key 请求也不能在同一次请求内自动采用替换对象。
4. 单条条件 `UPDATE` 同时绑定 message ID、父 session ID、观测 key、owner 或显式 shared 许可、assistant/completed。状态在预读后改变也不能退回裸 ID 更新。SQL 未匹配返回 404，不产生评分信号或缓存淘汰。
5. 从 `RETURNING` 标量构建不可变回执并提交，然后释放门闩；在门闩外增加 `FEEDBACK_RATINGS`、对负评调用缓存淘汰并准备 HTTP 回执。后续只使用已固定的原 ID、key、rating 和正文，不重读可能已重建的消息。失败或未确认的数据库写入不据预读结果宣布成功。

回执是 `{success, message_id, rating, session_id, session_key}`，包括真实已提交的父会话实例，rating 0 原样保留。新私人工作台同时核对消息 ID、rating、session ID 和原 key，错实例的 HTTP 200 不展示为评分成功。保留原前三字段供兼容客户端读取，不承诺拒绝额外字段的旧客户端可以无缝升级。

门闩忙返回 409，协调不可用返回 503；不能因此换用最新 key 自动重发。正常重复评分仍是一次新的已接受提交：数据库保存最新 rating，计数器记录提交次数，不是唯一投票人数或消息级幂等账本。缓存淘汰继续保持原 best-effort 边界，成功评分不等于远端所有缓存已清除。

独立整合验证曾发现，已提交评分后的远端淘汰若仍持会话门闩，会使同期 clear 返回 409；新增真实时序回归后将门闩缩到数据库提交。缓存仍可能影响该评分请求的返回时延，本轮没有增加全局 cleanup budget，但它不再独占咨询的协调门闩。清空发生在评分提交后，也不会改变该旧回执所证明的已提交事实。

## QA 提案、审核与纠正

QA 的 queue、GET proposal 和 POST verdict 都要求 active admin，公共 Demo 也不例外。本轮没有建立企业租户隔离；管理员可审核的范围仍按现有全局运营权限，不把 session key 当租户凭证。

| 接口 | 用途与关键字段 |
| --- | --- |
| `GET /api/v1/sessions/review-queue` | 返回仍符合抽样条件的提案，每项有 `session_id`、`session_key`、`review_version`、bot/human 水位、预览与当前 QA/解决状态 |
| `GET /api/v1/sessions/{id}/review` | 显式获取当前提案，也允许获取已审核会话以纠正旧判断；不是自动把旧命令升级为新授权 |
| `POST /api/v1/sessions/{id}/review` | body 为 `resolved`、可选 `note`、`expected_session_key` 和 `expected_review_version`；提交前重新验证当前证据 |

两个 expected 字段在请求模型中可省略，但执行时缺失、旧实例或旧版本均拒绝 409；key 格式或严格 64 位小写十六进制版本格式错误为 422。合法但不存在的版本也不是提交许可。缺失目标为 404；非管理员不因知道提案而获得审核权限。

`review_version` 是服务器生成的 opaque 摘要，客户端保存并原样提交，不自己根据显示文字重建。它覆盖：

- owner、session key、automation epoch、resolved_at、reopened_count 和当前单条 QA 裁决。
- 当前所有 bot 存储行的有序 `id/role/status/user_rating/feedback_text/created_at` 投影摘要，包含反馈变化；不是只取最新消息 ID，也不是消息全文正文哈希。
- 最新 human message ID，以及工单 ID、状态、接单人。

正常 API 的新 bot 问题、反馈改变、新人工消息、工单状态、实例轮换和另一位 reviewer 的裁决都会让旧提案失效。system 摘要行也在 bot 投影中，新增摘要可保守地使版本变化。普通标题变化不充当新的审核证据。bot/human 正文沿用不可变消息协议；本版本不承诺检测任意绕过应用的 SQL 正文原地修改，human 部分也不是所有历史正文的逐行摘要。

POST 在同一会话门闩内重读并比较 key/完整版本，再执行条件更新。SQL CAS 额外绑定 owner、key、epoch、解决/重开状态、bot/human 最大 ID 及先前 QA reviewed_at；它不是在 SQL 内重新计算全文哈希。正常反馈、聊天和人工写入共用门闩，因此不会在版本检查与提交之间改变证据。该并发保证要求所有正常 writer 遵守协调域，PG 仍需实际连接验收。

成功 POST 返回当前 `session_id/session_key/review_version`、bot/human 水位、`resolved` 和 `resolved_at`。返回的版本已包含这次裁决；再次纠正须携当前版本，可通过明确 GET 获取。旧提案重放拒绝而不覆盖较新的判断，也不把“重新打开审核”描述成已经保存。

GET proposal 返回预览与版本，不是全量 bot/human 正文导出。opaque 摘要也不是审核者已阅读全部历史或判断语义正确的证据；完整运营审核体验与读取权限仍按对应接口和后续交付范围评估。

QA 裁决继续覆盖 `session_metadata.qa_verdict` 的单条当前值；resolved 裁决设置 resolved_at，unresolved 纠正清除它，两者都不增加客户 reopened_count。这个字典不是完整审计日志，也不能证明每个历史审核动作或审核者身份均已独立留存。

## 抽样、一致性与开销

review queue 仍为有界 best-effort 抽样：初选窗口内已服务、无工单、无负评、无解决证据且未审核的会话；按既有候选上限与随机抽样策略取样。每个候选再在门闩内核当前资格和提案，忙会话跳过，协调不可用返回错误；少于 limit 不代表其它会话都已解决，也不是全量 census。

消息投影通过 `yield_per=256` 分批流式读取并持续更新 SHA256，不把全部投影一次载入应用列表。但每次仍遍历该会话所有 bot 消息，DB 工作量为 O(N)；工单行也会读取。256 是取数批次，不是最多只检查 256 条，也不证明数据库驱动、吞吐、跨副本或生产容量已达标。当前方案未增加持久 review-version 列或新索引。

## 升级与验证范围

本轮没有新数据库迁移，沿用第25轮 `c53a7d9e8421` head。先完成既有实例迁移并升级后端，再升级私人工作台与 QA 调用方；私有评分旧客户端无 key 会拒绝，QA 旧客户端缺提案版本也会拒绝，不兼容混跑不能作为成功。

本轮新增默认后端 51 项：15 项反馈 unit 和 36 项独立真实 JWT/API/SQLite 评估（17 feedback、19 QA），包含评分提交后缓存等待不再阻塞咨询的时序回归。前端当前完整 312 Jest + 55 Node = 367；相对第25轮净 13，其中 7 项属于本轮评分保护，另 6 项来自同步出现并保留的 locale 修改，不能全归为本轮作者新增。React/jsdom 和受控 fetch 不代替真实浏览器或前后端实网验收。

最终完整后端为 3581 passed、3 skipped、5 warnings，204.69s；既定核心 93.57%（5553 statements/357 missed），70% 门槛未变，不据此宣称全部新质量 API 的覆盖率。Ruff 维护范围、app+probe_live 的 478 文件格式检查、strict Mypy app477+三 CLI、pip check 通过。初轮 3580 项全量是缓存门闩修正前的历史结果；新增真实回归后基于最终源码重新跑完整门禁。

类型/静态范围、502 BE/86 FE 源码冻结、14 BE/6 FE 增改及前后证据见[本轮结果](../reviews/quality-mutations-results-2026-10-28.json)。既有[会话实例合同](session-incarnation.md)和[评分消息身份合同](message-feedback.md)继续适用。

活动时间、列表计数与自动刷新候选本轮延期；购买编排仍是独立方案和下一阶段工作，本轮没有实现购买或支付，也没有调用真实交易接口。没有安装包、修改依赖门槛、执行生产迁移、提交、推送或部署。
