# 私人咨询与人工接续合同

2026-10-28，本地实现与验证范围。前端位于配套 `portfolio-fe` 的 `/support/`，公开 RAG 演示页提供入口；尚未发布。公开 Demo 的共享会话与登录后的私人咨询是两种体验。

## 客户和客服实际使用

客户用邮箱与密码注册、登录，服务端 `/auth/me` 确认有效身份；注册只能得到普通客户权限。新建咨询通过 `/sessions` 取得服务端分配的 ID，不使用共享 session 1。客户先咨询 AI 或申请人工；等待期间可给工单补充消息。已领取后，机器人暂停处理，双方使用明确标注作者的人工消息。客服以管理员账号登录同一工作台，查看等待与处理中队列，领取后读取原 AI 对话、引用和人工消息，发送回复，或「回复并结束处理」。

工作台的 token 存在当前标签页的 sessionStorage，与公开 Demo 的 localStorage token 分开；401 退出工作台并卸载私人内容、取消在途请求。页面不提供客户端角色切换。停止等待只取消接收，不能证明业务操作撤销；AI POST 不自动重试。人工消息送达未确认时，保留原草稿与不可变重试请求，用户可以重试同一条消息；后台轮询 GET 可以确认已提交但回执丢失的消息。

客服列表通过 `is_private` 区分共享旧工单。旧共享工单只能查看原 AI 历史和行政关闭，不能向匿名共享会话投递人工消息；关闭提示明确说明没有向客户发送处理说明。新版私人消息不会被拼进共享机器人的上下文。

## 身份和数据边界

`ChatSession.session_metadata.access_mode=private` 是服务端写入的访问标记。正常认证创建接口一律创建私人会话，客户端不能在创建或更新请求里覆盖该 metadata。未标记的旧 Demo 行继续共享。**即使 DEMO_MODE=true，私人会话仍要求实际认证 owner**；body user_id 不提供私人访问权。普通历史、清除、机器人 REST/stream 都执行该规则，机器人和历史还在实际执行时再次核对归属，防止预检查后对象变化。

知识目录、版本状态、原始/规范化全文及上传/替换/重试/删除等管理也在两模式都需 active admin，普通注册客户不能管理共享知识（见[知识访问合同](knowledge-access.md)）。质检队列及改判在两种模式下均需管理员；反馈仅允许完成的 AI assistant 消息，私人消息在 Demo 下也只能由其 owner 评分。客服管理员通过获派工单读取客户内容，不能仅凭 admin 绕过通用 chat-history 的 owner 规则。管理员队列仍可见工单摘要，现有管理权限不是租户隔离。

人工正文独立保存在 `handoff_messages`，含 session_id、ticket_id、author_user_id、author_kind（customer/staff）、client_message_id、content 和时间。角色、作者、所属会话和工单全部由服务端确定；客户不能代客服发言或关闭工单。人工消息不是 `messages` 的 assistant，不带 AI 处理状态、评分或引用，不计入机器人 served/completed/failed 统计。两张表的整数 ID 可能相同；前端以通道区分，只用 AI message_id 评分，不能声称旧 feedback 接口能辨别误传的人工作品 ID。

客户只能读写自己的私人会话；新发送时须有当前 open/claimed 工单。客服新发送和读取需 active admin 且是当前 claimed 工单接单人；已关闭工单只允许原获派客服精确重放已经提交的 UUID，不能新发或读取整段历史。每次请求重新核对，其他接单人和不存在对象统一不可访问。原 AI 历史只读，人工阅读按会话包含过去工单的人工消息。已经下载到浏览器的内容不承诺瞬间撤回；权限撤销后页面停止提供新发送控件。

认证、chat、sessions、handoff 和 feedback 路由正常及可处理的 HTTP/验证失败响应均加 `Cache-Control: private, no-store`，客户端 fetch 也禁用缓存、只发 Bearer、不带 cookie。

## 接口与可核验回执

当前会话操作还须核对 `expected_session_key`，完整必填范围、422/409 和发现/重新打开例外见[会话实例合同](session-incarnation.md)。key 不是授权；owner 与获派客服校验保留。

| 用途 | 方法与路径 | 条件 |
| --- | --- | --- |
| 客户增量读取/发消息 | GET/POST `/api/v1/handoff/sessions/{session_id}/messages` | 认证 owner、private |
| 客服增量读取/发消息 | GET/POST `/api/v1/handoff/tickets/{ticket_id}/messages` | 获派 active admin、private；读取/新发送须 claimed，原已提交 POST 精确重放可 resolved |
| 客服原 AI 上下文 | GET `/api/v1/handoff/tickets/{ticket_id}/history` | 获派 active admin、claimed |
| 接单/行政关闭 | POST `/api/v1/handoff/tickets/{ticket_id}/claim` 或 `/resolve` | admin；原接口兼容 |

人工 POST body 是 `{client_message_id: UUID, content, expected_session_key: UUID}`；客户/客服都必带规范会话实例 key，新版客户工作台另带严格正整数 `expected_ticket_id`，客服可另带严格布尔 `resolve`。客户请求在原事务/会话门闩内核对最新工单，ID 不符则在回执查找、解决信号或消息写入前返回 409；这个字段只是收件约束，不能授予会话权限。省略或 null 保留旧客户端的会话级路由，不能宣称旧客户端也有新防串票保证。服务端接受非空正文最多 8000 字符，当前页面输入限制 4000 字符。回执是 `{message, ticket, reused, session_key}`，含真实已提交 ID、作者、正文、工单状态及会话实例。前端核对实例 key、会话、作者、UUID、正文及工单，不能用错误回执宣布送达。

`(session_id, author_user_id, client_message_id)` 唯一。相同 key、正文、所属工单和结束处理意图是同一请求；即使原请求已关闭工单，客户在该票仍是会话最新工单时、或原获派客服通过固定工单接口，仍可精确重放得到原回执。变更正文、resolve 意图或把旧 key 迁移到新工单返回 409。重试原消息不会发送已经编辑的新草稿；新消息使用新 UUID，不自动重试 POST。

GET 使用 `after_id` 和 `limit`（1–100）。返回 `messages, ticket, next_after_id, has_more, session_key`；先核对实例 key，再合并消息和游标。游标是最后实际返回消息 ID，空页保留原游标，不直接跳到全局最大 ID。页面串行拉完 has_more 后每 3 秒读取增量，切换会话/退出时取消，不建立并行轮询。AI 历史与人工游标独立；客户页面展示最近 100 条 AI 消息，客服按原上下文 offset 接口读取全部页。实时分页不是不可变历史导出。

工作台在同一会话内保留已观察到的最新工单：ID 较新的票优先，同一票只按 open→claimed→resolved 前进；旧 GET、迟到的旧 POST 回执和旧 null 快照不能把工单状态回退。消息仍独立合并，GET 游标只跟随实际读取，切换会话/工单由现有 React key 重挂并从零游标开始。该策略依赖当前后端工单 ID 顺序及不回退的生命周期，不是跨系统通用事件版本协议，也不保证实时知道服务端全部变化。状态、作者、AI 角色及登录 token_type 必须是实际 JSON 字符串，不接受数组字符串化的伪造枚举。

`resolve=true` 在同一数据库事务内插入最终客服正文、关闭工单并推进 automation_epoch；任一步失败都不确认完成。旧 `/resolve` 仍是无正文行政关闭，不能据此宣称已送达说明。两种关闭均不设置客户问题的 resolved_at，不能当作用户已确认解决。

客户在人工通道发送的新消息使用与 AI 通道相同的保守解决规则：明确全句确认才设置解决时间，否定、疑问、引用或附带新请求不算确认；已解决会话收到其他非空客户消息会清除解决时间并将 reopened_count 增加一次。规则仍按整个会话判断，不能辨别是不是同一问题，普通致谢也不作为解决确认。只有实际提交的新 customer 消息执行这个状态机，客服正文及关闭不执行；精确 UUID 重放不会刷新时间或再次重开，更不能用旧确认覆盖新求助。正文和结果信号同事务，插入失败同时回滚。

人工确认可以成为已有机器人服务会话的解决证据，但不增加机器人 served/completed/failed 分母；只有人工消息的会话即使有解决时间，仍不冒充机器人服务会话。

## 接管、并发和恢复

SessionCoordinator 对同一个会话串行化机器人图、工具调用、持久化、流式取消收尾、接单、人工写入、关闭、历史清除和会话删除。已经开始的机器人先完成；在其实际操作或收尾未结束时接单返回 409，不能先宣称接手再放任旧业务继续。接单已完成时，新机器人 REST/stream 返回 423，不进入缓存、图或工具。SSE 预检查与响应体之间仍可能发生接单；响应体重新取得权限，失败只发 STREAM_ERROR，可能带原传输 DONE，不发成功 completion 或普通正文。

生产 PostgreSQL 使用 transaction advisory try-lock，namespace 固定、ID 为正 31 位整数；锁有单独连接池（pool_size 取 DATABASE_POOL_SIZE，max_overflow=0，pool_timeout=1 秒），不占用嵌套业务查询的连接池，也不对父会话行持有跨图 FOR UPDATE。锁获取预算 1 秒，忙返回 409，无法建立协调返回 503。机器人轮次沿用配置的 CHAT_STREAM_MAX_SECONDS，流式收尾另有既有时限。锁连接数和吞吐尚未做容量验收。

SQLite 仅提供同一运行引擎、同一进程内的互斥，不是跨进程或多副本证明。仓库已把独立锁连接与业务连接竞争测试加入现有 `test_ticket_concurrency.py`，CI 的隔离 PostgreSQL 步骤会执行，但本轮本机没有真实 PG 执行结果。

关闭工单把 automation_epoch 推进，后续机器人使用新的 checkpoint thread；原 AI 历史仍可阅读，旧待确认退款不能被一句「确认」恢复。人工内容以明确引用的用户上下文补充到新机器人轮次，不能冒充工具执行结果或高优先级指令。清除/删除在同一会话门闩内清掉所有旧 epoch 的 checkpoint；删除前清理失败则不确认删除，防止旧私人状态被 Demo ID 重用读回。数据库行与 checkpoint 清理不是跨存储原子事务，失败可能已部分清理，但不能返回虚假的完成结果。

连接丢失可能释放数据库 advisory lock，无法撤销已经发送给外部业务系统的请求。**真实支付和业务接口仍须提供业务幂等、权限、结果查询或 fencing**；当前 mock 工具验证不能证明外部退款恰好一次。工作台/接口不承诺有人在线、响应时间或审核通过。第二十三轮在此基础上增加稳定提案 UUID 与独立执行账本，阻止同提案因 checkpoint 故障再次调用；未知结果不自动重试，详见[确认业务合同](durable-actions.md)。

## 迁移与部署

本合同引入的 revision `a31e8d7f6290` 接在 `d4c9f1a82e70` 后，增加 automation_epoch（默认 0）、人工表、外键、作者约束、幂等唯一约束和会话游标索引；支持已有 create_all schema 的采用。启动初始化在连接模型/向量依赖前读取必需列和表，缺迁移时不就绪。

1. 备份并验证恢复；停止/排空所有旧应用实例，暂停对外写入。
2. 运行 `alembic upgrade head`，部署全部新版应用，确认就绪、认证和私人会话访问再开放流量。不能让旧版实例与新版私人会话共存：旧 Demo 二进制会忽略 private 标记。
3. 在任意私人会话、人工消息或非零 automation_epoch 存在时，降级会拒绝。不能绕过 guard 或直接回滚旧镜像；应保留新 schema 和隐私保护，部署向前修复。若确需旧版，必须另行制定并核验数据与权限迁移方案。

SQLite 在线检查后可对空库往返；SQLite 离线 downgrade 拒绝生成 DROP，要求在线检查。PostgreSQL 在线/离线 downgrade 都在排他锁下检查数据再 DROP；本轮仅对 PG SQL 生成进行了验证。数据 guard 不能阻止运维者直接启动旧二进制，排空旧实例是部署条件。

## 验证与剩余工作

真实 JWT/FastAPI、外键 SQLite、编译图、项目 mock 退款、实际持久化及消息失败触发器覆盖双向闭环、精确重放、接管竞争、SSE 竞争、隐私、指标和旧确认隔离；全部阻止外部 socket。React 测试通过实际 SupportApi/fetch 边界控制 HTTP，验证注册/角色、队列、双方消息、丢回执、取消、失效登录、增量游标和伪造响应。

这些结果不包含真实 PostgreSQL advisory lock、多副本、真实模型、业务/支付系统、远程 CI、浏览器视觉交互或生产容量。原始双向会话与迁移成绩见[第20轮证据](../reviews/human-conversation-results-2026-10-28.json)；客户后续消息与页面状态一致性的当前成绩见[第21轮证据](../reviews/human-followup-results-2026-10-28.json)。


## 草稿与待确认发送恢复（第二十四轮）

本节保留第24轮引入的 v1 设计及历史验证；现行 v2 增加 sessionKey，旧 v1 不自动采用当前 key，按[会话实例合同](session-incarnation.md)隔离和显式清理。

当前工作台按 API origin+路径、已验证 user、customer/staff、session，以及客服 ticket 分开暂存草稿、修订号、草稿原收件通道和不可变人工尝试。数据存在当前标签页 sessionStorage，不保存额外 token、完整历史或错误诊断。每身份最多 20 个 scope、每 scope 5 条待确认尝试、每条 4000 字符、整个身份 envelope 最多 256 KiB UTF-8；满额/坏格式不静默淘汰。新人工发送必须先保存 UUID、原正文、resolve 和原 ticket，再 POST；保存失败不发起新 POST，保留当前页草稿并提示。通用保存错误不能推断此前消息未送达。

刷新、换会话或 401 后，同一用户重新经 me 和当前 scope 读取授权后才恢复正文，不自动 POST。401 卸载私人内容但保留待同用户核实的记录；显式退出清除此 API 在当前标签页的全部用户 journal，清理失败明确提示。其它账号、角色或未获授权的 scope 不能显示旧正文。清除本地重试只移除恢复副本，不撤销或删除可能已经提交的消息。

非空草稿保留原 AI/人工通道及具体工单。工单关闭、新工单出现或 AI 草稿期间转人工，都需要显式移动草稿，再单独发送。不可变原请求不会随草稿移动；客户 POST 的 expected_ticket_id 在轮询尚未发现工单变化时也会防止串票。已确认回执或 GET 清除同 UUID；迟到回执不覆盖更新草稿。同页客服恢复入口与旧 Panel 同步 canonical journal，即使草稿 revision 相同也不复活已删尝试；空记录删除时只保留更新且尚未保存的本地草稿。

客服的已保存工作入口不依赖活动队列。先核对有效管理员身份，仅显示通用工单编号；打开草稿须当前获派 GET 成功。closed GET 的 404 不是已关闭证明，也不能剥夺原 POST 精确重放：客服可明确选择原消息重试，它若此前未送达且当前仍允许发送，可能实际发出原消息。服务端精确重放先核对原 resolve 意图；客户端只在回执作者、会话、工单、当前接单人、UUID、正文匹配，且结束处理尝试得到 resolved 状态后，才展示该条服务端正文、条件移除本地记录，不重新打开工单或展示整段已关闭历史。非当前打开 scope 可显式丢弃本地保存工作，服务端不受影响。

sessionStorage 不是加密隔离、服务端备份或跨设备同步；同源脚本及浏览器会话恢复仍是边界。MDN 说明刷新可保留数据，有 opener 的新标签页可得到初始副本，因此不承诺各标签页绝无同一请求副本，服务端 UUID 仲裁仍不可少。设计参考：[MDN sessionStorage](https://developer.mozilla.org/en-US/docs/Web/API/Window/sessionStorage)。这不是实际浏览器验收，本轮工具明确返回 No browser is available。

第24轮没有新增数据库 schema、迁移或依赖；当轮要求先升级后端支持 expected_ticket_id，再升级工作台。第25轮另需 c53a7d9e8421 和工作台/operator 实例键协议，见[升级合同](session-incarnation.md)。旧后端严格 extra-forbid 会拒绝新版多出的字段，不能让该错误被当作已送达。此前私人会话和执行账本迁移/排空旧实例要求仍适用。完整本地回归、真实 SQLite/JWT 与 React 传输边界、源码 SHA 和限制见[续接验证](../reviews/support-continuity-results-2026-10-28.json)。


第二十五轮为会话生命周期增加服务端 session_key。正常关单保持 key，显式清历史先轮换 key 并推进 epoch；同数字 ID 的新会话不能接收旧 key 的私有请求。新前端绑定原 key 且不迁移旧恢复正文，运维探针也按原实例核实。详情和当前验证见[实例隔离合同](session-incarnation.md)及[第25轮结果](../reviews/session-incarnation-results-2026-10-28.json)。
