# 会话实例标识与旧请求隔离

2026-10-28，第二十五轮本地实现。`session_key` 是服务器生成的会话实例标识，用于核对客户端正在操作哪个生命周期的会话；它不替代认证、所有权、接单权限或企业租户隔离。

## 修复的实际问题

原工作台以 user/session/ticket 数字 ID 保存草稿及人工重试请求。在受支持的本地 SQLite 上，通过正常 API 关闭并删除最高 ID 会话后，新建会话和工单可自然复用这两个数字。原人工消息回执随旧会话级联删除，旧浏览器请求便可能在新会话中被当作新消息执行。

独立真实 JWT/FastAPI/SQLite/SessionCoordinator 复现：客户 1 的 session 21/ticket 1 删除后，客户 2 正常新建得到同数字 ID；原获派客服 3 的旧 UUID/正文返回 200、reused=false，客户 2 可读到该旧正文。没有手工指定主键、模拟服务端结果或外部网络请求。PostgreSQL 常规序列删除不自然回收 ID；本轮未在 PG 上复现序列重置或备份恢复，不据 SQLite 宣称 PG 有相同常规行为。

## 标识与生命周期

ChatSession 新增唯一、非空、规范小写 UUID `session_key`。每次新建行由服务器独立生成，不使用共享默认值，也不接受客户端指定。既有记录迁移时逐行回填；合法已有 key 保留，损坏或重复旧 key 拒绝自动修复。

| 操作 | session_key | automation_epoch |
| --- | --- | --- |
| 新建会话 | 新 UUID | 初始 0 |
| 普通消息、读取、标题/配置修改 | 保持 | 按原规则 |
| 人工正常关单 | 保持，原已提交 POST 可精确重放 | 推进 |
| 显式清空历史 | 先换新 UUID | 同事务推进 |
| 删除后新建，即使数字 ID 相同 | 新 UUID | 新行初始 0 |

清空操作在原会话门闩内先提交 key 轮换及 epoch 推进，再清理 checkpoint 和历史。后续清理失败时可能已完成标识切换，接口不会伪称全部清空成功；旧请求仍失效，新 epoch 不读取旧 checkpoint。此过程不是 SQL 与 checkpoint 的跨存储原子事务。重新打开当前咨询后才可使用新实例；不能自动把旧草稿或旧授权搬过去。

ActionLedger 的执行记录继续保留：清空聊天不撤回业务调用，删除会话仅使其 live 引用置空，旧执行身份不能借数字 ID 复用读回或重试。会话新 key 也不表示能回滚真实付款、订单、退款等外部业务。

## HTTP 契约

Session 的 create/list/get/update DTO 返回 `session_key`；队列、claim/resolve 结果、人工 page/ACK、人工原 AI 历史和 AI 历史/REST 回复携带对应 key。公共 Demo 或内部 legacy adapter 的部分响应字段可为空；新工作台需要规范、匹配的 key 才接收正文和回执。

| 接口 | 当前实例约束 |
| --- | --- |
| 客户/客服人工 POST messages | body 必填 `expected_session_key`；缺失或格式不合法时验证失败 422（认证/角色拒绝仍可先发生） |
| 私人 human GET、客服原 AI history GET | query `expected_session_key`；授权后缺失或不符返回 409 |
| 私人 AI POST、SSE POST | body `expected_session_key`；授权后缺失或不符在响应头前返回 409，真实执行在门闩内再核对 |
| 私人 AI history GET / history DELETE | query `expected_session_key`；授权后缺失或不符 409，执行/读取在门闩内再核对 |
| 私人 session PUT | body `expected_session_key`；原值不符则拒绝，实际 UPDATE 还按 owner/ID/key 条件原子匹配 |
| 私人 session DELETE、claim/resolve | query `expected_session_key`；原门闩/事务内核对后才进行变更 |
| owner Session list / GET detail | 无正文的当前对象发现入口，不要求旧 key；保留认证 owner 校验，供显式重新打开 |
| 未标记的共享 Demo | 保留原身份与无 key 行为；若显式提供 key，仍核对匹配；不允许它绕过 private 标记 |

key 校验位于原 owner/assigned admin 授权之后；知道 key 不能获得会话。外来客户或非获派客服的 401/403/404 规则不因 key 改变。SSE 预检和执行之间发生状态竞争时，body 内层仍拒绝图/业务调用，仅返回错误控制结果，不能将 HTTP 已开始当作业务完成。

客户人工请求另保留 `expected_ticket_id` 收件约束：相同会话实例内换到新工单，也不能把原不确定请求改投新票。新工作台始终携原 ticket；旧接口省略这个 hint 的同实例路由仍按既有规则，本轮并未把它改成全局必填。

第25轮的本合同限定已初始化 HTTP/工作台列出的会话操作；内部手工构造且省略 key 的旧 service adapter 不代表远端客户端可绕过 mandatory DTO。第26轮另为私人 feedback 增加原实例核对，并为 QA verdict 增加会话实例和证据快照双重约束，见[质量写入合同](quality-mutations.md)。会话 key 不能代替质检证据版本，也不能将各接口的保护范围泛化为所有接口的统一版本协议。

## 浏览器 journal v2

v2 scope 在原 API origin+路径、已验证 user、customer/staff、session/ticket 之外加入 `sessionKey`。React Panel key、增量游标、正在打开的客服恢复 scope 也包含它。人工发送、AI 请求、历史读取、claim/resolve 均携原 key；HTTP 200 若回执 key 不匹配，也不得合并正文、游标或确认状态。

v1 不含可核验的会话实例，不能自动补当前 key。当前 API namespace、当前已验证用户存在任何 v1 envelope 时，journal 恢复/保存 fail closed；不解析或显示旧正文，不自动发送，展示通用提示及显式清理入口。

- 「清除旧的本地恢复数据」只删除当前 API/当前用户的 v1，保留 v2、其它用户和其它 API。
- 显式退出删除当前 API 在当前标签页所有用户的 v1/v2；401 仅清 support token 并卸载私人 UI，保留待同用户重新验证的恢复记录。
- 旧 v2 客服工作仍可在 /me 与角色核实后显示通用工单编号。打开/精确重试使用原 key，当前新对象不符便拒绝，不自动关联新的会话。
- 客户「刷新咨询列表」刷新当前列表页，并独立 GET 选中的 owner Session，即使它不在当前页。响应仅在原选中 ID+旧 key 仍匹配时应用；新 key 重新挂载 Panel，旧稿不搬移。404 保留原选中并报错，不自动选择其他咨询。
- 旧客户 scope 保留本地记录但没有统一的历史 scope 恢复列表入口；不能声称每份孤立草稿都能在当前界面逐项找回。

原不可变 UUID/正文/resolve/ticket、同页清理同步、迟到回执保护、较新未保存草稿及容量限制继续生效。新建/恢复 Panel 不触发 POST，移除本地副本也不删除服务端消息。sessionStorage 仍是同源脚本可读的本地明文暂存，不是加密保管或跨设备备份。

## 运维探针与升级

新 head `c53a7d9e8421` 接在 `b42f9c8e7310` 后。启动在构造模型/向量依赖前读取 session_key 列，缺迁移不就绪。ORM 和 CLI 必须提供服务器生成的每行 key；旧 raw INSERT 不再允许省略 key，没有空串或固定 UUID fallback。

1. 备份并排空旧应用实例、在途请求和 SQLite 写入者。旧应用不会强制新实例契约，不能与新版混用并声称保护已生效。
2. 在受控环境验证并应用迁移，再部署支持 key 的全部后端、工作台和 operator 客户端。新前端对旧服务的无 key 响应拒绝展示；新后端对旧私有写请求拒绝，不能宣称无缝混跑。
3. SQLite 迁移在 Alembic autocommit 维护块中，关闭该连接外键检查，开启 BEGIN IMMEDIATE，重建父表并检查 foreign_key_check，成功提交或错误回滚后恢复 FK ON。已有消息、工单及账本 live FK 均须保留。它会提交前序迁移工作，不承诺整条链单事务或零停机。
4. SQLite 在线 Python 路径测试了错误回滚与 FK 恢复；离线 SQL 执行器需人工核 foreign_key_check 输出，脚本任意中断不具备 Python finally 的清理保证。PostgreSQL 本轮只核离线 SQL，不能替代真实连接/锁/升级验收。
5. 降级在 private 会话或任何 human 行存在时拒绝。空库/仅公共数据路径可按 guard 降级，但不表示旧客户端、旧镜像或回退旧备份整体安全；执行账本等后续 guard 仍生效，优先向前修复。

`scripts/probe_live.py` 的认证模式同步校验新建会话 key 的规范性和本次运行内唯一性，所有聊天/owner history 请求带原 key，回复、历史与已知 owner 的工单旁证均核对实例。匿名共享模式不额外加 key；21 case、24 轮聊天场景及原重试/证据门槛不变。没有运行线上探针或真实生产请求。

更早数据库快照可能同时回退 key、回执及账本。本轮不承诺恢复旧快照后外部业务 exactly-once，真实业务仍需供应商幂等与对账；也不宣称新增企业租户隔离、浏览器或容量验证。

完整前后证据、默认测试及源码冻结见[本轮结果](../reviews/session-incarnation-results-2026-10-28.json)。[私人会话合同](human-conversation.md)和[执行账本合同](durable-actions.md)的其它边界继续适用。最近活动排序缺陷仍待后续处理，本轮没有实现该候选；[购买编排方案](conversational-commerce-assessment.md)仅为 proposed 评估，未新增购买或支付能力。
