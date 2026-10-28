# 回答身份、处理状态与用户反馈

2026-10-28。本文定义聊天接收与反馈归属，不把消息处理完成解释成客户问题解决或业务操作成功。

## 服务端身份从哪里来

`ChatMessagePersister.persist_turn` 将本轮用户消息、assistant 消息及会话解决状态变更放在同一数据库事务中。assistant flush 获得主键后，只在 commit 成功时返回不可变回执；不通过查询“最后一条历史”寻找消息。事务失败返回空回执，保留聊天路径既有的持久化 fail-open 行为。提交后的压缩/事实提取调度失败不撤回已经确认的回执。

`POST /api/v1/chat` 增加可空 `message_id`、`status`。未配置 persister、没有回答或持久化失败时，字段为空；收到回答本身不能替代持久化证据。

流式客户端可在请求体设置 `include_completion: true`。默认客户端不增加事件。正常流在持久化成功后、`[DONE]` 前发送：

```text
event: completion
data: {"message_id":431,"session_id":7,"status":"completed"}

data: [DONE]

```

这是命名元数据，不能进入回答正文。错误、超时、取消和最终持久化清理超时不发送完成回执；有部分正文的失败流仍按既有规则保存为 FAILED。数据库已提交后网络丢失确认仍可能发生；本协议不提供业务幂等重试保证，客户端不自动重发聊天 POST。

`GET /api/v1/chat/history` 为每条数据库消息增加 `message_id`、`status`、`user_rating`。内存历史没有可靠身份时保留为空，未知状态不补成完成。来源引用仍是回答时的快照。

## 前端使用回执的条件

实际 companion 前端验证正安全整数 ID、请求会话相符和 `status=completed`，直到收到完整 `[DONE]` 且正文非空才释放回执。坏回执、冲突回执、回执后的额外正文、EOF、停止接收和错误均不能获得评分资格。无回执的旧服务仍能展示已接收的回答，不出现伪造评分入口。

历史仅对身份有效且处理完成的 assistant 回答提供反馈。重复 ID 不推断对应关系；同样的正文可有不同 ID，评价始终指向对应消息。欢迎内容、未知历史及失败回复不能评分。

## 写入反馈与授权

`POST /api/v1/feedback` 沿用现有 active-user 认证，strict 模式继续核对会话所有权，外人、非 owner 的管理员和不存在的消息均为 404。Demo 原有存在性模式不升级成匿名评分许可。Repository 写入进一步限定为 **COMPLETED ASSISTANT**；USER、SYSTEM、FAILED、PENDING、PROCESSING 不接受评价。拒绝路径不计评分指标，也不驱逐缓存。

第26轮私人评分另须提交原 `expected_session_key`；授权后缺失或不符返回409。服务端先定位，再于会话门闩内读取最新权限和实例，并以同一 SQL 条件绑定消息、父会话、owner、key 与完成状态。只有提交后的不可变回执才能触发评分指标和缓存驱逐；驱逐在门闩外使用原回执正文，不重读可能已被替换的数字消息 ID。详见[质量写入合同](quality-mutations.md)。

评分保留 `-1`、`0`、`1` 兼容值；`0` 是原协议清除值，不是好评。新工作台只在 `success=true`、消息 ID、rating、session_id 和原 session_key 全部匹配时更新显示；公开共享 Demo 客户端保留原 ID/rating 兼容协议，不据此声称具有私有实例保护。每条消息有同步提交门闩，失败显示未确认，允许用户核实后手动重试，不自动重发。

当前公开页面仍使用共享 Demo 会话，并无新增登录入口。没有既有访问令牌时，实际可评分的回答提示登录，不发送匿名反馈；strict 模式还要求访问令牌拥有该会话。本轮没有放宽权限以恢复按钮。真实支付、客户最终解决、匿名登录体验、人工发送/接收及多租户知识隔离仍有独立交付边界。

历史验证见[第十八轮记录](../reviews/message-feedback-results-2026-10-28.json)：当轮31 项新增单元和 22 项实际 SQLite/JWT/编译图 HTTP 回归均通过；当轮完整默认 3063 通过、3 opt-in 跳过，核心覆盖 93.29%。第26轮的当前实例、并发及证据版本结果见[质量写入验证](../reviews/quality-mutations-results-2026-10-28.json)。这些本地结果不等于真实 PostgreSQL、Redis 驱逐、模型质量或浏览器交互验收。
