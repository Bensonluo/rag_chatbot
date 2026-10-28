# 客服工作台与工单会话上下文

配套前端新增 `/support/` 登录工作台，客户可创建私人咨询，客服可领取、回复和发送最终说明后结束处理。公开演示页提供入口；当前为本地改动，尚未发布。完整身份、消息、并发和升级条件见[私人会话合同](../design/human-conversation.md)。

以下为原 AI 历史接口：客服先领取工单，再读取该工单关联的已持久化会话。接口在 demo 和 strict 模式下均要求有效的、未停用的管理员账号，而且账号必须是该工单当前的接单人。

```bash
# 使用客服账号的 JWT；工单号由创建转接请求的结果提供
curl -X POST -H "Authorization: Bearer $STAFF_TOKEN" \
  http://localhost:8000/api/v1/handoff/tickets/1/claim

curl -H "Authorization: Bearer $STAFF_TOKEN" \
  'http://localhost:8000/api/v1/handoff/tickets/1/history?skip=0&limit=50'
```

历史响应包含 `ticket_id`、`session_id`、`skip`、`limit`、`total` 和 `messages`。`limit` 范围为 1–100，默认 50；`skip` 为非负偏移。消息按创建时间、ID 从旧到新排列，返回 ID、角色、正文、时间戳及已验证的来源和引用。分页前过滤内部 system 消息，不返回工具执行详情、调试信息等任意 metadata。普通客户的消息也不会被当作引用依据。

会话 ID 来自工单，客户端额外传入 `session_id` 不能选择其他会话。分页读取是实时的；活跃会话可能继续写入消息，`total` 计数和页面内容不保证来自同一数据库快照。需要后续消息时应重新读取，不能将一次分页响应当作完整历史导出。

| 请求身份或工单状态 | 结果 |
| --- | --- |
| 无 token、无效 token、账号已删除 | 401 |
| 普通客户、停用账号 | 403 |
| 当前接单的 active admin，工单为 claimed 且有领取时间 | 200 |
| 工单不存在、尚未领取、已解决、分配给其他客服 | 相同的 404 |

成功和权限 404 响应均带有 `Cache-Control: private, no-store`。每次请求重新核对当前工单状态和接单人；解决后，后续历史请求被拒绝。PostgreSQL 下，工单锁贯穿权限检查和历史数据投影；已获准的在途响应不是可瞬间撤回的下载。SQLite 不验证 PostgreSQL 的行锁行为。

通用 `/api/v1/chat/history` 的私人会话所有权规则在 Demo 与 strict 模式下均生效：客服不能仅凭管理员身份通过该接口读取客户会话。此功能也不改变已有工单摘要列表对管理员的可见范围。

## 大额退款审核

默认退款工具遇到超过自动处理阈值的演示订单时，需要人工审核，不会生成退款单或执行付款。用户确认后，系统通过同一 HandoffService 创建或复用高优先级 `refund_threshold` 工单，保存订单、原因、金额和本次工具审计。只有持久工单结果成功才回复“人工客服请求已记录”；这不代表客服已接单、审核通过或退款完成。英文确认模板提供的 `confirm` 命令可直接使用。

工具结果中的 `status=escalated` 保持原接口兼容，只表示自动流程要求人工审核。判断请求是否实际记录应检查 `handoff_outcome`、`handoff_ticket_id` 与持久工单，不能单凭该工具字段或自然语言确认。已有工单可能保留原原因和接单人；新增退款审核证据保存在工单的 `escalation_contexts.refund_threshold` 中。

工单写入失败或未接入 handoff 服务时，回复明确说明尚未确认转接成功、尚未退款，并提供重试命令。待重试上下文保留在会话 checkpoint 中；之后输入「转人工」或 `human agent` 只重试工单，仍使用原退款的高优先级和订单，即使中间已查询另一订单，也不会再次调用退款工具。用户明确取消或放弃这条未提交申请后，该重试上下文被清除；这不取消已经创建的工单。工单成功记录后也会清除待重试标记，后续普通人工请求不继承旧退款原因。

退款工具目前为 mock，这一流程验证人工审核请求的记录与交接，不证明已接通真实支付或业务审批系统。重试上下文依赖 checkpoint 可用性；数据库与 checkpoint 同时不可用时，不能保证恢复尚未提交的请求。

## 验证

```bash
.venv/bin/pytest --no-cov -q \
  app/tests/unit/api/v1/test_handoff_history.py \
  app/tests/unit/services/handoff/test_ticket_history.py \
  app/tests/evaluation/test_handoff_context_flow.py \
  app/tests/evaluation/test_refund_manual_review.py
```

流程测试使用真实 JWT、编译图、FAQ 实现、临时 SQLite 和消息持久化，验证客户聊天、转接、领取、历史引用及解决后拒绝访问。FAQ 向量为本地替身，LLM 调用为零，不测真实模型或向量服务质量。PostgreSQL 的读锁与解决操作竞争测试位于已有 `test_ticket_concurrency.py`，通过专用的 `RAG_TEST_DATABASE_URL` 在隔离测试库执行；现有 CI PostgreSQL 步骤会运行该文件。

退款审核专项另覆盖实际工具确认、工单写入拒绝、跨订单重试、订单权限、活动工单复用、取消、正常与流式正文，以及 strict JWT HTTP 最终确认；禁止 socket 外部连接。结果及限制见[退款审核验证记录](../reviews/refund-review-results-2026-10-28.json)。这些本地 SQLite 结果不替代生产 PostgreSQL、真实支付、浏览器或代理验证。

人工消息使用独立的 `/handoff/sessions/{id}/messages`（客户）与 `/handoff/tickets/{id}/messages`（客服）增量接口，不混入上面的原 AI 历史。`resolve=true` 将最终说明、关闭和新 automation_epoch 同事务提交；旧无正文 `/resolve` 仅行政关闭，不表示给客户送达说明或确认问题解决。领取成功后机器人暂停，新请求 423；竞争中的操作 409。共享旧 Demo 工单没有私人消息通道。
