# 客服读取工单会话上下文

客服先领取工单，再读取该工单关联的已持久化会话。接口在 demo 和 strict 模式下均要求有效的、未停用的管理员账号，而且账号必须是该工单当前的接单人。

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

通用 `/api/v1/chat/history` 的 strict 所有权规则保持独立：客服不能仅凭管理员身份通过该接口读取客户会话。此功能也不改变已有工单摘要列表对管理员的可见范围。

## 验证

```bash
.venv/bin/pytest --no-cov -q \
  app/tests/unit/api/v1/test_handoff_history.py \
  app/tests/unit/services/handoff/test_ticket_history.py \
  app/tests/evaluation/test_handoff_context_flow.py
```

流程测试使用真实 JWT、编译图、FAQ 实现、临时 SQLite 和消息持久化，验证客户聊天、转接、领取、历史引用及解决后拒绝访问。FAQ 向量为本地替身，LLM 调用为零，不测真实模型或向量服务质量。PostgreSQL 的读锁与解决操作竞争测试位于已有 `test_ticket_concurrency.py`，通过专用的 `RAG_TEST_DATABASE_URL` 在隔离测试库执行；现有 CI PostgreSQL 步骤会运行该文件。
