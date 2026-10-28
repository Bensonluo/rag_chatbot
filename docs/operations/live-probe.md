# 在线客服契约探针

`scripts/probe_live.py` 对正在运行的服务执行固定的中英文客服场景。完整场景集为 **21 个 case、24 轮聊天请求**，覆盖投诉收集过程中转人工、人工请求的措辞变体、退款槽位收集，以及退货政策和退款时效问答。请求串行执行；重试和身份、会话、历史、工单读取不计入这 24 轮。某轮失败后停止该 case，其他 case 继续执行。

探针会在目标服务真实创建消息和转人工工单，认证模式还会创建会话，可能消耗模型额度。请使用指定测试实例与探针账号。脚本不提供自动删票或删除客户记录的清理流程。

## 运行方式与凭证

从仓库根目录执行。以下命令假定服务运行在专用测试实例 `http://localhost:8000`；连接其他测试实例时替换 `--base-url`。凭证环境变量由密钥设施预先注入，命令只传变量名，不包含 token 值，也不打印变量内容。

匿名 demo 模式：

```bash
.venv/bin/python scripts/probe_live.py \
  --base-url http://localhost:8000 \
  --timeout 120 --pace 1
```

认证账号与归属会话：

```bash
.venv/bin/python scripts/probe_live.py \
  --base-url http://localhost:8000 \
  --auth-token-env CS_PROBE_ACCESS_TOKEN \
  --timeout 120 --pace 1
```

同时要求独立客服读取工单证据：

```bash
.venv/bin/python scripts/probe_live.py \
  --base-url http://localhost:8000 \
  --auth-token-env CS_PROBE_ACCESS_TOKEN \
  --ticket-token-env CS_PROBE_STAFF_TOKEN \
  --require-ticket-evidence \
  --timeout 120 --pace 1
```

`CS_PROBE_ACCESS_TOKEN` 是普通探针账号的有效 access token；`CS_PROBE_STAFF_TOKEN` 是具有工单列表读取权限的 active admin access token。脚本不自动登录、注册、刷新 token 或读取服务签名密钥。指定的环境变量缺失、空白或包含无效字符时，在发送请求前以配置错误退出。目标 URL 只接受 HTTP(S)，拒绝内嵌用户名密码、query 和 fragment。

聊天账号和客服观察者使用独立 HTTP client。通过 Python 接口注入 client 时，也不能把同一个对象同时用于这两个角色，以免客服凭证覆盖聊天身份；注入 client 的原 Authorization header 会在运行结束或异常退出时恢复。

## 三层结果分别证明什么

| 模式或检查 | 实际验证 | 不能据此推断 |
| --- | --- | --- |
| 默认匿名模式 | 使用随机正 31 位 ID，并避免本次运行内重复；检查返回会话 ID 与请求一致 | 服务端已预留会话、严格所有权或跨运行绝无碰撞；这是 best-effort 隔离 |
| `--auth-token-env` | `GET /api/v1/auth/me` 获取身份；每个 case `POST /api/v1/sessions` 获取服务端 ID，检查归属、规范 session_key，且本次运行内 ID/key 均未复用 | 任意指定 ID 都可使用，或管理员可以绕过聊天会话所有权 |
| owner history readback | 认证 case 聊天完成后，`GET /api/v1/chat/history` 回读该会话，携原 expected_session_key，逐轮核对实例、角色、正文、sources 和 citations | 服务重启恢复、复制一致性、备份可恢复或真实业务执行成功 |
| `--ticket-token-env` | 独立凭证只用 `GET /api/v1/handoff/tickets` 读取列表，核对工单 ID、会话 ID；已知认证 owner 时还核对 `user_id` 与原 session_key | 客服已领取、真人在线、已接通、未来响应时间或退款完成 |

认证探针只使用服务端新建响应中的实例 key，不自行生成或在冲突后采用当前新 key。聊天/历史/已知 owner 的工单回执若实例不符则失败；匿名共享路径保留原无 key 行为。第25轮会话协议及升级顺序见[会话实例合同](../design/session-incarnation.md)。

报告分别列出 owner history readback 和只读工单观察结果。未配置客服观察者时，明确输出 `Ticket persistence: NOT VERIFIED`；即使所有回复契约通过，也不能把 API 返回的工单 ID 当作独立工单读取证据。匿名模式没有认证 owner，工单观察只核对 ID 与会话对应关系。

`--require-ticket-evidence` 用于防止遗漏客服观察者配置：未提供 `--ticket-token-env` 或注入的 `ticket_client` 就以配置错误退出。提供观察者后，401/403、坏响应或未能读到本次工单都会导致相关 case 失败，不会降级为仅检查文案。

## 回复契约

每轮先验证 HTTP 200、JSON 对象、非空字符串正文、正确的正整数会话 ID、意图字符串及 metadata 对象；布尔值不能充当整数 ID。非法 JSON 或报文类型错误记录为 FAIL，不中断整轮运行。

- **转人工**：要求 `intent=handoff`、`handoff_outcome=queued`、有效正整数 `handoff_ticket_id`，且文案中的数字工单号与 metadata 一致。仅有 `ticket #` 或“工单号”字样不能通过。
- **任务收集**：要求退款或投诉意图、相应的首个 `pending_slots`，并且没有意外转接结果。自然语言中出现“订单号”或投诉问句不能替代任务状态。
- **知识问答**：要求 `faq` 或 `policy` 意图、非空来源与引用，校验引用的必要结构、版本身份、excerpt、位置范围及其与 sources 的对应关系。正文和引用 excerpt 还需符合场景中的政策锚点。英文知识场景检查正文不夹带 CJK；英文退款追问也需要有依据的时效回答。

本场景集面向仓库提供的固定示例政策：退货 7 天、获批后支付宝/微信 1–3 个工作日到账。更换知识库或政策后须审查并调整场景，不能直接把这组锚点当作任意企业的政策。公开 citation DTO 允许未知版本；本探针对这些有依据的政策回答要求可用版本身份，合法但缺版本的 document snapshot 仍会 FAIL，表示未满足本探针的证据合同。

这些检查针对结构、状态一致性和固定知识锚点。引用形式正确、有“7 天”等锚点、英文正文没有 CJK，都不能证明整段回答已获得充分证据支持；报告不会将其称作完整语义质量或真人接手证据。完整回答质量与裁判控制另见[回答质量评测](answer-quality-evaluation.md)。

## 工单读取的范围

观察者只读取列表，不会 claim、resolve 工单，也不会通过领取工单来获得历史权限。它按 `open`、`claimed`、`resolved` 依次扫描，每个状态最多 **10 页、每页 200 条**，找到本次待核对工单后可以提前结束。

这是有界、实时的列表观察。动态队列中的新增、状态迁移和偏移分页变动，以及扫描上限，都可能使本次工单未被观察到。此时 FAIL 表示没有完成所要求的读取验证，不能解释为数据库中一定不存在工单。

本探针不调用接单客服的工单历史接口。该接口的 404 可能表示未领取、已解决、分配给其他客服或不存在；不能用它替代工单存在性判断。认证 owner 的通用历史回读失败同样只能说明本次验证未完成。权限与接口边界见[客服工单上下文](handoff-context.md)。

## 超时、限流与输出

`--timeout` 必须是有限正数，默认 120 秒，用于 HTTP client 超时配置；它不是整个 case 或整次运行的总时限。`--pace` 必须是有限非负数，默认 0，表示 case 之间的等待，不是每轮之间的等待。

聊天 POST 只对明确的 HTTP 429 重试，最多重试 6 次，即最多 7 次尝试；最后一次失败后不再额外等待。`Retry-After` 支持延迟秒数和 HTTP-date；缺失、无效、非有限值或已过期日期使用 15 秒默认等待，每次等待最多 60 秒。HTTP-date 和延迟秒数的含义来自 [RFC 9110 §10.2.3](https://www.rfc-editor.org/rfc/rfc9110.html#section-10.2.3)；60 秒上限和回退等待是本探针的策略，并非 RFC 规定。

其他聊天 HTTP 错误或传输错误直接记为失败，不自动重发可能已经完成写入的 POST。会话分配和辅助 GET 不使用这套聊天 429 重试逻辑。

标准输出包含 case、固定失败原因和各层验证范围，不打印任意服务端响应正文、模型原文、异常文本或凭证。需要调查失败时，应在指定测试实例按对应请求检查服务侧记录，不通过放开控制台正文来收集真实客户数据。

| 退出码 | 含义 |
| --- | --- |
| `0` | 当前配置要求的全部 case 通过；未启用的验证层仍保持 NOT VERIFIED |
| `1` | 至少一个 case 或要求的回读、工单观察失败 |
| `2` | argparse 配置错误，例如无效数值、无效环境变量或缺少必需观察者 |

脚本只调用 REST 聊天，不覆盖 SSE 分帧、断连、浏览器、代理或生产容量；这些范围不能从一次在线探针通过中推导。离线合成与 HTTP/SSE 回归的运行方式见[回答质量评测](answer-quality-evaluation.md)。
