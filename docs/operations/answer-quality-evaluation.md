# 回答质量评测

本产品的目标是让访客得到有依据、可执行、如实描述处理结果的客服回答。评测分别检查路由、检索、引用/持久状态、要点覆盖和整段回答的证据支持及语言一致性，不能用某一项通过代替实际解决问题。

## 无网络回归

```bash
.venv/bin/pytest --no-cov -q app/tests/evaluation
.venv/bin/python scripts/run_pipeline_eval.py --mode recorded --output /tmp/pipeline-contracts.json
```

第二条命令运行实际 `ChatService` 和编译对话图、生产 BM25 实现、引用构建、真实 SQLite 消息/工单持久化。10 个双语场景共 14 轮包括知识问答、无证据、转接成功、数据库拒绝创建工单，以及政策→转人工→再次政策问答。每个场景创建并清理独立临时数据库，故障通过 SQLite 插入触发器产生。另用两套独立图和临时库实际执行 `human agent` / 「转人工」，验证命令能力；这些探针不是当前会话的处理结果，也不计入 14 轮主场景分母。

`recorded` 使用从模型提示中摘录资料的确定性替身，**通过只表示路由和产物契约通过，不是回答语义质量成绩**。不使用生产数据库、FAQ、向量检索、重排、Redis、Agent 外部工具或 PostgreSQL checkpoint；额外通过真实 JWT 请求隔离的历史 HTTP 路由，记录 demo/strict 下管理员读取他人会话的实际状态、会话内容和无效 token 的 401；这不覆盖完整应用中间件、聊天 HTTP/SSE 或部署配置。这些限制也写入 JSON 报告。不是完整生产配置的端到端评测。

## HTTP/SSE 与会话写入契约

独立的路由回归补充上述评测未覆盖的聊天 HTTP/SSE 边界：

```bash
.venv/bin/pytest --no-cov -q \
  app/tests/evaluation/test_strict_chat_ownership.py \
  app/tests/evaluation/test_http_stream_integrity.py \
  app/tests/unit/services/chat/test_stream_cancellation.py
```

这些用例使用实际 JWT 校验、临时 SQLite、真实编译对话图和持久化服务；模型只在 provider 边界提供本地确定性输出。strict 模式下普通用户和管理员都只能向自己的会话发送消息，外部和不存在的会话在进入图、发送 SSE 头或写入 checkpoint 之前统一返回 404；身份冲突仍为 403，匿名仍为 401。demo 的匿名行为保留。

正常流逐字比较消费端解码的正文、持久消息及历史，覆盖英文句号、中文无标点尾段、跨块 PII 脱敏、LF 换行/空行和版本引用。CR/CRLF 按 SSE 规则规范化为 LF，原始 CR 不能注入额外事件；不宣称保留 CR 字节。断连专项还使用实际 ASGI 2.3 `http.disconnect` 与 2.4 发送失败，检查已经发送的部分正文在请求结束前保存为 FAILED，取消继续传播，生产者关闭。流式收尾有两秒上限；正常流写入超时会发送 `[STREAM_ERROR]` 并抑制最终引用，断连清理超时只记录日志。数据库无法及时完成时不能保证持久化成功。`[DONE]` 只表示传输结束，客户端须保留此前的错误状态。没有把直接调用生成器的关闭测试冒充实际 HTTP 断连证据。

结果记录见 [HTTP/SSE 边界结果](../reviews/http-stream-results-2026-10-23.json)。这些合成路由测试不测真实模型语义、生产 PostgreSQL checkpoint、反向代理、浏览器连接或全应用部署；也不会改变原有模型评测报告的覆盖范围。

## 真实模型与裁判

```bash
# 先检查裁判的正例、反例和提示注入控制
.venv/bin/python scripts/run_pipeline_eval.py --mode live --calibration-only \
  --judge-provider settings --output /tmp/judge-controls.json

# 运行真实模型图和语义裁判；可选择独立 provider 降低同模型偏差
.venv/bin/python scripts/run_pipeline_eval.py --mode live \
  --provider settings --judge-provider minimax --threshold 1 \
  --output /tmp/pipeline-live.json
```

`settings` 使用应用默认模型工厂，包括已配置的故障切换；显式 provider 使用该 provider 的配置。实际返回的模型名称、调用/错误次数写入报告。调用只发送合成 fixture；不读取真实客户会话。命令会产生模型费用。每次模型调用和图运行均有超时，控制台不输出 provider 异常内容或推理过程，完整合成回答写入权限为 0600 的指定报告。

真实模式首先执行 23 个裁判控制，覆盖政策依据、失败说明和拒答、语言一致性、队列位置、客服接单、历史访问，以及独立命令探针成功和当前请求失败的区别。正例包括真实记录对应的排队/接单状态、已观察到的管理员历史访问，以及失败后的可用重试命令；反例包括虚构赔偿/政策、错误队列位置、未接单却声称客服接手、访问被拒绝却声称能看历史、能力探针冒充当前成功、无依据的立即接通承诺和裁判指令注入。历史中重复一条虚构赔偿也不能为它自身提供事实依据。

每个控制同时检查整体通过结论、语言判定及对应的事实/动作/承诺维度，不能因错误理由碰巧拒绝而通过校准。任何控制不符合预期或裁判报文无效，均非零退出并停止后续质量门禁；不会把解析失败当成正确识别反例。这些合成控制只是最小能力检查，不能估计裁判在真实分布中的准确率。

通过控制后，裁判接收本轮实际检索快照、工单/工具结果，以及独立观察的队列和历史访问结果。排队位置从仓储重新读取，并与原始工单排序核对；工单 status、assigned_to、claimed_at 决定是否真正接单。历史内容只证明保存和访问结果，不能用其中的机器人陈述为自己的事实或动作背书。分别判断整段事实是否有依据、动作成功声明是否真实、拒答是否恰当、承诺是否受支持、解释和建议输入的命令是否符合预期语言，并检查预期要点。语言判定允许必要的专名、标识符、URL 和简短原文引用，但英文回答建议输入「转人工」仍应失败。预期要点不作为事实证据。回答即使答全要点，附加虚构赔偿仍失败。每个要点始终进入分母，裁判错误记为未通过。契约失败和模型调用错误不可用较低语义阈值放行。

报告以 `rubric_version: 3-operational-evidence` 标识新增操作证据；裁判必须返回布尔值 `language_consistent`，缺失或字符串值均按无效报文失败。旧报告保留原义，不能据此宣称通过新增操作证据要求。可用命令只证明能提出请求，不保证当前创建成功、客服在线、立即接通或未来完成时间。

报告标明模型裁判是有误差的测量，以及生成/裁判模型是否重合；相同模型存在相关偏差，模型名称不同也不能证明独立。通过这些合成场景不等于真实用户解决率或上线容量达标。评测设计参考 [Anthropic 的 agent eval 指南](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) 中的执行轨迹、结果与多类裁判，以及 [Ragas 的 faithfulness 定义](https://github.com/vibrantlabsai/ragas/blob/main/docs/concepts/metrics/available_metrics/faithfulness.md) 对证据支持与要点覆盖的区分。

## 既有门禁的含义

- `run_answer_eval.py`：直接给定 FAQ 正确上下文后的语义要点覆盖，**不测生产检索或整段事实支持**。生成/裁判错误保留分母并导致非零退出。
- `run_retrieval_eval.py`：检索指标门禁；拒绝 NaN、Inf、越界指标及阈值，不将无效结果当作通过。
- `claim_eval.py`：规则事实拦截。按实际被检索、选中 subject 并触发数字违规的事实 ID 统计覆盖，不能通过新增同 topic 未测事实扩充分母却保持全覆盖。空/坏/重复 golden 直接拒绝。

新增 fixture 应同时提供正确回答与容易误判的错误回答。应保留失败报告，先区分应用行为、裁判失效和 fixture 错误，再修复并重新运行，不降低阈值来隐藏失败。
