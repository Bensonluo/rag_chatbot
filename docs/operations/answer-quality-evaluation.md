# 回答质量评测

本产品的目标是让访客得到有依据、可执行、如实描述处理结果的客服回答。评测分别检查路由、检索、引用/持久状态、要点覆盖和整段回答的证据支持，不能用某一项通过代替实际解决问题。

## 无网络回归

```bash
.venv/bin/pytest --no-cov -q app/tests/evaluation
.venv/bin/python scripts/run_pipeline_eval.py --mode recorded --output /tmp/pipeline-contracts.json
```

第二条命令运行实际 `ChatService` 和编译对话图、生产 BM25 实现、引用构建、真实 SQLite 消息/工单持久化。10 个双语场景共 14 轮包括知识问答、无证据、转接成功、数据库拒绝创建工单，以及政策→转人工→再次政策问答。每个场景创建并清理独立临时数据库，故障通过 SQLite 插入触发器产生。

`recorded` 使用从模型提示中摘录资料的确定性替身，**通过只表示路由和产物契约通过，不是回答语义质量成绩**。不使用生产数据库、FAQ、向量检索、重排、Redis、Agent 外部工具或 PostgreSQL checkpoint；不覆盖 HTTP 鉴权/SSE。这些限制也写入 JSON 报告。不是完整生产配置的端到端评测。

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

真实模式首先执行八个裁判控制：支持充分的政策、正确的失败说明和拒答，以及虚构赔偿、错误政策版本、失败后谎报成功、无证据编造和回答中的裁判指令注入。任何控制不符合预期或裁判报文无效，均非零退出并停止后续质量门禁；不会把解析失败当成正确识别反例。八个控制只是最小能力检查，不能估计裁判在真实分布中的准确率。

通过控制后，裁判接收本轮实际检索快照和工单/工具结果。分别判断整段事实是否有依据、动作成功声明是否真实、拒答是否恰当、承诺是否受支持，并检查预期要点。预期要点不作为事实证据。回答即使答全要点，附加虚构赔偿仍失败。每个要点始终进入分母，裁判错误记为未通过。契约失败和模型调用错误不可用较低语义阈值放行。

报告标明模型裁判是有误差的测量，以及生成/裁判模型是否重合；相同模型存在相关偏差，模型名称不同也不能证明独立。通过这些合成场景不等于真实用户解决率或上线容量达标。评测设计参考 [Anthropic 的 agent eval 指南](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) 中的执行轨迹、结果与多类裁判，以及 [Ragas 的 faithfulness 定义](https://github.com/vibrantlabsai/ragas/blob/main/docs/concepts/metrics/available_metrics/faithfulness.md) 对证据支持与要点覆盖的区分。

## 既有门禁的含义

- `run_answer_eval.py`：直接给定 FAQ 正确上下文后的语义要点覆盖，**不测生产检索或整段事实支持**。生成/裁判错误保留分母并导致非零退出。
- `run_retrieval_eval.py`：检索指标门禁；拒绝 NaN、Inf、越界指标及阈值，不将无效结果当作通过。
- `claim_eval.py`：规则事实拦截。按实际被检索、选中 subject 并触发数字违规的事实 ID 统计覆盖，不能通过新增同 topic 未测事实扩充分母却保持全覆盖。空/坏/重复 golden 直接拒绝。

新增 fixture 应同时提供正确回答与容易误判的错误回答。应保留失败报告，先区分应用行为、裁判失效和 fixture 错误，再修复并重新运行，不降低阈值来隐藏失败。
