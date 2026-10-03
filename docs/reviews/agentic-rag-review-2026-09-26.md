# Agentic RAG 智能客服评审

评审日期：2026-09-26。代码基线：`f029be6`，当前本地工作区。评审范围：后端主链路、测试、评测、部署配置；没有访问线上客户数据，也没有验证线上实际部署状态。

**结论：作为架构展示项目，功能比较完整；作为承接真实客户与真实交易的客服系统，仍有明确的上线阻断项。下一步应优先完善现有链路的正确性、安全边界和可验证性。**

README 已明确定位为单实例技术 Demo，因此不能把未接真实订单系统视为违反当前项目定位；下面区分当前代码缺陷与商业化所需能力。P1 表示应优先修复，接真实数据前必须关闭；P2 表示质量与工程改进。这里的优先级不代表线上已发生事故。

## 已经做得比较好的部分

- 有真实的 LangGraph 条件路由、槽位收集、任务切换与恢复，不是单次向量检索包装。
- Agent 有步数上限和请求内 LLM 调用预算；退款、退货有显式确认机制。
- 检索、FAQ、重排、人工工单、知识缺口记录、用户长期信息等已有实现及接线。
- 有输入输出检查、政策数字校验、流式句子缓冲、模型故障恢复等有价值的设计。
- 有 Redis 限流、缓存分层、Prometheus 指标、告警、迁移和 CI；默认 Compose 数据服务绑定回环地址。
- 测试量较大：本次默认测试集合实际运行 **1551 项，全部通过，耗时 33.08 秒**。

这些是可以继续建设的基础。主要问题集中在模块之间：一个模块正确返回，不等于整个会话、流式传输、缓存与持久化始终满足相同约束。

## 验证结果与边界

| 检查 | 本次结果 | 解释 |
|---|---|---|
| 默认 pytest 集合 | 1551 passed | 使用 `--no-cov`，没有重新测量覆盖率 |
| Ruff lint | 通过 | `ruff check app/` |
| Ruff format | 通过 | 346 个文件格式符合要求 |
| Mypy strict | **失败：10 个文件、30 个错误** | 含测试注解及生产 factory 的 Redis 协议类型不匹配；不是 GitHub 远程 CI 状态查询 |
| 完整图的多轮模拟 | 发现旧工具结果污染新一轮 | 实际编译 LangGraph，仅外部服务使用模拟对象 |
| 完整图的安全拦截模拟 | 输入被拦截后工具仍执行 | 使用真实输入检查与默认 Demo 工具 |
| 输出、缓存、流式持久化、清空状态模拟 | 复现下文缺陷 | 全部在本地、使用虚构内容，无真实交易 |
| 真实模型回答质量、真实数据库集成、负载与恢复 | 本次未验证 | 不能据此给出线上准确率、并发容量或 SLA 结论 |

执行命令：

```bash
.venv/bin/python -m pytest -q --no-cov -o addopts=--import-mode=importlib
.venv/bin/python -m ruff check app/
.venv/bin/python -m ruff format --check app/
.venv/bin/python -m mypy app/
```

## 按优先级排序的具体发现

### 1. [P1] 身份与资源权限没有形成完整边界

> **修复更新（2026-09-27，d12a1cd，方案② DEMO_MODE 开关）**：新增 `DEMO_MODE` 设置（默认 `true`，公共演示行为完全不变——匿名访客按 body `user_id` 聊天、按裸 `session_id` 读写历史）。`DEMO_MODE=false` 时关闭本条主路径：身份只取认证上下文（匿名 chat/流式/历史 401；body `user_id` 与令牌矛盾 403）；历史读取/清空先经 `SessionService.get_session_by_id` 归属校验（他人会话与不存在会话统一 404，不泄露存在性）；知识库上传/删除要求管理员（`is_admin`）。实现位于 `app/api/deps/authorization.py`，调用点在端点 try 块之前，避免 401/403/404 被通用异常处理器吞成 500。**剩余**：feedback 接口的消息归属校验；检索 ACL 服务端注入；访客凭据方案（①，被本方案取代暂不做）；`_owned_order` 匿名放行在严格模式下经入口 401 已不可达。

**已确认。** [聊天入口](/Users/luopeng/Documents/GitHub/rag_chatbot/app/api/v1/chat.py:352) 在未登录时采用请求体中的 `user_id`，流式入口有相同逻辑。[历史读取](/Users/luopeng/Documents/GitHub/rag_chatbot/app/api/v1/chat.py:454) 和历史删除忽略 `current_user`，按外部传入的 `session_id` 操作。图的 `thread_id` 也只由该会话编号构成。

> **修复更新（2026-10-01，148507d + 9c1b2e5，#1 全部子项闭环）**：feedback 消息归属（148507d）：严格模式先经 message→session→user 归属探测，他人/不存在统一 404，评价与 downvote 缓存失效不再触碰他人消息；demo 姿态保持存在性检查。检索 ACL 服务端注入（9c1b2e5）：`VectorSearchRequest` 新增独立 `acl_filters` 通道，仅由服务端 `retrieval_acl_scope()` 从身份上下文构造（永不出自请求体/槽位抽取），白名单不适用于它；hybrid 在分发点把 scope 折入两腿的 filters（键冲突时 ACL 覆盖，伪造业务 filter 无法放宽）；filter-miss 回退丢业务 filter 但**保留 ACL 通道**（召回保护不成为越权通道）；L2 检索缓存 key 折入 scope；agent `search_knowledge_base` 工具同通道（无旁路）。当前 KB 全公开（无归属列），scope 为空约束——今日行为零变化，此为 P2 私有文档的接入缝（完整设计 docs/design/per-user-architecture.md；demo 单用户共享为最终形态、隐私不设防是用户 2026-10-01 决策，严格模式即每用户产品）。**#1 无剩余项。**

本地入口模拟确认：匿名请求的 `user_id=1` 被直接传给服务；匿名历史请求进入历史读取函数。默认订单工具的 [_owned_order](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/tools.py:180) 只在 caller_id 非零时检查归属，本地匿名查询 Demo 订单返回成功。

另一个边界是知识库：[上传](/Users/luopeng/Documents/GitHub/rag_chatbot/app/api/v1/documents.py:159) 与[删除](/Users/luopeng/Documents/GitHub/rag_chatbot/app/api/v1/documents.py:324) 仅要求普通登录，没有管理员角色或文档所有权检查；项目又开放普通用户注册。普通客户可修改共享客服知识库。[反馈接口](/Users/luopeng/Documents/GitHub/rag_chatbot/app/api/v1/feedback.py:28) 同样没有检查消息归属。

**影响：** 接真实数据后存在身份冒用、读取/污染他人会话、知识库投毒与删除、操纵反馈和缓存失效的路径。应用代码已确认存在该边界问题；线上是否另有网关限制未验证。

**建议：** 身份只取认证上下文；匿名访客使用服务端签发的访客会话凭据；聊天、历史、反馈统一校验会话所属用户。知识库写操作单独要求知识管理员权限。检索中的 ACL 由服务端注入，不能放进可取消的业务筛选条件中。这符合 [OWASP 对对象级授权的要求](https://api-security.owasp.org/editions/2023/en/0xa1-broken-object-level-authorization/)。

### 2. [P1] 输入检查标记 blocked 后，图仍继续执行工具

**修复更新（2026-09-27）：已修复。** 图入口的安全检查后新增阻断分支（`route_after_guardrail`）：被拦截的轮次直接结束，不再进入缓存、意图识别、工具执行或生成；拒绝文案由 guardrail 节点自行推送至流式队列。工具执行、Agent 循环、待确认动作的执行边界增加 blocked 二次校验（纵深防御）。新增 9 个回归用例（`test_blocked_turn.py`），验收即评审要求的标准：被拦截时工具调用次数为零。下文保留修复前的取证记录。

**完整图已复现。** [guardrail_node](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:174) 返回 `blocked=True`；但[图路由](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/graph.py:143) 无条件继续到缓存和意图识别。缓存节点遇到 blocked 只是 miss，并没有结束本轮。工具执行节点也未检查 blocked。

模拟输入包含已被正则识别的提示注入文本与查询订单意图，得到：

```text
input_blocked = true
graph_blocked = true
intent = query_order
tool_executions = 1
has_tool_result = true
```

**影响：** 最后向用户显示拒绝，不代表后台没有执行动作。即使当前工具只是 Demo，接真实业务工具前也必须修复。

**建议：** 安全检查之后增加明确的阻断分支，直接输出拒绝并结束；工具执行边界再做权限与本轮安全状态校验。验收必须断言被拦截时工具调用次数为零，而不只断言回复包含拒绝文本。

### 3. [P1] 跨轮状态残留，会让检索成功的新问题使用旧工具结果

**修复更新（2026-09-26）：已修复于当前工作区。** 图入口新增 `begin_turn`，在安全检查、缓存和路由之前重置 `TurnState` 中的临时字段；保留待确认动作、槽位、任务栈及任务已执行标记。每轮生成独立 `turn_id`，完成后的消息元数据记录该编号。上一轮工具信息单独保存在 `last_tool_execution`，仅用于转人工上下文，不参与本轮生成。新增 21 个完整图回归用例，覆盖普通/流式响应、政策事实检查、缓存、确认、任务栈与 checkpoint 恢复；下文保留修复前的取证记录。该更新仅关闭本条问题，不代表其他评审项已修复。

**完整图已复现。** [ChatService 输入](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/chat/chat_service.py:139) 每轮只提供消息、会话和用户；checkpoint 会保留其他字段。[handle_switch_node](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:379) 清理部分任务状态，却没有清理旧 `tool_result`。[生成分支](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:907) 优先使用工具结果，再考虑检索资料。

实际两轮模拟：第一轮「查询订单 ORD1001」产生工具结果；第二轮「退货政策是什么」被正确识别为 policy，也检索到 1 条政策资料，但生成提示仍包含上一轮订单工具结果。

**影响：** 意图识别和检索单测都可以通过，最终回答却可能答非所问。旧工具结果还会影响事实检查是否跳过。`blocked`、`sources`、`executed_tools` 等临时字段也应统一审计生命周期。

**建议：** 显式区分“会话持久状态”和“当前轮临时状态”，入口重置后者，并用 turn_id 关联工具结果、检索结果和输出。保留 pending_confirmation、任务栈等真正需要跨轮延续的状态。

### 4. [P1] 流式内容先发出再脱敏；direct 分支首次生成绕过输出检查

**修复更新（2026-09-27）：已修复。** 新增 `PIIStreamRedactor`（guardrails/stream_redactor.py）：句子缓冲 PII 脱敏，与声明流式门同构的 feed/flush 契约——PII 模式为无空白连续串，句子边界不会切断一个完整值，跨 chunk 的手机号/邮箱仍能整值捕获。`_call_llm` 在接入任意 guardrail 时，把全部流式发射（含声明门未武装的闲聊路径）先过 redactor 再入队，flush 与失败路径同样排空；guardrail 故障时原样放行（不阻断对话）。direct 层缓存未命中路径在答案离开节点前、写入 L1 前补上 `check_output`。验收即评审标准：流式拼接、图最终 response 一致且均已脱敏（新增 8 个用例：两种传输 × 两半泄漏 + redactor 单元契约 + 非流式 RAG 回归钉）。下文保留修复前的取证记录。

**已复现。** [_call_llm](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:1547) 将过滤思考内容、部分政策校验后的文本放入队列，但没有做输出 PII 检查。[generate_response_node](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:848) 在生成结束后才脱敏，此时已发出的内容无法撤回。

模拟输出被切成两个 chunk，结果如下：

```text
用户实际收到：联络邮箱：audit@example.invalid。
图最终 response：联络邮箱：[REDACTED]。
```

[direct_response_node](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:965) 在缓存未命中时直接生成并返回，既未经过统一输出检查，也未经过统一的完整响应事实检查。本地非流式 direct 调用同样返回未脱敏邮箱。

**建议：** 所有生成分支共用“缓冲 → 事实与隐私检查 → 发送 → 持久化”的出口；对跨 chunk 的手机号、邮箱等保留必要缓冲。验收要求流式拼接、最终状态与持久化正文一致，且均通过相同检查。

### 5. [P1] 匿名不等于无上下文，共享答案缓存可串用其他会话内容

**修复更新（2026-09-27）：已修复。** `_call_llm` 在生成提示词引入了会话历史或跨会话用户画像时，在每次请求的 config 上打 `generation_used_history` 标记；generate_response 节点将其发布为 `personalized` 轮次标记，ChatService 的 L0 写入门与 direct 层的 L1 写入门（`_maybe_put_semantic`）据此拒绝写入——写缓存的答案按构造即为无上下文版本，匿名会话的历史派生答案不再进入共享缓存。新增 7 个回归用例（`test_cache_eligibility.py`，覆盖非流式/流式 × L0/L1，并守卫「无历史仍正常写缓存」不过度收紧）。存量受污染条目随 KB epoch 轮换整体失效。下文保留修复前的取证记录。

**已复现。** [_call_llm](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:1517) 会引入当前会话历史，包含匿名会话。[L0 写入条件](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/chat/chat_service.py:366) 检查匿名、来源、待收集槽位等，却没有检查答案是否使用了历史。缓存键也不包含会话上下文。[L1 写入](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:977) 有类似问题。

本地模拟中，会话 A 的历史是「我使用 A 款设备」，随后问「可以吗？」；含该历史信息的答案成功写入共享缓存。会话 B 发送相同短问句，命中并收到「此前您说我使用 A 款设备……」。

**建议：** 共享缓存仅允许经确定性判定、完全不依赖会话历史的公开问答。使用了历史、长期记忆或任务栈的生成结果不写入全局 L0/L1；如确有需要，将这类缓存限定在同一用户/会话并加入上下文版本。读写两端都执行同一套资格判定。

### 6. [P1/P2] 中文关键词检索偏弱，多轮检索与候选召回仍有提升空间

**中文问题已复现。** [KeywordSearch](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/retrieval/hybrid_search.py:159) 使用 `\b\w+\b`，会把连续中文当成一个词；所谓 BM25 实际是集合交集比例，没有 IDF、词频或长度归一化。

```text
文档：商品支持七天无理由退货
查询：无理由退货
关键词分支结果：0 条
```

这不代表向量分支也一定失败，但关键词召回对典型中文客服问题帮助有限。其他代码层面的改进点：

- [RAG 检索](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:651) 以本轮消息和槽位扩展为主；历史直到生成时才加入。「那运费呢？」等追问需要先结合历史改写成独立检索问题。
- 同一路径 `top_k=3` 后才重排，重排只能重排已召回的三个候选。应区分较大的候选池和较小的最终上下文，用评测确定数量。
- 无检索结果时进入通用生成。政策/承诺类问题应有确定性的证据不足分支，澄清或转人工，并区分“没有相关知识”和“检索服务失败”。

**建议：** 优先建立中文分词/字符 n-gram 与真正 BM25 的对照基线，再评估更大候选池、多轮查询改写的增益；不要只凭模块名字判断混合检索已经有效。

> **修复更新（2026-09-27，commit c043c58，部分修复）**：关键词腿的两处硬缺陷已闭环。(1) 中文召回——`\b\w+\b` 把连续中文当单 token，改为 CJK 字符 bigram 分词（Elasticsearch cjk_bigram 同款基线，零新依赖），复现用例「商品支持七天无理由退货」× 查询「无理由退货」从 0 命中恢复为正确召回；ASCII 词保持整词。(2) 真 BM25——原「匹配词数/查询词数」集合交集比例替换为 Okapi BM25（Robertson IDF、饱和词频、长度归一化，k1=1.5 b=0.75），语料统计每次查询计算一次，IDF 在元数据过滤下仍按全语料计。融合层是 RRF（只看排名），分数尺度变化不影响 API。+8 测试（复现用例、分词契约、IDF/TF/长度归一化排序、ASCII 回归护栏），全套 1628 通过。**剩余子项未动**：多轮检索查询改写（历史到生成才加入）、top_k=3 候选池过小再重排、检索空结果时的确定性证据不足分支（澄清/转人工）、以及对照评测基线（bigram vs 分词、候选池大小增益的量化）。

> **修复更新（2026-09-27，commit 5da47ff，多轮查询改写已闭环）：** 追问（「那运费呢？」）现在先结合最近历史改写成独立检索问题再进入向量/BM25 腿（新模块 `dialogue/query_rewriter.py`，128 token 轻量调用、temperature 0；空/超长输出拒收；任何失败 fail-open 回原文检索）。`QUERY_REWRITE_ENABLED` 开关默认开；槽位抽取仍读原始消息。历史每轮只取一次：`_turn_history` 以 turn_id 记忆化，改写/agent/生成共享同一快照，消除了双取与跨读不一致。+5 测试（改写生效、无历史不调用、失败回退、空/超长拒收、开关关闭），全套 1639 通过，cov 87.28%。**剩余**：top_k=3 候选池扩大再重排、对照评测基线。

> **修复更新（2026-09-27，commit 9ada2c1，#6 第三子项）**：检索空结果的确定性分支已闭环。TurnState 新增 `retrieval_ran` / `retrieval_degraded`（begin_turn 重置——`retrieved_docs` 被预置为空列表，单靠空无法区分「检索过没找到」与「没检索」）；rag_lookup 上报两标志，`_search_once` 吞掉的 VectorClientError（双腿全挂）与外层异常都会标记 degraded；生成节点新增 Case 4b：检索跑过且结果为空 → 固定文案、零 LLM 调用——「暂未在知识库中找到」（知识缺失，建议换说法/转人工）与「检索服务暂时不可用」（服务故障，建议稍后再试/转人工）是两句话，因为正确的下一步不同。FAQ-miss 测试同步重锁：miss 仍到达 hybrid 检索腿，但空知识库以 no-evidence 分支收尾，不再无上下文自由生成。+6 测试（空检索/崩溃/双腿全挂/图意图空/闲聊不受影响/标志跨轮重置），全套 1634 通过。

> **修复更新（2026-10-01，对照评测基线闭环，#6 无剩余项）**：新增离线确定性检索评测基线（`app/services/evaluation/retrieval_eval.py` + 48 例双语金标集 `retrieval_golden_set.json` + `scripts/run_retrieval_eval.py`）：真实 BM25 关键词腿 + 真实 RRF 融合跑真实种子语料，词法替身充当向量腿——无模型下载、无网络、CI 可跑；替身与关键词腿同源相关，融合臂绝对值是模型值（真实嵌入会去相关两腿），模块 docstring 明示这一边界，换真向量客户端不动金标集与指标。基线量化了此前两个未量化的修复：① 分词对照（c043c58）——旧整词分词 zh recall@1/3/10 = 0.000，CJK bigram = 0.917/1.000/1.000；en 两臂逐位一致，A/B 只差分词一个维度；② 候选池宽（2436d2a）——融合臂 recall@3 0.917 → recall@10 0.979（+6.2pp，48 例中 4 例金标文档排在融合 4-10 名，池宽 10 是它们被重排器看到的唯一途径）。`--threshold` 门禁可选（fused recall@3），与 answer eval 退出码约定一致。+8 测试（金标集 schema/指标手算/确定性/结构/A-B 单维隔离/池宽方向性/旧分词 0 命中复现）。

### 7. [P1] 知识库 epoch 与关键词索引更新不是原子过程

> **修复更新（2026-09-27，commit 249dfd4，单实例已闭环）**：删除文档现在同步清理关键词索引——
> `delete_document` 在删除向量、切换 epoch 之后立即调用 `purge_keyword_document`（经
> `keyword_refresh` 的进程级注册表找到本进程 hybrid 实例，fail-open：清理失败仅告警，周期重建
> 仍是兜底）。两条检索腿在 delete 返回前都已干净，"删除后绝不返回该文档"有了跨存储回归用例
> （向量腿为空 + 关键词腿 purge 后检索为空）。旧索引结果写入新 epoch 缓存的窗口随之消失。
> 多副本场景与既有 refresher 同一 doctrine：各副本在下一个重建周期收敛；共享外部索引仍是
> 大语料时的长期方案。+7 测试，全套 1620 通过，覆盖率 87.1%。

**静态链路推导，未做真实 Redis/Qdrant 联调复现。** [文档删除](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/documents/ingestion.py:371) 删除向量数据后立即切换 epoch，但各进程的[关键词索引](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/retrieval/keyword_refresh.py:78) 默认每 300 秒才重建。

在此窗口里，旧关键词结果仍可能被检出，并写入新 epoch 的 L2 或 L0。关键词索引之后刷新时没有再切换 epoch，因此旧答案的影响可能持续到 L0 默认 24 小时 TTL，而非只持续 5 分钟。是否命中该路径取决于旧文档是否被关键词分支召回等条件。

**建议：** 发布一个完整可用的知识版本后再切换读取版本；或者给每条候选附带版本/删除标记，融合与缓存前过滤。所有实例确认新索引就绪前，避免把旧索引结果写进新版本缓存。对“删除后绝不返回该文档”增加跨存储回归用例。

### 8. [P1/P2] 流式审计不完整，清空历史不清空待确认动作

> **修复更新（2026-09-27，commit f3b1dec）**：两半均已修复。
> ① 流式审计：`process_message_stream` 的 finally 持久化与非流式路径对齐——补上 `intent`、`sources`、
> `executed_tools` 审计字段，并以 `status` 区分终态（client_disconnect / graph_error / budget_exhausted
> 的部分内容记为 FAILED，完整回合记 COMPLETED）。② 清空历史：`clear_chat_history` 现调用
> `adelete_thread` 删除对话检查点线程（fail-open，删除失败仅告警不报错）——staged 的
> `pending_confirmation` 随消息一起销毁，清空后一句裸"确认"不再执行已废弃的退款。
> 新增 6 测试（`test_turn_finalization.py`），全套 1610 通过，覆盖率 87%。

**已复现。** [流式持久化](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/chat/chat_service.py:341) 只传正文和 LLM 调用次数，没有像非流式路径那样保存 intent、sources、executed_tools。正常完成的模拟 refund 响应，其保存参数确实不含这些字段。异常退出写入部分内容时，也没有传入失败状态，沿用 persister 默认 COMPLETED。

此外，[clear_chat_history](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/chat/chat_service.py:409) 只清 memory_strategy 的消息。使用实际 MemorySaver 预置退款 pending_confirmation 后，调用清空历史，该待确认动作仍在 checkpoint 中。

**建议：** 流式与非流式共用 turn finalization，把实际执行记录、引用、终态与中断原因一起保存；业务写操作审计不能只依赖尽力而为的消息落库。明确清空历史与重置会话的产品语义，重置时同步清除 checkpoint 中的待确认动作和任务状态。

### 9. [上线前必需] 业务工具目前是 Demo，真实执行需要幂等与事务边界

> **修复更新（2026-09-27，commit 1525b69，部分修复）**：可当场落地的缺口已闭环——确定性确认分支
> （`_handle_meta_intent`）此前执行退款等不可逆工具却不写 `executed_tools`，审计与人工交接上下文
> 均为空白；现与 Agent 路径记录同一 tool/ok/args/summary 结构（失败尝试记 `ok=False`），并 pin 了
> 双击「确认」幂等（首次确认消费 staged gate，重复确认不重复执行）。其余项（不可伪造 action_id、
> 幂等键、写并发控制、真实业务系统事务边界、评测验收场景）属于接真实售后系统时的范围，demo 阶段
> 维持声明限制。+3 测试，全套 1613 通过。
>
> **修复更新（2026-10-03，commit 9c1e6d4，「有期限」半边闭环）**：本条点名的「不可伪造**且有期限**的
> action_id」中，期限此前完全缺失——周一 staged 的退款，周四一句裸「确认」照样执行。现在 staged 动作
> （确定性确认门 + Agent 路径）携带 `staged_at`（墙钟——checkpoint 跨进程重启持久）；确认时超过
> `CONFIRMATION_TTL_SECONDS`（默认 900s，0=关闭恢复旧行为）的 staged 动作直接丢弃不执行（验收=
> 工具调用次数为零），并按轮次语言提示重新发起（`CONFIRMATION_EXPIRED` 双语表）。迁移兼容：无
> `staged_at` 的存量 checkpoint 不受影响照常执行（测试钉住）。「不可伪造」半边在本架构下由构造保证
> ——staged 动作存于服务端 checkpoint 状态、从不经客户端往返，无伪造面。**#9 剩余**：接真实售后系统
> 时的幂等键/写并发控制/事务边界/对账——维持声明限制。+5 测试，全套 1809 通过。

这是当前项目声明过的范围限制。[默认退款、退货等 handler](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/tools.py:201) 返回模拟结果与随机单号；这能展示工具调用和确认交互，不能证明真实售后闭环已完成。

接业务系统时，显式确认只是第一步，还需服务端校验可退款金额/状态、不可伪造且有期限的 action_id、重复确认/超时重试的幂等键、同会话写操作并发控制、执行结果查询与对账。确认后实际执行的审计也要覆盖确定性分支：当前 [_handle_meta_intent](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/nodes.py:1410) 没有生成与 Agent 路径一致的 executed_tools 记录。

建议先接通一个低风险、可核验的真实只读工具，再逐个接入有副作用操作。上线验收加入双击确认、断线重试、执行成功但本地落库失败等场景。

### 10. [P2] 测试数量可观，但当前覆盖率和回答评测不能代表完整产品质量

- [覆盖率配置](/Users/luopeng/Documents/GitHub/rag_chatbot/pyproject.toml:278) 排除了主路径 graph.py、nodes.py、tools.py；默认 coverage 的模块列表也不包含完整 chat/agent/API。因此 README 的 core coverage 不能解释成整个系统覆盖率。
- [集成聊天测试](/Users/luopeng/Documents/GitHub/rag_chatbot/app/tests/integration/test_chat_pipeline.py:1) 使用 scripted graph，主要验证适配层，不能覆盖真实节点之间的交互。
- [一处状态回归测试](/Users/luopeng/Documents/GitHub/rag_chatbot/app/tests/unit/services/dialogue/test_dialogue_flow.py:34) 名为状态重置测试，却只断言手工字典已有的内容，没有运行被测实现。
- 回答 golden set 有 20 例，意图集 45 例、claim 集 18 例。[真实模型评测脚本](/Users/luopeng/Documents/GitHub/rag_chatbot/scripts/run_answer_eval.py:73) 直接按 faq_ids 注入正确政策，再生成答案，绕过了真实意图路由、FAQ 匹配、检索、缓存和工具。这适合测“给对资料后能否回答”，不能测完整客服解决能力。
- 本地 `mypy app/` 返回 30 个错误，而 CI 正在执行同一条检查命令。应先恢复严格检查可重复通过，并锁定依赖版本；仓库目前没有依赖锁文件，也没有发现 CI 依赖漏洞审计步骤。本次没有进行 CVE 扫描。

> **修复更新（2026-09-27，commit 5c5ffc4，部分修复）**：mypy 子项已闭环——本地 `mypy app/` 从 55 个错误恢复到 0（CI 执行同一命令），其中 53 个为测试文件类型标注（yaml.safe_load 返回 Any、Pregel.ainvoke 需要 RunnableConfig 配置字典、测试桩与正式服务接口的 cast 边界等），2 个为 factory.py 两个缓存装配点把 redis-py 客户端直接传给窄 RedisSeam 协议；`app.tests.*` 的 mypy override 补充放宽 `disallow_incomplete_defs`（测试方法带 fixture 参数的部分标注不携带信息）。改动仅限类型标注与 cast，行为不变（全套 1620 通过，覆盖率 87.10%）。剩余子项未动：graph.py/nodes.py/tools.py 的覆盖率排除、集成测试真实性、评测脚本绕过真实管线、依赖锁文件与 CVE 扫描。

> **修复更新（2026-10-01，覆盖率排除子项闭环；检索评测子项见 #6 基线）**：主路径三件套（graph.py/nodes.py/tools.py）+ slot_filling/factory.py 移出 coverage omit——先测量后摘除：un-hide 后实测 graph 100% / nodes 93.64% / tools 96.75%，排除项藏的是健康数字而非风险；slot_filling/factory 实测 0%（chat 端点懒初始化路径，套件从未触达），补 6 个 factory 契约测试（rule_based/hybrid 双腿/hybrid 无 LLM 降级/未知类型/开关开关两态）至 100%。新增元测试 `app/tests/test_coverage_honesty.py`：maintained core 重新进入 omit 列表即刻失败（5 用例，parametrize 逐文件 + dialogue/slot_filling 前缀整体扫描），防回退。GraphRAG 实验路径维持排除（默认关闭，非维护核心，注释说明）。README 数字同步：1785 tests / 89.25%（主路径计入后 gate 覆盖率反升 87.61→89.25%——被隐藏文件覆盖率高）。剩余 #10 子项：集成测试真实性、依赖锁文件与 CVE 扫描（评测真实性=检索半边已闭环，见 #6）。

> **修复更新（2026-10-02，b79797e，依赖锁 + CVE 扫描子项闭环）**：新增运行时依赖锁 `requirements-lock.txt`——以 python:3.11-slim（Docker/CI 同解释器；3.12 语义解析可能选出排除 3.11 的版本，constraints 会直接弄死 3.11 构建）对 pyproject base 依赖做全新解析（`pip --dry-run --report`，不触碰本地 venv），100 个精确 pin；Dockerfile 与 CI 三处安装均以 `-c requirements-lock.txt` 约束，CI 新增 audit job（pip-audit 对锁查 OSV/PyPI 通告库）。`scripts/generate_lock.py` 一键重生成，锁契约由 `app/tests/unit/test_dependency_lock.py` 钉死（全 pin、base 依赖全覆盖、pin 满足 pyproject specifier、dev 工具不得泄漏，5 用例）。首次审计即抓到真雷：**PYSEC-2026-1325（CVE-2024-23342，Minerva 时序攻击）命中 ecdsa 0.19.2，上游无修复版**——按业界标准缓解从依赖树根除：python-jose（拖入 ecdsa/rsa/pyasn1）整体迁移到已是 base 依赖的 PyJWT（app/core/security.py：encode/decode 异常面 1:1，exp 显式 int 上线，token 线格式不变）；静态嵌入 jose 铸的兼容 token 测试证明**换库后既有线上 token 仍可验**，alg=none 拒绝一并钉住；types-python-jose 随之删除。迁移后锁 107→100 包，pip-audit **0 已知漏洞**，builder 镜像实测无 ecdsa/rsa/pyasn1/python-jose。顺手修 test_candidate_pool.py 既存 strict-mypy 违例（sources 可空类型裸操作，新缓存下显形）。suite 1794 / cov 89.25% 不变。**#10 仅剩：集成测试真实性。**
>
> **修复更新（2026-10-03，efc5c45，集成测试真实性闭环，#10 全部子项完成）**：本条开头点名的假测试（test_dialogue_flow.py「状态重置」用例——手工构造字典再断言自己写入的字面量，从未运行被测实现）已原位替换为真编译图跨轮测试：同一 checkpointed 线程先「查询订单 ORD1001」工具轮、再「退货政策是什么」检索轮，断言第二轮生成 prompt 锚定「参考资料：」而非残留「工具执行结果：」，且 `executed_tools`/`tool_result` 复位——即 #3 begin_turn 契约的图级回归钉。另新增组合层集成 `app/tests/integration/test_chat_pipeline_real_graph.py`（10 用例）：ChatService × 真图 × 真 MemorySaver，真规则意图/默认工具注册表/默认输入护栏，仅桩网络缝（录制式 LLM + 单命中 hybrid）；场景覆盖跨轮任务切换（response/stream 双传输参数化）、跨会话隔离（会话 2 不得见到会话 1 的 ORD1001/工具结果）、确认门全生命周期（stage→确认恰执行一次→重复确认不重执行，组合层复钉 1525b69 幂等）、阻断轮后同线程恢复（评审 #2 验收 + 未问的恢复问题）。scripted-graph 适配层契约仍由 test_chat_pipeline.py 保留——两文件互补，互不替代。套件 1794→1804，cov 89.25% 不变，ruff + strict mypy 0 错。**#10 无剩余项（mypy / 覆盖率排除 / 依赖锁+CVE / 集成真实性四个子项全部闭环）。**

**建议：** 保留现有单元测试；补充直接跑编译后图的跨轮集成测试。回答评测调用实际 ChatService/HTTP 接口，覆盖普通问答、追问、多意图、无答案、过期政策、注入、缓存命中、退款确认和人工交接。指标分开测检索相关性、回答准确性、依据一致性、任务成功率、P95 延迟与单会话成本；[官方 RAG 评测文档](https://docs.langchain.com/langsmith/evaluate-rag-tutorial) 也将检索和回答评估分开。

### 11. [P2] “一次解决率”当前更接近自动留存代理指标

[get_one_shot_stats](/Users/luopeng/Documents/GitHub/rag_chatbot/app/repositories/ticket_repository.py:228) 把有机器人回复、没有人工工单且没有差评的会话算作成功。用户没解决就离开、机器人反复说抱歉、没有人评分，都可能被算作成功；逻辑也没有要求一次交互解决。

**建议：** 保留该指标但准确命名，另行记录明确 resolved 状态、业务动作成功凭证或用户确认、同一问题短期重开/再次求助；结合人工抽检估计解决率。不要用没有转人工来替代已经解决。

> **修复更新（2026-09-27，commit b3341f0）：** 命名已诚实化。该指标实际度量 = 有机器人回复 + 无人工工单 + 无差评 = 行业标准叫 **containment rate（自动处理率）**，不是"一次解决率"。全量重命名（无别名，演示系统无外部消费者）：`get_one_shot_stats`→`get_containment_stats`、字段 `one_shot_rate`→`containment_rate`、API `/sessions/one-shot-stats`→`/sessions/containment-stats`、Prometheus gauge `chat_one_shot_rate`→`chat_containment_rate`、histogram/refresher 函数同步、告警 `OneShotRateLow`→`ContainmentRateLow`、Grafana 面板/图例改名、settings `ONE_SHOT_STATS_*`→`CONTAINMENT_STATS_*`、README 口径改为「未升级≠已解决」。repo/端点/模块 docstring 均明示 contained ≠ resolved。全套 1634 通过，cov 87.23%。剩余未做（需业务语义设计，非纯代码）：明确 resolved 状态字段、业务动作成功凭证、用户确认信号、同一问题短期重开检测、人工抽检流程。

> **修复更新（2026-10-03，session resolution 状态闭环）：** 上条剩余中的三个数据半边已落地——**resolved 状态字段**（`chat_sessions.resolved_at` + `reopened_count`，迁移 c5f2a8e3d17b）、**用户确认信号**（`app/services/chat/resolution.py`：确定性「解决了/搞定了/solved/all set」短语判定，精度优先——裸「谢谢」是礼貌不是解决，不触发；「还没解决」被负向后行排除）、**同一问题短期重开检测**（已解决会话的任意非确认后续消息 → 重开：`resolved_at` 置空 + `reopened_count`+1；确认分支先判，重复「解决了」只刷新时间戳永不误重开）。挂载点在 `ChatMessagePersister.persist_turn` 的事务内、fail-open 与持久化其余部分同 doctrine。`get_containment_stats` 增 `sessions_resolved`/`sessions_reopened` 交叉表（与服务过会话交叉，独立于失败信号——resolved+escalated 双计，供人工抽检定位）。**#11 剩余**：业务动作成功凭证（接真实售后系统时）与人工抽检流程（运营侧），均非纯代码项。+28 tests，全套 1837 通过，cov 89.26%。

> **修复更新（2026-10-03，人工抽检队列闭环）：** 「人工抽检估计解决率」半边的可代码化核心已落地——`GET /api/v1/sessions/review-queue` 采样 **containment 分子中无正面证据的人群**（窗口内服务过 ∧ 无工单 ∧ 无差评 ∧ `resolved_at` IS NULL ∧ 未抽检过）：正是评审点名的「用户没解决就离开也可能被算作成功」的膨胀风险人群；随机抽样（非普查，limit×3 候选封顶）+ 末条用户消息 200 字预览供分诊。`POST /api/v1/sessions/{id}/review` 记录质检判定：任何判定使会话离开队列（`session_metadata.qa_verdict`，覆盖式单判定非审计日志）；「已解决」判定同时落 `resolved_at`——与用户确认路径同语义、汇入 `sessions_resolved` 交叉表，「抽检确认解决率」由此可算。严格模式下两端点 admin-only（质检读他人会话+影响 KPI，属运营面，同 KB 写 doctrine）。**#11 剩余**：仅业务动作成功凭证（接真实售后系统时）。+18 tests，全套 1855 通过，cov 89.26%。

### 12. [P1/P2] 启动降级与就绪检查可能报告虚假的可服务状态

> **修复更新（2026-09-27，commit 097e69e）：** 四条静默降级路径全部闭环。(1) API 装配层拒绝 graph=None 的服务（initialize_chat_service 抛 RuntimeError → lifespan 保持 not_ready）；(2) 两种传输先行快速失败：POST /chat 映射为 503（GraphUnavailableError），/stream 按既有约定输出 STREAM_ERROR 帧，不再以 AttributeError 500 暴露；(3) checkpointer 新增 degraded 标志，Postgres 模式被替换为 MemorySaver 时 lifespan 扣留就绪，编排器看到的是 not-ready 副本而不是悄悄遗忘会话状态的副本；(4) backup.sh 的 pg_dump 失败分支改为 exit 1（并跳过保留期清理，保住最近一次好备份），退出码可被监控捕获。+8 测试（app/tests/unit/test_readiness_honesty.py），全套 1604 通过。

[工厂](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/chat/factory.py:344) 构图异常后仍返回 graph=None 的 ChatService；[lifespan](/Users/luopeng/Documents/GitHub/rag_chatbot/app/main.py:49) 随后可将 chat_ready 标记为 True，但 process_message 无条件调用 graph.ainvoke。与此同时，[Postgres checkpointer](/Users/luopeng/Documents/GitHub/rag_chatbot/app/services/dialogue/checkpointer.py:55) 连接失败会降级到进程内 MemorySaver，即使是多副本部署。

**影响：** 服务可能 /ready 成功却无法聊天，或请求换副本后丢失待确认状态。缓存、重排可以按业务需要降级；核心图、身份和交易状态不能用同样的降级策略。

**建议：** 生产模式启动时验证核心图和必需依赖；必需依赖失效则拒绝就绪。区分存活、就绪与可选能力降级，监控降级原因。对数据库、Qdrant/Neo4j 数据和知识来源分别定义备份/重建/恢复流程。已有 PostgreSQL 备份脚本，但本次未验证恢复演练；其 pg_dump 失败分支还会继续执行并输出成功提示，应保证失败退出码可被监控捕获。

## 十二维检查面板

以下是“承接真实业务”的检查状态，不是代码完成百分比。存在组件并不代表该维度整体通过；PASS 也仅限本次检查范围。

```text
╔══════════════════════════════════════════════════════════════╗
║              PRODUCTION READINESS REPORT                     ║
║              Project: rag_chatbot                            ║
║              Type: Python/FastAPI | Date: 2026-09-26          ║
╠══════════════════════════════════════════════════════════════╣
║  Overall: 2/12 dimensions ready  ██░░░░░░░░░░                 ║
╠══════════════════════════════════════════════════════════════╣
║  #  Dimension           Status      Details                  ║
║  1  Code Quality        ❌ FAIL     Mypy: 30 errors           ║
║  2  Test Coverage       ⚠️ WARN     Main graph excluded       ║
║  3  Security            ❌ FAIL     Auth and guard bypasses   ║
║  4  Documentation       ✅ PASS     Demo scope documented     ║
║  5  Dependency Health   ❌ FAIL     No dependency lock        ║
║  6  CI/CD Pipeline      ❌ FAIL     Strict check fails locally║
║  7  Observability       ⚠️ WARN     Audit/KPI gaps            ║
║  8  Error Handling      ❌ FAIL     Invalid graph fallback    ║
║  9  Configuration       ✅ PASS     Typed settings/examples   ║
║  10 Performance         ⚠️ WARN     Cache safety; no load run ║
║  11 API Contracts       ❌ FAIL     Stream/REST behavior gap  ║
║  12 Database            ⚠️ WARN     Persistence/reset gaps    ║
╚══════════════════════════════════════════════════════════════╝
```

```text
┌─────────────────────────────────────────────────────────────┐
│  BLOCKERS (6 failing dimensions, 4 warnings)                 │
├─────────────────────────────────────────────────────────────┤
│  FAIL Security/API: close authorization and output gaps      │
│  FAIL Errors: reject readiness for missing core graph        │
│  FAIL Code/CI: repair strict types and verify current gate   │
│  FAIL Dependencies: lock runtime and add dependency audit    │
│  WARN Tests: exercise actual graph and full answer pipeline  │
│  WARN Observability: persist audits and define resolution    │
│  WARN Performance: fix contextual/stale cache admission      │
│  WARN Database: reset checkpoints and validate restoration   │
└─────────────────────────────────────────────────────────────┘
```

## 建议实施顺序与验收

| 顺序 | 改进包 | 验收标准 |
|---|---|---|
| 第一批 | 身份归属、知识库写权限、安全阻断、输出出口、临时状态、共享缓存资格 | 无法越权；blocked 时零工具调用；订单→政策正确切换；SSE/REST/落库同样安全；跨会话不串答案 |
| 第二批 | 中文检索、多轮改写、候选重排、知识版本发布、无证据分支 | 完整问答集测出召回与准确率；删除文档不再出现在任何检索/缓存答案中；证据不足能澄清或转人工 |
| 第三批 | 真实业务适配、幂等与审计、会话重置、工单衔接 | 重复请求不重复执行；失败可对账；重置无遗留动作；工单与实际处理结果可追踪 |
| 持续门禁 | 修复 Mypy、扩大真实图测试与回答评测、锁定依赖、就绪/恢复/负载验证 | 每次发布都有可重跑的质量报告；真实模型质量和容量指标有实测依据 |

第一批完成前，不建议接真实客户隐私数据或不可逆业务动作。作为作品展示，可以继续保留现有能力，同时准确展示 Demo 限制。本次仅新增评审报告，未修改业务实现；原有未跟踪的 docker-compose.local.yml 未改动。
