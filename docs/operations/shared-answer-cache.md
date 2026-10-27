# 共享答案缓存的上下文边界

L0 精确答案与 L1 语义答案均跨会话复用正文。相同问题不代表相同适用条件：用户事实、历史、摘要或未完成任务都可能改变应答。这两层只在匿名、无任务状态且没有可用会话历史的条件下重放和写入。

## 读取和写入资格

以下任一条件存在时绕过共享答案，继续正常管线：非零用户身份、已填/待填槽位、待确认操作、挂起任务、已执行任务标记、待重试退款审核，或当前轮已被判定不可共享。配置历史 provider 的图还必须成功读到空上下文；有原始历史、有效摘要或读取失败均不能重放。没有配置历史 provider 表示该图未启用历史功能，其匿名无状态请求仍可命中。

`shared_cache_blocked` 是当前轮的粘性否决标记。第一次历史读取失败后，即使后续生成重试恢复为空，也不允许本轮写回通用答案。新一轮会重置该标记并重新检查资格；标记为 false 本身不证明会话为空。回答生成保留既有的尽力读取历史行为，存储不可用时仍可能缺少个性化信息；本修复阻止共享缓存把未知上下文当成可复用许可，不保证故障期间生成必然准确。

缓存命中仍执行原有事实检查和输出安全检查。KB epoch、模型与 persona 标签、L1 意图白名单、负反馈逐出和来源版本快照保持原契约。没有上下文不等于任意匿名首轮输入或生成正文都已通过完整隐私审计，本轮不作此声明。

## 历史读取与摘要

`ChatMessagePersister.get_history` 的默认调用仍在数据库故障时返回空列表；工厂绑定的上下文 provider 使用 `include_summary=True, raise_on_error=True`，让缓存资格能区分“确实空”和“不可读取”。生成器自行处理严格读取的异常，缓存故障或历史故障不会使缓存成为服务可用性的前置条件。

只有非空有效摘要才截去其覆盖的原始消息。空或空白摘要不能隐藏仍可读的原始历史；若仅有无效摘要且没有可读原始上下文，严格读取抛出未知状态，默认读取行为仍可返回空列表。后台摘要生成器本轮未改。

历史 memo 仅在非空 turn ID 下复用，键同时包含会话与轮次；没有 turn ID 时重新读取。它只保留一条快照，并发轮次可能互相淘汰而重读，不能据此宣称并发下固定快照或最多读取一次。顺序回归中的缓存资格与生成复用了同一轮历史。资格检查本身不调用模型；匿名热命中仍为零生成，有上下文时按正常管线重新生成。

## 验证

```bash
.venv/bin/pytest --no-cov -q \
  app/tests/evaluation/test_contextual_answer_cache.py \
  app/tests/unit/services/chat/test_answer_cache.py \
  app/tests/unit/services/chat/test_semantic_cache_node.py \
  app/tests/unit/services/chat/test_persistence.py \
  app/tests/unit/services/chat/test_chat_factory.py \
  app/tests/unit/services/dialogue/test_answer_cache_node.py \
  app/tests/unit/services/dialogue/test_llm_history.py \
  app/tests/unit/services/dialogue/test_turn_lifecycle.py
```

实际 L0 图用例覆盖 normal/stream、匿名热命中、用户事实、原始历史、摘要、provider 异常与真实 SQLite 读取失败，并独立检查 Redis GET/SET 次数、生成原始提示、持久正文及引用。L1 另有真实 SemanticCacheService、确定性 embedding 和本地 Redis 命令替身的 14 组独立对照；节点回归固定其读取与写入边界。

结果见[上下文缓存证据](../reviews/context-cache-results-2026-10-28.json)。验证未连接外部模型、Redis、向量服务或生产数据库，不代表真实模型质量、线上延迟、容量、浏览器、分布式会话串行化或完整匿名隐私验收。
