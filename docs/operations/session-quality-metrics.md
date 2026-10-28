# 会话质量统计：失败、完成与解决证据

`GET /api/v1/sessions/containment-stats?window_days=7` 返回窗口内的会话统计，沿用既有的活跃账号认证与全局聚合口径。`window_days` 范围为 1–90，默认 7。统计依据已持久化消息，不证明客户端收到完整回复。

## 指标含义

| API 字段 | 含义 | Prometheus Gauge |
| --- | --- | --- |
| `sessions_served` | 窗口内至少有一条 assistant 消息的会话数，不限处理状态 | `chat_sessions_served` |
| `sessions_with_failed_replies` | 窗口内至少有一条 assistant 状态为 FAILED 的会话数 | `chat_sessions_with_failed_replies` |
| `sessions_with_completed_replies` | 窗口内至少有一条 assistant 状态为 COMPLETED 的会话数 | `chat_sessions_with_completed_replies` |
| `sessions_latest_reply_failed` | 窗口内最新 assistant 消息状态为 FAILED 的会话数 | `chat_sessions_latest_reply_failed` |
| `containment_rate` | served 会话中，无任何工单且无窗口内差评的占比；无 served 时为 null | `chat_containment_rate` |
| `sessions_resolved` | served 会话中当前有明确用户或质检解决证据的会话数 | 既有 API 字段，本轮未新增 Gauge |
| `sessions_reopened` | served 会话中当前重开计数大于零的会话数 | 既有 API 字段，本轮未新增 Gauge |

失败与完成是可以重叠的集合，不能堆叠成互斥人群。最新消息按 `created_at DESC, id DESC` 排序，仅考虑同窗口内的 assistant；后续 user/system 消息不会清除“最新 assistant 失败”。PENDING、PROCESSING、遗留未知或 NULL 状态都不当作完成。FAILED 本身不能区分断连、图错误或超时，也不能覆盖从未产生持久消息的请求；这类请求还需结合流式 outcome 计数和日志。

处理状态完成不等于问题解决，后续完成也不抹去窗口内先前失败；解决时间戳同样不会删除失败证据。本轮不修改既有 containment 公式：工单信号仍是 served 人群中任意时间/状态工单，差评按窗口内 assistant 计，解决/重开是当前会话状态，均不是本轮新增的窗口事件计数。

## 如何读数

假设单个会话没有工单、差评或明确解决证据：

| 窗口内 assistant 状态顺序 | 有失败 | 有完成 | 最新失败 | containment | 明确解决 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAILED | 1 | 0 | 1 | 100% | 0 |
| FAILED → COMPLETED | 1 | 1 | 0 | 100% | 0 |
| COMPLETED → FAILED | 1 | 1 | 1 | 100% | 0 |
| FAILED → PROCESSING | 1 | 0 | 0 | 100% | 0 |

因此 100% 自动处理代理率可能同时伴随失败回复，不能读成全部请求成功解决。`sessions_latest_reply_failed=0` 也不能单独证明已恢复或已解决。

## 刷新与仪表盘

DB 桥接按部署配置 `CONTAINMENT_STATS_WINDOW_DAYS` 计算窗口，默认 7 天，按 `CONTAINMENT_STATS_REFRESH_SECONDS` 周期刷新。Grafana 时间选择器显示这些滚动统计的历史样本，不会把 DB 窗口自动改成所选的小时数；API 参数与部署窗口也可以不同。

每个副本导出 `chat_containment_stats_last_success_timestamp_seconds`。只有完整统计读取成功才更新它；DB 故障或缺合同字段时全部 Gauge 保留旧值，时间戳不前进。成功读到空窗口时，绝对计数归零，API 比例为 null；原比例 Gauge 保留旧值，仪表盘通过 `served > 0` 过滤它。

仪表盘比例和会话计数分开，使用不同单位。首次成功刷新前的副本不参与比例/计数展示，避免把初始化零值当作已知零失败；成功后的真实零计数仍可显示。跨副本维持平均聚合，不求和重复的 DB 统计。刷新年龄来自最近成功副本的最大时间戳，不能证明所有副本都新鲜；平均计数也不保证是同一时点的原子快照。

三个新增人群在同一个 SQL 聚合中计算。完整 API 报告仍含多个 SQL 语句，不宣称并发写入下所有字段来自固定快照。第十四轮只补统计与仪表盘。当前 containment 告警另按目标 job、逐副本新鲜度、scrape 健康及有效数值筛选，并独立报告统计不可用和时钟偏差，见[会话统计告警](containment-alerts.md)。仪表盘仍使用原初始化/流量过滤，没有自动采用这些告警条件；图上均值不能直接替代告警值。其他队列告警没有因此获得新的有效性门控。

## 验证

```bash
.venv/bin/pytest --no-cov -q \
  app/tests/evaluation/test_failed_reply_visibility.py \
  app/tests/unit/repositories/test_containment_stats.py \
  app/tests/unit/services/handoff/test_containment_metrics.py \
  app/tests/unit/deploy/test_grafana_dashboards.py
```

真实 SQLite、JWT HTTP 与 Prometheus 文本导出验证统计一致、排序、时间边界、未知状态及 DB 故障保留旧值。独立 SQL 探针检查固定查询数量和 PostgreSQL 编译；本轮未执行 PostgreSQL、PromQL 引擎或 Grafana 浏览器。前后结果见[失败会话可见性证据](../reviews/failed-reply-visibility-results-2026-10-28.json)。

时间戳和向量过滤设计分别参考 [Prometheus instrumentation](https://prometheus.io/docs/practices/instrumentation/) 与 [PromQL operators](https://prometheus.io/docs/prometheus/latest/querying/operators/)；这些资料不代替本项目的运行验证。

## 人工通道的客户解决证据

私人会话中新提交的 customer 人工消息也参与现有 resolved_at/reopened_count 状态机，与 AI 客户正文使用相同保守识别规则。staff 回复或关闭工单不作为客户解决；否定、疑问、引用、附带新求助不确认解决。已解决会话收到其他非空客户消息时按现有会话粒度重开一次，不宣称能区分是否同一问题。正文与状态同事务，精确 UUID 重放不刷新解决时间，也不重复重开或覆盖后来状态。

这些证据可更新已有 served 会话的 sessions_resolved/reopened；人工消息不增加机器人分母或失败/完成回复群体。只有人工消息的会话不进入这些机器人 served 指标。字段、刷新和原 containment 公式均不变。具体回归见[人工后续消息记录](../reviews/human-followup-results-2026-10-28.json)。
