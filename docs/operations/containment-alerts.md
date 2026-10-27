# Containment 告警与统计可用性

本页对应 [`alerts.yml`](../../deploy/prometheus/alerts.yml) 中的三个会话统计告警。它们分别报告代理指标偏低、统计证据不可用、统计时钟超出容差；都不能单独证明客户问题未解决或聊天服务发生故障。

`containment_rate` 的分母仍是窗口内至少有一条 assistant 消息的会话，包括失败回复；分子是其中没有任何转人工工单、且没有窗口内差评的会话。它不是 resolution。FAILED/COMPLETED、用户或质检解决证据及重开情况需要分别查看，定义见[会话质量统计](session-quality-metrics.md)。本轮不修改这些统计公式或其他 handoff Gauge。

## 采样与有效性合同

当前 [`prometheus.yml`](../../deploy/prometheus/prometheus.yml) 每 15 秒抓取一次 `rag-chatbot-api` job 的 `/metrics`，通过 `api` 的 A 记录发现副本，每 30 秒更新 DNS 发现。三个告警只选择该 job，不将其他 job 的同名指标混入计算。

应用默认每 300 秒刷新一次统计，后台循环先等待再首次查询。只有完整统计读取成功才更新 `chat_containment_stats_last_success_timestamp_seconds`；失败时旧 Gauge 和时间戳都保留。成功读取空窗口会更新成功时间戳、将计数归零，同时保留原比例 Gauge，因为空窗口的比例未定义。

对每个当前发现的 `(job, instance)`，有效快照必须同时满足：

- `up == 1`，本次抓取成功。
- `chat_sessions_served >= 0`。
- `0 <= chat_containment_rate <= 1`。
- 成功时间戳存在、严格大于零，且不超过 Prometheus 当前时间加 60 秒。
- Prometheus 当前时间减成功时间戳不超过 900 秒。

缺指标、尚未成功刷新、陈旧、越界数值、过大的未来时间戳或抓取失败都不能作为有效统计。有效的零流量快照是可用统计，不是低率，也不是不可用。

这些条件在副本级匹配后才聚合。副本共享同一 DB 统计，不能将其求和重复计算；同一 job 内仍要求使用一致的数据源与统计窗口配置。副本独立刷新、Gauge 保留旧值及多个 SQL 读取都意味着这不是整体原子数据库快照。

## 三种告警

| 告警 | 条件与等待 | 含义 |
| --- | --- | --- |
| `ContainmentRateLow` | 只纳入有效且 `served > 0` 的副本，按 `job` 求平均；低于 0.7 持续 30 分钟 | 当前可用快照的自动处理代理率偏低，需要检查人工需求、差评和实际解决证据 |
| `ContainmentStatsUnavailable` | 当前发现的目标中至少一个不满足有效性条件，或整个 job 的 `up` 不存在，持续 10 分钟 | 统计证据缺失或部分副本不可信，不能据此判断业务率高低 |
| `ContainmentStatsClockSkew` | 成功抓取的副本中，成功时间戳领先 Prometheus 超过 60 秒；按 `job` 取最大领先值，立即触发，并从表达式首次不再匹配的求值时刻起保持 10 分钟 | 先检查时钟、时间戳单位与数据来源；这不是客户故障或低率判断 |

`ContainmentRateLow` 的 70% 是初始运营阈值，未经生产基线校准，也不是已验证的 SLO。目前只要求窗口内有流量，没有最小样本量门槛；一个会话也可能产生持续低率。30 分钟描述的是规则条件连续成立的时间，不是只计算最近 30 分钟的会话。DB 滚动窗口默认仍为 7 天，由 `CONTAINMENT_STATS_WINDOW_DAYS` 控制。

不可用告警在 job 级判断“始终至少有一个坏目标”。即使坏副本在这 10 分钟内轮换，只要每次求值都有坏目标，计时也能继续；它不是要求同一个副本坏满 10 分钟。正常副本不会掩盖坏副本。整个 job 无 `up` 时的数值是缺失信号，不能当作真实坏副本数量。

三个告警可以同时出现。例如，一个副本未来时间戳异常，另一个有效副本的代理率持续偏低，统计与业务信号应分别处理。不可用副本不再参与低率计算；低率条件因数据消失而清除，不代表产品质量改善。

## 时间容差与告警延迟

首次始终没有成功刷新时，不可用条件从观察到初始化零值或其他无效状态开始计时，持续 10 分钟后告警。默认首次刷新通常在 5 分钟左右开始，因此正常初始化有等待空间；查询耗时、发现和抓取也会影响实际时刻。

对已经成功刷新过、随后停止更新的副本，旧快照超过 15 分钟后不再有效，再持续 10 分钟才触发不可用告警，即通常在上次成功快照约 25 分钟后告警。抓取直接失败或指标缺失不必等满快照年龄；实际触发时刻还取决于 Prometheus 的求值与序列存在状态。

60 秒未来容差用于避免小幅时钟偏差反复清掉 30 分钟的低率 pending 状态。例如应用时钟领先 5 秒，每次刷新后的某次求值都可能看到“未来”时间戳；完全不容忍会使低率计时每 5 分钟重置，而短暂不可用又达不到 10 分钟。超过容差的偏移由独立时钟告警立即捕获，`keep_firing_for: 10m` 保留短暂但反复出现的信号。保持时间从首次未匹配的规则求值起计，不是从最后一次匹配起计；例如 0 分钟匹配、1 分钟首次未匹配时，持续未匹配到 11 分钟才解除。保持期间仍 firing 不代表当下偏移一定仍大于 60 秒。

300 秒刷新周期、900 秒新鲜度、10 分钟不可用等待、60 秒未来容差和时钟告警保持时间需要一起维护。调整 `CONTAINMENT_STATS_REFRESH_SECONDS` 时必须人工核对规则与验证场景，它们不会自动读取应用配置。刷新配置为 0 会禁用统计任务；这仍表示 KPI 不可用，应按监控目标处理，不应将告警解释成 DB 已故障或客户服务已中断。

## 诊断步骤

先在 Prometheus 的 Alerts/Rules 页面确认告警名、job、pending/firing 状态和规则实际已加载，再按以下顺序检查。只查看指定指标与必要配置，避免导出整个运行环境。

1. **检查发现与抓取。** 在 Targets 页面确认 `rag-chatbot-api` 是否存在，以及各 instance 的抓取错误。查询 `up{job="rag-chatbot-api"}`。有 `up=0` 与整个 job 没有 `up` 是不同线索；后者要核对 job 名、DNS 发现、抓取配置与规则选择器。
2. **逐副本查看统计证据。** 查询下面几条原始序列，保留 `job`、`instance`，不要先求全局平均。时间戳为 0 表示从未成功刷新；不要将它解释成一次发生在 1970 年的真实刷新。
3. **定位刷新问题。** 核对 `CONTAINMENT_STATS_REFRESH_SECONDS` 与 `CONTAINMENT_STATS_WINDOW_DAYS` 的指定值、刷新任务启动状态，以及 `containment_refresh_seconds_count` 的 `outcome="ok"`/`"error"` 变化。错误增长支持刷新失败的判断；没有增长也可能是任务未启用、未启动或进程重启，不能反推 DB 健康。
4. **检查时钟。** 对比应用与 Prometheus 主机时间，确认时间戳是 Unix 秒而非毫秒。未来偏移也会影响新鲜度判断；不要通过手工改 Gauge、时间戳或放宽业务率阈值掩盖问题。
5. **统计可用后再看业务。** 对低率检查窗口内 served 样本量、工单与差评的组成，以及独立的解决、重开、质检和失败回复证据。人工需求或流量构成变化也能影响代理率，不能仅凭低率断定模型、检索或业务系统故障。

可在 Prometheus 查询页逐条执行：

```promql
up{job="rag-chatbot-api"}
```

```promql
chat_containment_stats_last_success_timestamp_seconds{job="rag-chatbot-api"}
```

```promql
chat_containment_stats_last_success_timestamp_seconds{job="rag-chatbot-api"} - time()
```

```promql
chat_sessions_served{job="rag-chatbot-api"}
```

```promql
chat_containment_rate{job="rag-chatbot-api"}
```

当前 Grafana 会话面板使用成功时间戳大于零等展示过滤，仍可能显示成功后的陈旧值，且不是本页告警的完整 job、新鲜度和 `up` 门控。刷新年龄面板取最近成功副本的最大时间戳，不能证明全部副本新鲜。排障应以按 job/instance 拆开的序列及本页规则为准，不能拿图上均值直接替代告警表达式。

## 恢复确认

修复发现、连接、任务配置或时钟原因后，等待真实后台刷新和下一次抓取，确认成功时间戳持续前进、目标与统计字段重新有效。正常 DB 恢复后，循环会继续尝试，无需为清除告警而伪造一次成功刷新。

不可用条件消失后可在下一次规则求值解除；时钟告警还可能保留 10 分钟。若代理率仍低，新的有效序列需要重新满足 30 分钟业务条件。零流量时只确认统计可用，不能报告“低率恢复为成功解决”。

发现集合恢复有效不证明期望副本数量或容量已恢复：DNS 已移除的某个副本不会继续作为坏目标计数。期望副本数、聊天可用性、生产容量及备份恢复需要各自的监控或验证。Prometheus 自身不可用时，也不能靠其本机规则保证发出通知。

## 验证与通知边界

规则时序 fixture 位于 [`containment-alerts.test.yml`](../../deploy/prometheus/containment-alerts.test.yml)。使用与 Compose 一致的 Prometheus **2.47.2** 工具检查与执行：

```bash
promtool check rules deploy/prometheus/alerts.yml
promtool test rules deploy/prometheus/containment-alerts.test.yml
```

规则引擎的语法和合成时序验证不等于真实生产抓取或客户质量验证，也不证明远端 CI 已通过。当前仓库 Prometheus 配置没有 Alertmanager 目标，Compose 未配置 Alertmanager 服务与通知 receiver；Prometheus 计算出 firing 可以在其界面查看，不能据此声称邮件、Slack 或值班通知已经送达。通知路由、receiver、网络与端到端投递演练需要另行配置和验证。

PromQL 匹配与集合过滤参考 [Prometheus operators](https://prometheus.io/docs/prometheus/latest/querying/operators/)，`up` 的含义参考 [Jobs and instances](https://prometheus.io/docs/concepts/jobs_instances/)，pending/保持行为参考 [Alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)。固定版本对 `keep_firing_for` 的支持见 [Prometheus v2.47.2 rulefmt 源码](https://raw.githubusercontent.com/prometheus/prometheus/v2.47.2/model/rulefmt/rulefmt.go)。

旧 `HandoffShareHigh` 的“containment inverse”等注释不能延续使用：其分母是可能重叠的漏斗层触达，不是本页会话分母，不能等同于 `1 - containment_rate`。该漏斗层触达趋势指标及其余 handoff 告警不是本页三种信号的替代品。

本轮实际验证为 13 条规则检查、46 个合成时序场景和 315 次原生告警断言；这些断言不是 pytest 用例数。前后证据及完整应用回归范围见[告警验证记录](../reviews/containment-alert-results-2026-10-28.json)。
