'use client';

/**
 * Live execution-chain panel for the CS agent demo.
 *
 * Renders pipeline trace events (multiplexed onto the chat SSE stream by the
 * backend) as per-turn stage cards: 意图/槽位/检索/记忆/声明门控/工具链/转人工,
 * plus client-measured turn metrics (TTFT, total, chunks, events).
 * Pure presentational + one pure reducer — all streaming lives in the chat
 * component that feeds it.
 */

import { Activity, Gauge, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { TraceEvent } from '@/lib/api';
import { useLanguage } from '@/lib/i18n/context';

// —— stage metadata ——————————————————————————————————————————————

type StageIcon = typeof Activity;

const STAGE_META: Record<string, { label: [string, string]; icon: StageIcon }> = {
  'cs.guardrail_input': { label: ['输入护栏', 'Input Guardrail'], icon: Activity },
  'cs.intent': { label: ['意图识别', 'Intent'], icon: Activity },
  'cs.slots': { label: ['槽位抽取', 'Slot Filling'], icon: Activity },
  'cs.faq': { label: ['FAQ 快速路径', 'FAQ Fast Path'], icon: Activity },
  'cs.retrieval': { label: ['混合检索', 'Hybrid Retrieval'], icon: Activity },
  'cs.memory': { label: ['记忆召回', 'Memory Recall'], icon: Activity },
  'cs.claim_gate': { label: ['声明门控', 'Claim Gate'], icon: Activity },
  'cs.agent': { label: ['Agent 工具循环', 'Agent Tool Loop'], icon: Activity },
  'cs.tool': { label: ['工具执行', 'Tool Execution'], icon: Activity },
  'cs.generation': { label: ['LLM 生成', 'LLM Generation'], icon: Activity },
  'cs.direct': { label: ['直答生成', 'Direct Response'], icon: Activity },
  'cs.handoff': { label: ['转人工', 'Human Handoff'], icon: Activity },
  'cs.pipeline': { label: ['管道入口', 'Pipeline Entry'], icon: Activity },
  'cs.pipeline.stream': { label: ['流式管道', 'Streaming Pipeline'], icon: Activity },
  'cs.persistence': { label: ['会话持久化', 'Persistence'], icon: Activity },
};


const TOOL_META: Record<string, [string, string]> = {
  // Shopping tools
  commerce_graph_context: ['商品图谱语义查询', 'Product Graph Context'],
  commerce_graph_upsert: ['图谱更新', 'Graph Upsert'],
  commerce_discover_catalog: ['商品类目发现', 'Catalog Discovery'],
  commerce_propose_category: ['动态铸类', 'Dynamic Category Mint'],
  commerce_search_products: ['商品搜索', 'Product Search'],
  commerce_get_product: ['商品详情查询', 'Product Detail'],
  commerce_update_cart_lines: ['更新购物车', 'Update Cart'],
  commerce_prepare_checkout: ['生成报价', 'Prepare Quote'],
  commerce_purchase_status: ['查询购买状态', 'Purchase Status'],
  commerce_commit_checkout: ['确认下单', 'Commit Checkout'],
  commerce_mock_pay: ['模拟付款', 'Mock Payment'],
  commerce_remember_unsupported_domain: ['记录目录外领域', 'Remember Unsupported Domain'],
  // Support / knowledge tools
  search_knowledge_base: ['知识库检索', 'Knowledge Search'],
  escalate_to_human: ['转人工', 'Escalate to Human'],
  process_refund: ['处理退款', 'Process Refund'],
  process_return: ['处理退货', 'Process Return'],
  query_order_status: ['订单状态查询', 'Order Status'],
  track_shipping: ['物流跟踪', 'Shipping Tracking'],
  submit_complaint: ['提交投诉', 'Submit Complaint'],
  get_recent_orders: ['最近订单查询', 'Recent Orders'],
};

function toolTitle(stage: string, detail: unknown): [string, string] | undefined {
  if (stage !== 'cs.tool' || !isRecord(detail)) return undefined;
  const tool = typeof detail.tool === 'string' ? detail.tool.trim() : '';
  if (!tool) return undefined;
  return Object.prototype.hasOwnProperty.call(TOOL_META, tool)
    ? TOOL_META[tool]
    : [tool, tool];
}

export interface StageCard {
  key: number;
  stage: string;
  status: 'running' | 'done' | 'error';
  ms?: number;
  title?: [string, string];
  details: Record<string, unknown>[];
}

export interface TurnStats {
  ttftMs: number | null;
  totalMs: number | null;
  chunkCount: number;
  eventCount: number;
  stagesDone: number;
}

export const EMPTY_STATS: TurnStats = {
  ttftMs: null,
  totalMs: null,
  chunkCount: 0,
  eventCount: 0,
  stagesDone: 0,
};

/** Merge one trace event into the stage-card list (pure). */
export function applyTraceEvent(
  cards: StageCard[],
  ev: TraceEvent,
  nextKey: { n: number },
): StageCard[] {
  const lastIndexOf = [...cards].reverse().findIndex((c) => c.stage === ev.stage);
  const idx = lastIndexOf === -1 ? -1 : cards.length - 1 - lastIndexOf;

  if (ev.status === 'start') {
    return [...cards, {
      key: nextKey.n++,
      stage: ev.stage,
      status: 'running',
      title: toolTitle(ev.stage, ev.detail),
      details: [],
    }];
  }
  if (idx === -1) {
    // detail/end arrived without a start (mid-stream attach) — still show it
    const card: StageCard = {
      key: nextKey.n++,
      stage: ev.stage,
      status: 'running',
      title: toolTitle(ev.stage, ev.detail),
      details: [],
    };
    if (ev.status === 'detail') card.details.push(ev.detail ?? {});
    if (ev.status === 'end' || ev.status === 'error') {
      card.status = ev.status === 'end' ? 'done' : 'error';
      card.ms = ev.ms ?? undefined;
    }
    return [...cards, card];
  }
  return cards.map((c, i) => {
    if (i !== idx) return c;
    const copy: StageCard = { ...c, details: [...c.details] };
    copy.title = copy.title ?? toolTitle(ev.stage, ev.detail);
    if (ev.status === 'detail') copy.details.push(ev.detail ?? {});
    if (ev.status === 'end' || ev.status === 'error') {
      copy.status = ev.status === 'end' ? 'done' : 'error';
      copy.ms = ev.ms ?? copy.ms;
    }
    return copy;
  });
}

function fmtMs(ms: unknown): string {
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) return '';
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

// —— per-stage detail rendering ————————————————————————————————————

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function stringItems(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item: unknown): item is string => typeof item === 'string' && item.trim().length > 0)
    : [];
}

function textValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function countValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function positiveCount(value: unknown): number | null {
  const count = countValue(value);
  return count !== null && count > 0 ? count : null;
}

function isScalar(value: unknown): value is string | number | boolean {
  return typeof value === 'string' || typeof value === 'boolean' ||
    (typeof value === 'number' && Number.isFinite(value));
}

function DetailView({ stage, d, isEn }: { stage: string; d: Record<string, unknown>; isEn: boolean }) {
  if (stage === 'cs.intent') {
    const conf = typeof d.confidence === 'number' && Number.isFinite(d.confidence) &&
      d.confidence >= 0 && d.confidence <= 1 ? Math.round(d.confidence * 100) : null;
    return (
      <div className="flex items-center gap-2 text-xs">
        <span className="rounded bg-indigo-500/15 px-1.5 py-0.5 font-mono text-indigo-500">
          {textValue(d.intent, '?')}
        </span>
        {conf != null && (
          <span className="text-muted-foreground">
            {isEn ? 'confidence' : '置信度'} <span className="font-mono">{conf}%</span>
          </span>
        )}
      </div>
    );
  }
  if (stage === 'cs.slots') {
    const filledEntries = Object.entries(isRecord(d.filled_slots) ? d.filled_slots : {});
    const fieldsKnown = (d.filled_slots === undefined || isRecord(d.filled_slots)) &&
      filledEntries.every(([, value]) => isScalar(value)) &&
      (d.pending_slots === undefined || (Array.isArray(d.pending_slots) && d.pending_slots.every(value => typeof value === 'string')));
    const filled = filledEntries
      .filter(([, value]) => isScalar(value));
    const pending = stringItems(d.pending_slots);
    return (
      <div className="flex flex-wrap gap-1.5 text-xs">
        {filled.map(([k, v]) => (
          <span key={k} className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-emerald-600">
            {k}={String(v).slice(0, 18)}
          </span>
        ))}
        {pending.map((k) => (
          <span key={k} className="rounded-full bg-amber-500/15 px-2 py-0.5 text-amber-600">
            {k}?
          </span>
        ))}
        {!filled.length && !pending.length && (
          <span className="text-muted-foreground">
            {fieldsKnown ? (isEn ? 'no slots' : '无槽位') : (isEn ? 'slot details unavailable' : '槽位信息未知')}
          </span>
        )}
      </div>
    );
  }
  if (stage === 'cs.retrieval') {
    const sources = stringItems(d.sources);
    return (
      <div className="text-xs">
        <div className="flex items-center gap-2">
          <span className="rounded bg-sky-500/15 px-1.5 py-0.5 font-mono text-sky-600">
            {textValue(d.mode, '?')}
          </span>
          <span className="text-muted-foreground">
            {isEn ? `${countValue(d.docs) ?? '?'} hits` : `命中 ${countValue(d.docs) ?? '?'} 条`}
          </span>
        </div>
        {sources.length > 0 && (
          <div className="mt-1 flex flex-col gap-0.5 font-mono text-[10px] text-muted-foreground">
            {sources.map((s, i) => (
              <span key={i} className="truncate">↳ {s}</span>
            ))}
          </div>
        )}
      </div>
    );
  }
  if (stage === 'cs.claim_gate') {
    if (d.action === 'rewrite') {
      return (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 p-2 text-xs">
          <div className="mb-1 font-medium text-red-600">
            {isEn ? 'Claim intercepted' : '声明拦截'} · {textValue(d.reason)}
          </div>
          <div className="font-mono text-[11px] text-muted-foreground line-through decoration-red-500/70">
            {textValue(d.clause)}
          </div>
          <div className="mt-0.5 font-mono text-[11px] text-emerald-600">↳ {textValue(d.grounded)}</div>
        </div>
      );
    }
    const facts = countValue(d.facts);
    return (
      <div className="flex items-center gap-1.5 text-xs text-emerald-600">
        {facts === null ? (isEn ? 'fact count unavailable' : '事实数量未知') : isEn
          ? `${facts} curated facts verified`
          : `${facts} 条策展事实校验通过`}
      </div>
    );
  }
  if (stage === 'cs.agent' || stage === 'cs.tool') {
    const tools = stringItems(d.tools);
    const toolsKnown = d.tools === undefined || (Array.isArray(d.tools) && d.tools.every(value => typeof value === 'string'));
    return (
      <div className="flex flex-wrap items-center gap-1 text-xs">
        {tools.length ? (
          tools.map((t, i) => (
            <span key={i} className="flex items-center gap-1">
              {i > 0 && <span className="text-muted-foreground">→</span>}
              <span className="rounded bg-violet-500/15 px-1.5 py-0.5 font-mono text-violet-600">{t}</span>
            </span>
          ))
        ) : (
          <span className="text-muted-foreground">
            {toolsKnown ? (isEn ? 'no tools called' : '未调用工具') : (isEn ? 'tool details unavailable' : '工具调用信息未知')}
          </span>
        )}
      </div>
    );
  }
  if (stage === 'cs.handoff') {
    return (
      <div className="text-xs">
        <span className="rounded bg-orange-500/15 px-1.5 py-0.5 text-orange-600">
          {isEn ? 'Ticket' : '工单'} #{positiveCount(d.ticket_id) ?? '—'}
        </span>{' '}
        <span className="text-muted-foreground">
          {isEn ? 'queue position' : '排队位置'} {positiveCount(d.queue_position) ?? '—'}
        </span>
      </div>
    );
  }
  if (stage === 'cs.memory') {
    return (
      <div className="text-xs text-muted-foreground">
        {isEn
          ? `${countValue(d.facts) ?? '?'} cross-session facts recalled`
          : `召回 ${countValue(d.facts) ?? '?'} 条跨会话记忆`}
      </div>
    );
  }
  if (stage === 'cs.faq') {
    return (
      <div className="text-xs">
        <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-emerald-600">
          {d.hit === true ? `hit:${textValue(d.faq_id, '?')}` : d.hit === false ? 'miss → RAG' : '?'}
        </span>
      </div>
    );
  }
  if (stage === 'cs.guardrail_input') {
    return (
      <div className="text-xs">
        {d.blocked === true ? (
          <span className="text-red-600">
            {isEn ? 'Blocked' : '拦截'}：{stringItems(d.violations)[0] ?? ''}
          </span>
        ) : d.redacted === true ? (
          <span className="text-amber-600">{isEn ? 'PII redacted' : 'PII 已脱敏'}</span>
        ) : (
          <span className="text-muted-foreground">
            {d.blocked === false ? (isEn ? 'passed' : '通过') : (isEn ? 'status unavailable' : '状态未知')}
          </span>
        )}
      </div>
    );
  }
  return null;
}

function StageCardView({ card, isEn }: { card: StageCard; isEn: boolean }) {
  const meta = Object.prototype.hasOwnProperty.call(STAGE_META, card.stage) ? STAGE_META[card.stage] : undefined;
  const Icon = meta?.icon ?? Activity;
  return (
    <div
      className={`rounded-lg border p-2.5 transition-colors ${
        card.status === 'error'
          ? 'border-red-500/40 bg-red-500/5'
          : card.status === 'running'
            ? 'border-indigo-500/40 bg-indigo-500/5'
            : 'border-black/10 bg-black/[0.02] dark:border-white/10 dark:bg-white/[0.02]'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-sm font-medium">
          <Icon size={14} className="text-indigo-500" />
          {card.title?.[isEn ? 1 : 0] ?? meta?.label[isEn ? 1 : 0] ?? card.stage}
        </div>
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          {card.status === 'running' && <Loader2 size={12} className="animate-spin text-indigo-500" />}
          {card.ms != null && <span className="font-mono">{fmtMs(card.ms)}</span>}
        </div>
      </div>
      {card.details.length > 0 && (
        <div className="mt-1.5 flex flex-col gap-1.5">
          {card.details.map((d, i) => (
            <DetailView key={i} stage={card.stage} d={d} isEn={isEn} />
          ))}
        </div>
      )}
    </div>
  );
}

// —— panel ——————————————————————————————————————————————————————————

export function ExecutionChainPanel({
  stages,
  stats,
  turn,
}: {
  stages: StageCard[];
  stats: TurnStats;
  turn: number;
}) {
  const { locale } = useLanguage();
  const isEn = locale === 'en';

  return (
    <Card className="flex h-[699px] flex-col lg:h-[calc(100dvh-12rem)] lg:min-h-[699px]">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{isEn ? 'Execution Chain' : '执行链路'}</CardTitle>
          <span className="text-xs text-muted-foreground">
            {isEn ? 'Turn' : '第'} {turn} {isEn ? '' : '轮'}
          </span>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-3">
        <div className="min-h-[280px] flex-1 space-y-2 overflow-y-auto pr-1">
          {stages.length === 0 && (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              {isEn ? 'Send a message to watch each stage run live' : '发送一条消息，观察每一阶段的实时执行'}
            </div>
          )}
          {stages.map((c) => (
            <StageCardView key={c.key} card={c} isEn={isEn} />
          ))}
        </div>

        {/* Turn metrics */}
        <div className="rounded-lg border p-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <Gauge className="h-4 w-4 text-indigo-500" />
            {isEn ? 'Turn Metrics' : '本轮指标'}
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3">
            {[
              [isEn ? 'TTFT' : 'TTFT 首字', stats.ttftMs != null ? fmtMs(stats.ttftMs) : '—'],
              [isEn ? 'Total' : '总时长', stats.totalMs != null ? fmtMs(stats.totalMs) : '—'],
              [isEn ? 'Chunks' : '内容块', String(stats.chunkCount)],
              [isEn ? 'Events' : '管线事件', String(stats.eventCount)],
              [isEn ? 'Stages done' : '完成阶段', String(stats.stagesDone)],
            ].map(([k, v]) => (
              <div key={k} className="flex items-center justify-between gap-2">
                <dt className="text-muted-foreground">{k}</dt>
                <dd className="font-mono">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </CardContent>
    </Card>
  );
}
