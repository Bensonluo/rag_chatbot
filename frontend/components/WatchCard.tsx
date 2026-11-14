'use client';

import { type CommerceWatch } from '@/lib/api';

/**
 * Structured price-watch ask card ("pre-authorized condition + trigger + ask").
 *
 * Every rendered price and name comes from the server's structured watch
 * payload (durable rule row + provider price fact). When the reply text
 * disagrees with the card, the card wins. The card only ASKS whether to
 * order now: the order button reuses the ordinary chat channel, and the
 * actual purchase still goes through quote → order → payment confirmation.
 * Dismissal is local-only — the server already recorded the ask as
 * delivered, so no turn is spent on it.
 */
export type WatchCardStatus = 'active' | 'ordered' | 'dismissed';

export interface WatchCardProps {
  watch: CommerceWatch;
  status: WatchCardStatus;
  busy: boolean;
  isEn: boolean;
  onOrder: (watch: CommerceWatch) => void;
  onDismiss: (watch: CommerceWatch) => void;
}

export function watchKey(watch: CommerceWatch): string {
  return watch.watch_id;
}

function formatMinor(minor: number): string {
  return `¥${(minor / 100).toFixed(2)}`;
}

export function WatchCard({ watch, status, busy, isEn, onOrder, onDismiss }: WatchCardProps) {
  const disabled = status !== 'active' || busy;
  const saved = watch.target_price_minor - watch.current_price_minor;
  const title = isEn ? 'Price drop on your watch' : '你关注的价格降价了';

  return (
    <div
      data-watch-card={watch.watch_id}
      data-watch-status={status}
      role="group"
      aria-label={title}
      className={`w-full max-w-[80%] rounded-lg border p-3 ${status === 'active' ? 'border-primary/40 bg-background' : 'border-border bg-muted opacity-60'}`}
    >
      <p className="text-sm font-medium">{title}</p>
      <p className="mt-2 text-sm" data-watch-item>
        {watch.sku_name || watch.sku_id}
        {watch.sku_name ? <span className="ml-1 text-xs text-muted-foreground">[{watch.sku_id}]</span> : null}
      </p>
      <p className="mt-1 text-sm">
        {isEn ? 'Target price: ' : '目标价：'}
        <span data-watch-target>{formatMinor(watch.target_price_minor)}</span>
        <span className="mx-2 text-muted-foreground">·</span>
        {isEn ? 'Now: ' : '现价：'}
        <span data-watch-current className="font-medium">{formatMinor(watch.current_price_minor)}</span>
      </p>
      {saved > 0 && (
        <p className="mt-1 text-xs text-muted-foreground" data-watch-saved>
          {isEn ? `At or below your target by ${formatMinor(saved)}` : `已低于你的目标价 ${formatMinor(saved)}`}
        </p>
      )}
      <p className="mt-1 text-xs text-muted-foreground" data-watch-meta>
        {isEn
          ? 'Prices come from the server-confirmed watch rule and prevail over the reply text. Ordering still needs your explicit confirmation.'
          : '价格来自服务端确认的关注规则，与回复文字不符时以本卡为准；下单仍需你的明确确认。'}
      </p>
      <div className="mt-2 flex items-center gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onOrder(watch)}
          aria-label={watch.action_label}
          data-watch-order
          className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:cursor-default disabled:opacity-50"
        >
          {watch.action_label}
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onDismiss(watch)}
          aria-label={isEn ? 'Not now' : '先不用'}
          data-watch-dismiss
          className="rounded-md border px-3 py-1.5 text-sm disabled:cursor-default disabled:opacity-50"
        >
          {isEn ? 'Not now' : '先不用'}
        </button>
        {status === 'ordered' && (
          <span className="text-xs text-muted-foreground" data-watch-ordered>
            {isEn ? 'Order requested' : '已请求下单'}
          </span>
        )}
        {status === 'dismissed' && (
          <span className="text-xs text-muted-foreground" data-watch-dismissed>
            {isEn ? 'Dismissed' : '已忽略'}
          </span>
        )}
      </div>
    </div>
  );
}
