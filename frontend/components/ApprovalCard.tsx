'use client';

import { type CommerceApproval } from '@/lib/api';

/**
 * Structured purchase confirmation card.
 *
 * Every rendered amount, line and deadline comes from the server's
 * structured approval payload (durable quote confirmation). When the
 * assistant's reply text disagrees with the card, the card wins — the
 * component never parses or trusts model prose.
 */
export type ApprovalCardStatus = 'active' | 'done' | 'cancelled';

export interface ApprovalCardProps {
  approval: CommerceApproval;
  status: ApprovalCardStatus;
  busy: boolean;
  /** Injected epoch ms so countdown rendering stays deterministic in tests. */
  now: number;
  isEn: boolean;
  onConfirm: (approval: CommerceApproval) => void;
  onCancel: (approval: CommerceApproval) => void;
}

export function approvalKey(approval: CommerceApproval): string {
  return `${approval.purchase_id}:${approval.action}`;
}

function formatMinor(minor: number): string {
  return `¥${(minor / 100).toFixed(2)}`;
}

function countdown(expiresAt: string, now: number): number {
  const remaining = Math.floor((Date.parse(expiresAt) - now) / 1000);
  return remaining > 0 ? remaining : 0;
}

function mmss(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

export function ApprovalCard({ approval, status, busy, now, isEn, onConfirm, onCancel }: ApprovalCardProps) {
  const orderCard = approval.action === 'confirm_order';
  const remaining = countdown(approval.expires_at, now);
  const expired = orderCard && status === 'active' && remaining <= 0;
  const disabled = status !== 'active' || busy || expired;
  const title = approval.action === 'confirm_order'
    ? (isEn ? 'Confirm your order' : '请确认本次下单')
    : (isEn ? 'Confirm payment' : '请确认支付');

  return (
    <div
      data-approval-card={approvalKey(approval)}
      data-approval-status={expired ? 'expired' : status}
      role="group"
      aria-label={title}
      className={`w-full max-w-[80%] rounded-lg border p-3 ${status === 'active' && !expired ? 'border-primary/40 bg-background' : 'border-border bg-muted opacity-60'}`}
    >
      <p className="text-sm font-medium">{title}</p>
      <table className="mt-2 w-full text-xs">
        <thead>
          <tr className="text-muted-foreground">
            <th scope="col" className="text-left font-normal">{isEn ? 'Item' : '商品'}</th>
            <th scope="col" className="text-right font-normal">{isEn ? 'Qty' : '数量'}</th>
            <th scope="col" className="text-right font-normal">{isEn ? 'Unit price' : '单价'}</th>
            <th scope="col" className="text-right font-normal">{isEn ? 'Subtotal' : '小计'}</th>
          </tr>
        </thead>
        <tbody>
          {approval.lines.map(line => (
            <tr key={line.sku_id} data-approval-line={line.sku_id}>
              <td className="py-0.5">{line.name || line.sku_id}</td>
              <td className="py-0.5 text-right">{line.quantity}</td>
              <td className="py-0.5 text-right">{formatMinor(line.unit_price_minor)}</td>
              <td className="py-0.5 text-right">{formatMinor(line.subtotal_minor)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-sm font-medium">
        {isEn ? 'Total: ' : '合计：'}
        <span data-approval-total>{formatMinor(approval.total_minor)}</span>
      </p>
      <p className="mt-1 text-xs text-muted-foreground" data-approval-meta>
        {orderCard
          ? (expired
            ? (isEn ? 'Quote expired — request a new quote before ordering.' : '报价已过期，请重新试算后再下单。')
            : (isEn ? `Quote expires in ${mmss(remaining)}` : `报价 ${mmss(remaining)} 后过期`))
          : (isEn ? 'Order created; awaiting payment.' : '订单已创建，等待支付。')}
      </p>
      <p className="mt-1 text-xs text-muted-foreground">
        {isEn
          ? 'Amounts come from the server-confirmed quote and prevail over the reply text.'
          : '金额来自服务端确认的报价，与回复文字不符时以本卡为准。'}
      </p>
      <div className="mt-2 flex items-center gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onConfirm(approval)}
          aria-label={approval.action_label}
          data-approval-confirm
          className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:cursor-default disabled:opacity-50"
        >
          {approval.action_label}
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onCancel(approval)}
          aria-label={isEn ? 'Cancel' : '取消'}
          data-approval-cancel
          className="rounded-md border px-3 py-1.5 text-sm disabled:cursor-default disabled:opacity-50"
        >
          {isEn ? 'Cancel' : '取消'}
        </button>
        {status === 'done' && (
          <span className="text-xs text-muted-foreground" data-approval-done>
            {isEn ? 'Completed' : '已完成'}
          </span>
        )}
        {status === 'cancelled' && (
          <span className="text-xs text-muted-foreground" data-approval-cancelled>
            {isEn ? 'Cancelled' : '已取消'}
          </span>
        )}
      </div>
    </div>
  );
}
