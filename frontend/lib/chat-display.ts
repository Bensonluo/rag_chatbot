/**
 * Chat display sanitization.
 *
 * The agent's tool results travel through delimited machine channels
 * (`<catalog_data>`, `<graph_data>`, `<session_memory>`, `<order_timeline>`,
 * `<recent_orders>`). The backend prompt forbids echoing them, and this
 * module is the frontend's defense in depth: even if a reply regresses to
 * pasting a block, it never renders raw machine data to the visitor.
 */

import type { TraceEvent } from "./api";

/** Machine channels that may appear in assistant replies. */
const DATA_CHANNEL_TAGS = [
  "catalog_data",
  "graph_data",
  "session_memory",
  "order_timeline",
  "recent_orders",
] as const;

/** Fully-delimited blocks, non-greedy so adjacent blocks each match. The
 * pattern also consumes the block's own trailing newline so removing a
 * block that sits on its own line leaves no blank line behind. */
const PAIRED_BLOCK_PATTERN = new RegExp(
  `[ \\t]*<(?:${DATA_CHANNEL_TAGS.join("|")})>[\\s\\S]*?</(?:${DATA_CHANNEL_TAGS.join("|")})>[ \\t]*(?:\\n|$)?`,
  "g",
);

/** Streaming partial: an opening tag whose closing tag has not arrived yet. */
const UNCLOSED_BLOCK_PATTERN = new RegExp(
  `[ \\t]*<(?:${DATA_CHANNEL_TAGS.join("|")})>[\\s\\S]*$`,
);

/**
 * Remove machine data channels from an assistant reply.
 *
 * Strips paired `<tag>…</tag>` blocks (any tag in the channel list, opening
 * and closing names matching independently is accepted — servers emit
 * well-formed pairs), then any trailing unclosed block so a partially
 * streamed reply cannot flash raw data. Collapses the blank runs the
 * removals leave behind and trims the result.
 */
export function stripDataChannels(text: string): string {
  if (!text) return text;
  let out = text.replace(PAIRED_BLOCK_PATTERN, "");
  out = out.replace(UNCLOSED_BLOCK_PATTERN, "");
  // Collapse 3+ newlines left by block removal, then trim edges.
  out = out.replace(/\n{3,}/g, "\n\n").trim();
  return out;
}

// ---------------------------------------------------------------------------
// Live turn status
//
// The backend streams named `event: trace` frames for every pipeline stage,
// LLM round, and tool call. The cockpit panel renders the full waterfall;
// these helpers distill the same stream into ONE visitor-facing status line
// so a 30-second agent turn is never silent — visible step status beats
// bare typing dots when tool calls cannot stream tokens.

/** Visitor-facing turn status derived from trace events. */
export type TurnStatusKey =
  | "understanding"
  | "thinking"
  | "composing"
  | "searching"
  | "discovering"
  | "cart"
  | "quoting"
  | "ordering"
  | "paying"
  | "refunding"
  | "watching"
  | "order_status"
  | "preferences"
  | "saving"
  | "working";

const TOOL_STATUS: [pattern: RegExp, key: TurnStatusKey][] = [
  [/^commerce_(search_products|discover_catalog)$/, "searching"],
  [/^commerce_propose_category$/, "discovering"],
  [/add_to_cart/, "cart"],
  [/prepare_checkout/, "quoting"],
  [/place_order|confirm_order/, "ordering"],
  [/mock_pay/, "paying"],
  [/refund/, "refunding"],
  [/watch/, "watching"],
  [/timeline|recent_orders|purchase_status/, "order_status"],
  [/preference|unsupported_domain/, "preferences"],
];

/**
 * Map one trace event to the status visitors should see.
 *
 * `undefined` means "no change" — a tool finishing keeps its label until the
 * next stage event arrives, so the line never flickers between steps.
 * Callers clear the line when answer tokens start streaming or the turn ends.
 */
export function turnStatusKeyFor(event: TraceEvent): TurnStatusKey | undefined {
  if (event.stage === "cs.generation" && event.status === "start") {
    const step = event.detail?.step;
    return typeof step === "number" && step > 0 ? "composing" : "thinking";
  }
  if (event.stage === "cs.tool" && event.status === "start") {
    const tool = event.detail?.tool;
    if (typeof tool !== "string" || !tool) return "working";
    for (const [pattern, key] of TOOL_STATUS) if (pattern.test(tool)) return key;
    return "working";
  }
  if (event.stage === "cs.tool") return undefined; // keep the label through tool end/error
  if (event.stage === "cs.persistence" && event.status !== "error") return "saving";
  if (
    (event.stage === "cs.pipeline" || event.stage === "cs.agent" || event.stage === "cs.intent") &&
    (event.status === "start" || event.status === "detail")
  ) {
    return "understanding";
  }
  return undefined;
}

const TURN_STATUS_PHRASES: Record<TurnStatusKey, Record<"en" | "zh", string>> = {
  understanding: { en: "Understanding your request", zh: "正在理解你的请求" },
  thinking: { en: "Thinking", zh: "思考中" },
  composing: { en: "Composing the reply", zh: "正在组织回复" },
  searching: { en: "Searching the catalog", zh: "正在搜索商品目录" },
  discovering: { en: "Mapping the catalog", zh: "正在梳理商品类目" },
  cart: { en: "Updating the cart", zh: "正在更新购物车" },
  quoting: { en: "Preparing your quote", zh: "正在准备报价" },
  ordering: { en: "Placing the order", zh: "正在下单" },
  paying: { en: "Processing payment", zh: "正在处理支付" },
  refunding: { en: "Processing the refund", zh: "正在处理退款" },
  watching: { en: "Managing price alerts", zh: "正在管理降价提醒" },
  order_status: { en: "Checking order status", zh: "正在查询订单状态" },
  preferences: { en: "Updating preferences", zh: "正在更新偏好" },
  saving: { en: "Saving the conversation", zh: "正在保存对话" },
  working: { en: "Working with the storefront", zh: "正在调用商城服务" },
};

/** Localized label for a live turn status. */
export function turnStatusPhrase(key: TurnStatusKey, english: boolean): string {
  return TURN_STATUS_PHRASES[key][english ? "en" : "zh"];
}
