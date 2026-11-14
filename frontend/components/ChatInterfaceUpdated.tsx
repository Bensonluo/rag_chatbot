"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Send,
  Bot,
  User,
  AlertCircle,
  Wifi,
  WifiOff,
  Mic,
  Radio,
  Shield,
  AlertTriangle,
  ThumbsUp,
  ThumbsDown,
} from "lucide-react";
import { motion } from "framer-motion";
import {
  type ChatCompletion,
  type CommerceApproval,
  type CommerceWatch,
  type SourceCitation,
  type TraceEvent,
} from "@/lib/api";
import {
  GuestApi,
  GUEST_EXPIRED,
  guestScope,
  type GuestIdentity,
  type GuestScope,
} from "@/lib/guest-api";
import {
  stripDataChannels,
  turnStatusKeyFor,
  turnStatusPhrase,
  type TurnStatusKey,
} from "@/lib/chat-display";
import { SourceCitations } from "./SourceCitations";
import { RichText } from "./RichText";
import { useLanguage } from "@/lib/i18n/context";
import { VoiceButton, VoiceButtonMode } from "./VoiceButton";
import {
  ApprovalCard,
  approvalKey,
  type ApprovalCardStatus,
} from "./ApprovalCard";
import { WatchCard, watchKey, type WatchCardStatus } from "./WatchCard";
import {
  ExecutionChainPanel,
  applyTraceEvent,
  EMPTY_STATS,
  type StageCard,
  type TurnStats,
} from "./ExecutionChainPanel";

type ReplyStatus = "unknown" | "streaming" | "completed" | "failed";
type FailureReason =
  | "authentication"
  | "forbidden"
  | "session"
  | "rate_limit"
  | "server"
  | "network"
  | "interrupted"
  | "invalid"
  | "aborted"
  | "unknown";

/** One structured confirmation card; keyed by purchase + action so a stage
 * advance adds the next action's card instead of a duplicate control. */
interface ApprovalEntry {
  key: string;
  data: CommerceApproval;
  status: ApprovalCardStatus;
}

/** One price-watch ask card; keyed by watch_id — the server delivers each
 * ask exactly once, and the key keeps a re-delivery from stacking cards. */
interface WatchEntry {
  key: string;
  data: CommerceWatch;
  status: WatchCardStatus;
}

interface Message {
  id: string;
  status?: ReplyStatus;
  failure?: FailureReason;
  messageId?: number;
  rating?: 1 | -1;
  feedbackPending?: boolean;
  feedbackFailure?: FailureReason;
  scope?: GuestScope;
  /** Stream text arrived, but the server never confirmed persisting this turn. */
  unconfirmed?: boolean;
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  isVoice?: boolean;
  isWelcome?: boolean;
  intent?: string;
  guardrailBlocked?: boolean;
  guardrailViolations?: string[];
  citations?: SourceCitation[];
}

const WELCOME_EN =
  'Welcome to Agent Store — I\'m the store\'s AI agent, and I actually get things done: search → recommend → quote → your confirmation → order → separate payment confirmation (simulated, no real charge). I can also watch prices for you ("tell me when it drops under ¥300") and rebuild your order timeline from authoritative records. After-sales too: refunds / returns / orders / shipping / complaints — policy answers cite their sources, human handoff anytime, EN / 中文 both welcome.\n\nTry:\n· "Gift: noise-cancelling earbuds under ¥400" → the agent proposes candidates, prepares a quote, and orders only after you confirm; payment is another confirmation\n· "Tell me when ANC-02 drops under ¥300" → a pre-authorized price watch asks you exactly once, never ordering for you\n· "I want a refund" → the agent gathers details and executes only after your fresh confirmation; a repeated "confirm" won\'t refund twice';

const WELCOME_ZH =
  '欢迎来到 Agent Store——我是店里的 AI Agent，事情真的会办完：搜索 → 推荐 → 报价 → 你确认 → 下单 → 付款再确认（模拟支付，不真实扣款）。还能帮你盯价格、查订单：「ANC-02 降到 300 以内来问我」设预授权提醒，触发后只问你一次、绝不代下单；「我的订单怎么样了」从权威记录重建时间线。售后同样在行：退款/退货/订单/物流/投诉，政策问答附引用来源，随时转人工，中英文自动跟随。\n\n试试：\n·「送女友一副降噪耳机，预算 400 内」→ Agent 给候选、先报价，你确认才下单，付款再单独确认\n·「ANC-02 降到 300 以内来问我」→ 预授权降价提醒，只来问你一次，绝不代你下单\n·「我要退款」→ Agent 收集信息、执行前向你确认；重复说"确认"不会重复退款';

const STATUS_TEXT_EN = {
  checking: "Checking...",
  online: "Connected to Agent Store",
  offline: "Agent Store unavailable",
} as const;

const STATUS_TEXT_ZH = {
  checking: "检查中...",
  online: "已连接 Agent Store",
  offline: "Agent Store 当前不可用",
} as const;

const VOICE_LABELS_EN = { manual: "Manual", auto: "Auto" } as const;
const VOICE_LABELS_ZH = { manual: "手动录音", auto: "自动模式" } as const;

const STATUS_BAR_EN = {
  manual: "Click the microphone to start / stop recording",
  auto: "Click the button to start VAD auto-detection",
  offline: "API connection unavailable",
  notice:
    "The API connection is unavailable. Check the service before sending another request. For business actions, verify their processing status before resubmitting.",
} as const;

const STATUS_BAR_ZH = {
  manual: "点击麦克风开始/停止录音",
  auto: "点击按钮启动VAD自动语音检测",
  offline: "API 连接不可用",
  notice:
    "当前无法连接 API。请检查服务后再发送请求；涉及业务操作时，请先核实处理状态，再决定是否重新提交。",
} as const;

const PLACEHOLDER_EN = {
  online: "Type a message or use voice...",
  offline: "Type a message...",
} as const;

const PLACEHOLDER_ZH = {
  online: "输入消息或使用语音...",
  offline: "输入消息…",
} as const;

const FAILURE_TEXT: Record<"en" | "zh", Record<FailureReason, string>> = {
  en: {
    authentication: "Authentication failed. Sign in to continue.",
    forbidden: "You do not have permission to access this conversation.",
    session: "This conversation is unavailable.",
    rate_limit: "Too many requests. Wait before sending another request.",
    server: "The service could not finish the reply.",
    network: "The network connection was interrupted.",
    interrupted: "The reply was interrupted before it finished.",
    invalid: "The service did not return a valid reply.",
    aborted: "Stopped receiving the reply.",
    unknown:
      "The request failed. Its processing status could not be confirmed.",
  },
  zh: {
    authentication: "身份验证失败，请登录后继续。",
    forbidden: "您无权访问此会话。",
    session: "当前会话不可用。",
    rate_limit: "请求过于频繁，请稍后再发送请求。",
    server: "服务未能完成回复。",
    network: "网络连接中断。",
    interrupted: "回复在完成前中断。",
    invalid: "服务未返回有效回复。",
    aborted: "已停止接收回复。",
    unknown: "请求失败，无法确认处理状态。",
  },
};

function failureReason(error: unknown): FailureReason {
  if (!error || typeof error !== "object" || !("kind" in error))
    return "unknown";
  if (error.kind === "http") {
    const status = "status" in error ? error.status : undefined;
    if (status === 401) return "authentication";
    if (status === 403) return "forbidden";
    if (status === 404) return "session";
    if (status === 429) return "rate_limit";
    return "server";
  }
  switch (error.kind) {
    case "server_error":
      return "server";
    case "network":
      return "network";
    case "interrupted":
      return "interrupted";
    case "invalid_response":
      return "invalid";
    case "aborted":
      return "aborted";
    default:
      return "unknown";
  }
}

const FEEDBACK_FAILURE_TEXT: Record<
  "en" | "zh",
  Record<FailureReason, string>
> = {
  en: {
    authentication: "Your visitor session expired. Start again before rating.",
    forbidden: "You do not have permission to rate this reply.",
    session: "This reply is unavailable for feedback.",
    rate_limit: "Too many feedback requests. Try again later.",
    server: "Feedback could not be saved. Try again.",
    network:
      "Feedback was not confirmed. Check your connection before trying again.",
    interrupted: "Feedback submission was interrupted. Try again.",
    invalid: "Feedback was not confirmed by the service. Try again.",
    aborted: "Feedback submission was interrupted. Try again.",
    unknown: "Feedback could not be confirmed. Try again.",
  },
  zh: {
    authentication: "访客会话已失效，请重新开始后评价。",
    forbidden: "您无权评价这条回复。",
    session: "当前回复无法提交评价。",
    rate_limit: "评价请求过于频繁，请稍后再试。",
    server: "评价未能保存，请重试。",
    network: "尚未确认评价已保存，请检查连接后重试。",
    interrupted: "评价提交中断，请重试。",
    invalid: "服务未确认评价结果，请重试。",
    aborted: "评价提交中断，请重试。",
    unknown: "无法确认评价结果，请重试。",
  },
};

function isFeedbackEligible(
  message: Message,
): message is Message & { messageId: number } {
  return (
    message.role === "assistant" &&
    message.status === "completed" &&
    !message.isWelcome &&
    !message.guardrailBlocked &&
    Boolean(message.content.trim()) &&
    typeof message.messageId === "number" &&
    Number.isSafeInteger(message.messageId) &&
    message.messageId > 0
  );
}

export function ChatInterfaceUpdated({
  showExecutionChain = false,
}: {
  showExecutionChain?: boolean;
}) {
  const { locale } = useLanguage();
  const isEn = locale === "en";

  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      status: "unknown",
      content: isEn ? WELCOME_EN : WELCOME_ZH,
      isWelcome: true,
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  // Live agent status derived from trace events; null hides the line.
  // Visible step status keeps a 30s tool-calling turn from feeling frozen.
  const [turnStatusKey, setTurnStatusKey] = useState<TurnStatusKey | null>(null);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">(
    "checking",
  );
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [voiceDraftNotice, setVoiceDraftNotice] = useState(false);
  const [guest, setGuest] = useState<GuestIdentity | null>(null);
  const guestRef = useRef<GuestIdentity | null>(null);
  const [guestLoading, setGuestLoading] = useState(true);
  const [guestError, setGuestError] = useState<
    "expired" | "unavailable" | null
  >(null);
  const guestRequest = useRef<AbortController | null>(null);
  const historyRequests = useRef(new Set<AbortController>());
  const feedbackRequests = useRef(new Map<string, AbortController>());
  const [voiceMode, setVoiceMode] = useState<VoiceButtonMode>("manual");
  const [approvals, setApprovals] = useState<ApprovalEntry[]>([]);
  const [approvalNow, setApprovalNow] = useState<number>(() => Date.now());
  const [watches, setWatches] = useState<WatchEntry[]>([]);

  // Stick-to-bottom scrolling for the bounded chat viewport: new turns and
  // stream chunks keep the latest reply in view, but a reader who scrolled
  // up is never yanked back down.
  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);

  const handleMessagesScroll = useCallback(() => {
    const el = messagesScrollRef.current;
    if (!el) return;
    stickToBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  useEffect(() => {
    const el = messagesScrollRef.current;
    if (el && stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, isLoading, approvals, watches]);

  // One shared ticker drives every active order card's countdown; payment
  // cards have no expiry pressure, so no tick is scheduled for them alone.
  const hasLiveOrderCard = approvals.some(
    (entry) =>
      entry.status === "active" && entry.data.action === "confirm_order",
  );
  useEffect(() => {
    if (!hasLiveOrderCard) return;
    const timer = setInterval(() => setApprovalNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [hasLiveOrderCard]);

  // A freshly delivered approval card: append once per purchase+action and
  // grey out earlier action cards of the same purchase (state advanced).
  const acceptApproval = useCallback((approval: CommerceApproval) => {
    setApprovals((previous) => {
      const key = approvalKey(approval);
      const advanced = approval.action === "confirm_payment";
      const next = previous.map((entry) =>
        entry.key === key
          ? entry.status === "active"
            ? { ...entry, data: approval }
            : entry
          : advanced &&
              entry.status === "active" &&
              entry.data.purchase_id === approval.purchase_id
            ? { ...entry, status: "done" as ApprovalCardStatus }
            : entry,
      );
      if (next.some((entry) => entry.key === key)) return next;
      return [
        ...next,
        { key, data: approval, status: "active" as ApprovalCardStatus },
      ];
    });
  }, []);

  // A freshly delivered price-watch ask: append once per watch rule. The
  // server claims each ask exactly once; the key is the defensive second line.
  const acceptWatch = useCallback((watch: CommerceWatch) => {
    setWatches((previous) => {
      const key = watchKey(watch);
      if (previous.some((entry) => entry.key === key)) return previous;
      return [
        ...previous,
        { key, data: watch, status: "active" as WatchCardStatus },
      ];
    });
  }, []);

  // Execution-chain panel state (only exercised when showExecutionChain)
  const [chainStages, setChainStages] = useState<StageCard[]>([]);
  const [chainStats, setChainStats] = useState<TurnStats>(EMPTY_STATS);
  const [chainTurn, setChainTurn] = useState(0);
  const conversationStarted = useRef(false);
  const mounted = useRef(true);
  const nextRequestId = useRef(0);
  const activeRequest = useRef<{
    id: number;
    controller: AbortController;
  } | null>(null);

  useEffect(() => {
    mounted.current = true;
    const historyPending = historyRequests.current;
    const feedbackPending = feedbackRequests.current;
    return () => {
      mounted.current = false;
      const request = activeRequest.current;
      activeRequest.current = null;
      request?.controller.abort();
      guestRequest.current?.abort();
      guestRequest.current = null;
      historyPending.forEach((controller) => controller.abort());
      feedbackPending.forEach((controller) => controller.abort());
    };
  }, []);

  const resetMessages = () =>
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        status: "unknown",
        content: "",
        isWelcome: true,
      },
    ]);
  const startGuest = useCallback(async () => {
    if (guestRequest.current) return;
    const controller = new AbortController();
    guestRequest.current = controller;
    setGuestLoading(true);
    setGuestError(null);
    try {
      const identity = await GuestApi.bootstrap(controller.signal);
      if (!mounted.current || controller.signal.aborted) return;
      guestRef.current = identity;
      setGuest(identity);
      setGuestLoading(false);
      if (guestRequest.current === controller) guestRequest.current = null;
      historyRequests.current.add(controller);
      const scope = guestScope(identity);
      const history = await GuestApi.getChatHistory(
        scope,
        50,
        controller.signal,
      );
      if (
        !mounted.current ||
        controller.signal.aborted ||
        conversationStarted.current ||
        guestRef.current?.session.session_key !== scope.session_key
      )
        return;
      const turns: Message[] = history.messages.flatMap((message, index) =>
        message.role === "user" || message.role === "assistant"
          ? [
              {
                ...message,
                id: `history-${index}`,
                role: message.role,
                scope,
                status:
                  message.status === "completed"
                    ? "completed"
                    : message.status === "failed"
                      ? "failed"
                      : "unknown",
                messageId: message.message_id,
                rating:
                  message.user_rating === 1 || message.user_rating === -1
                    ? message.user_rating
                    : undefined,
              },
            ]
          : [],
      );
      if (turns.length) setMessages(turns);
    } catch (error) {
      if (mounted.current && !controller.signal.aborted)
        setGuestError(
          failureReason(error) === "authentication" ? "expired" : "unavailable",
        );
    } finally {
      historyRequests.current.delete(controller);
      if (guestRequest.current === controller) {
        guestRequest.current = null;
        if (mounted.current) setGuestLoading(false);
      }
    }
  }, []);
  useEffect(() => {
    const onExpired = (event: Event) => {
      const old = (event as CustomEvent<GuestScope>).detail;
      const current = guestRef.current;
      if (
        !current ||
        old?.session_id !== current.session.id ||
        old.session_key !== current.session.session_key
      )
        return;
      guestRef.current = null;
      setGuest(null);
      setGuestError("expired");
      activeRequest.current?.controller.abort();
      activeRequest.current = null;
      feedbackRequests.current.forEach((controller) => controller.abort());
      feedbackRequests.current.clear();
      historyRequests.current.forEach((controller) => controller.abort());
      setIsLoading(false);
      setInput("");
      resetMessages();
      setApprovals([]);
    };
    window.addEventListener(GUEST_EXPIRED, onExpired);
    void Promise.resolve().then(() => {
      if (mounted.current) return startGuest();
    });
    return () => {
      window.removeEventListener(GUEST_EXPIRED, onExpired);
    };
  }, [startGuest]);

  const restartGuest = () => {
    if (activeRequest.current || guestRequest.current) return;
    guestRef.current = null;
    setGuest(null);
    conversationStarted.current = false;
    feedbackRequests.current.forEach((controller) => controller.abort());
    feedbackRequests.current.clear();
    historyRequests.current.forEach((controller) => controller.abort());
    resetMessages();
    setInput("");
    setApprovals([]);
    void startGuest();
  };

  // Check API status on component mount
  useEffect(() => {
    let cancelled = false;
    const checkApiStatus = async () => {
      try {
        await GuestApi.healthCheck();
        if (!cancelled) setApiStatus("online");
      } catch {
        if (!cancelled) setApiStatus("offline");
      }
    };

    checkApiStatus();
    const interval = setInterval(checkApiStatus, 30000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const processMessage = useCallback(
    async (userQuery: string, isVoice = false) => {
      // This ref closes the same-event gap before React applies disabled controls.
      const identity = guestRef.current;
      if (
        !userQuery.trim() ||
        !mounted.current ||
        activeRequest.current ||
        !identity
      )
        return;
      const scope = guestScope(identity);
      const request = {
        id: ++nextRequestId.current,
        controller: new AbortController(),
      };
      activeRequest.current = request;
      const assistantId = `reply-${request.id}`;
      const isCurrent = () =>
        mounted.current && activeRequest.current?.id === request.id;
      const updateReply = (patch: Partial<Message>) => {
        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantId ? { ...message, ...patch } : message,
          ),
        );
      };

      conversationStarted.current = true;
      setMessages((previous) => [
        ...previous,
        {
          id: `question-${request.id}`,
          role: "user",
          content: userQuery,
          timestamp: new Date().toISOString(),
          isVoice,
        },
        {
          id: assistantId,
          role: "assistant",
          content: "",
          status: "streaming",
          scope,
          timestamp: new Date().toISOString(),
        },
      ]);
      setIsLoading(true);
      setVoiceError(null);

      const startedAt = performance.now();
      let firstTokenAt: number | null = null;
      let chunkCount = 0;
      let eventCount = 0;
      let stagesDone = 0;
      const traceKeys = { n: 1 };
      setChainStages([]);
      setChainStats(EMPTY_STATS);

      try {
        let assistantResponse = "";
        let completion: ChatCompletion | undefined;
        const onTraceEvent = (event: TraceEvent) => {
          if (!isCurrent()) return;
          eventCount += 1;
          setChainStages((cards) => applyTraceEvent(cards, event, traceKeys));
          const statusKey = turnStatusKeyFor(event);
          if (statusKey) setTurnStatusKey(statusKey);
          if (event.status === "end" || event.status === "error")
            stagesDone += 1;
        };
        const onCitations = (citations: SourceCitation[]) => {
          if (isCurrent()) updateReply({ citations });
        };

        const onCompletion = (receipt: ChatCompletion) => {
          if (isCurrent()) {
            completion = receipt;
            if (receipt.approval) acceptApproval(receipt.approval);
            if (receipt.watch) acceptWatch(receipt.watch);
          }
        };

        for await (const chunk of GuestApi.streamChat(
          userQuery,
          scope,
          onTraceEvent,
          onCitations,
          request.controller.signal,
          onCompletion,
        )) {
          if (!isCurrent()) return;
          if (firstTokenAt == null) {
            firstTokenAt = performance.now();
            // Answer tokens are streaming: the status line's job is done.
            setTurnStatusKey(null);
            setChainStats((stats) => ({
              ...stats,
              ttftMs: firstTokenAt! - startedAt,
            }));
          }
          chunkCount += 1;
          assistantResponse += chunk;
          updateReply({ content: assistantResponse });
        }
        if (!isCurrent()) return;
        if (assistantResponse.trim()) {
          // The completion receipt is the server's proof that the turn was
          // persisted. Text without it stays readable but never reads as received.
          updateReply(
            completion
              ? { status: "completed", messageId: completion.message_id }
              : { status: "unknown", unconfirmed: true },
          );
        } else {
          updateReply({ status: "failed", failure: "invalid", citations: [] });
        }
      } catch (error) {
        if (isCurrent()) {
          if (failureReason(error) === "authentication")
            window.dispatchEvent(
              new CustomEvent(GUEST_EXPIRED, { detail: scope }),
            );
          else
            updateReply({
              status: "failed",
              failure: failureReason(error),
              citations: [],
            });
        }
      } finally {
        if (isCurrent()) {
          setIsLoading(false);
          setTurnStatusKey(null);
          setChainStats((stats) => ({
            ...stats,
            chunkCount,
            eventCount,
            stagesDone,
            totalMs: performance.now() - startedAt,
          }));
          setChainTurn((turn) => turn + 1);
          activeRequest.current = null;
        }
      }
    },
    [acceptApproval, acceptWatch],
  );

  // Approval cards reuse the ordinary chat channel: the buttons send a
  // pre-filled confirmation/cancellation phrase, and the agent's existing
  // stage-gated tools decide what actually executes. No new endpoint.
  const handleApprovalConfirm = useCallback(
    (approval: CommerceApproval) => {
      if (activeRequest.current) return;
      // Payment confirmation is terminal for the card: flip it to done on the
      // click itself so the button cannot be re-armed after the payment
      // completes (order cards advance separately when the payment card
      // arrives, via acceptApproval's advanced-status logic).
      if (approval.action === "confirm_payment") {
        setApprovals((previous) =>
          previous.map((entry) =>
            entry.key === approvalKey(approval) && entry.status === "active"
              ? { ...entry, status: "done" as ApprovalCardStatus }
              : entry,
          ),
        );
      }
      // The phrase quotes the card's own total, so the instruction the agent
      // sees is card-derived — never model-narrated text.
      const amount = `¥${(approval.total_minor / 100).toFixed(2)}`;
      const phrase =
        approval.action === "confirm_order"
          ? isEn
            ? `Confirm order: place the order exactly as shown on the confirmation card (total ${amount}).`
            : `确认下单：按确认卡上的商品和金额下单（合计 ${amount}）。`
          : isEn
            ? `Confirm payment: pay the order amount shown on the confirmation card (${amount}).`
            : `确认支付：支付确认卡上的订单金额（${amount}）。`;
      void processMessage(phrase);
    },
    [isEn, processMessage],
  );

  const handleApprovalCancel = useCallback(
    (approval: CommerceApproval) => {
      if (activeRequest.current) return;
      setApprovals((previous) =>
        previous.map((entry) =>
          entry.key === approvalKey(approval) && entry.status === "active"
            ? { ...entry, status: "cancelled" as ApprovalCardStatus }
            : entry,
        ),
      );
      const phrase = isEn
        ? "Cancel this purchase; do not place the order."
        : "取消本次购买，暂不下单。";
      void processMessage(phrase);
    },
    [isEn, processMessage],
  );

  // The watch card's order button reuses the ordinary chat channel too: the
  // phrase quotes the card's own SKU and live price, and the agent's
  // existing quote → confirm gates decide what actually executes.
  const handleWatchOrder = useCallback(
    (watch: CommerceWatch) => {
      if (activeRequest.current) return;
      setWatches((previous) =>
        previous.map((entry) =>
          entry.key === watchKey(watch) && entry.status === "active"
            ? { ...entry, status: "ordered" as WatchCardStatus }
            : entry,
        ),
      );
      const price = `¥${(watch.current_price_minor / 100).toFixed(2)}`;
      const phrase = isEn
        ? `Order from my price-watch card: ${watch.sku_id}${watch.sku_name ? ` (${watch.sku_name})` : ""} at the watched price ${price} — please prepare a quote for my confirmation.`
        : `按降价提醒下单：${watch.sku_name || watch.sku_id} [${watch.sku_id}]，当前价 ${price}，请先给我报价确认。`;
      void processMessage(phrase);
    },
    [isEn, processMessage],
  );

  // Dismissal is local-only: the server already recorded the ask as
  // delivered, so no turn (and no LLM call) is spent on declining it.
  const handleWatchDismiss = useCallback((watch: CommerceWatch) => {
    setWatches((previous) =>
      previous.map((entry) =>
        entry.key === watchKey(watch) && entry.status === "active"
          ? { ...entry, status: "dismissed" as WatchCardStatus }
          : entry,
      ),
    );
  }, []);

  const handleFeedback = useCallback(
    async (localId: string, messageId: number, rating: 1 | -1) => {
      const identity = guestRef.current;
      if (!mounted.current || !identity) return;
      const scope = guestScope(identity);
      const key = `${scope.session_key}:${messageId}`;
      if (feedbackRequests.current.has(key)) return;
      const controller = new AbortController();
      feedbackRequests.current.set(key, controller);
      const updateFeedback = (patch: Partial<Message>) => {
        if (
          mounted.current &&
          guestRef.current?.session.session_key === scope.session_key
        )
          setMessages((previous) =>
            previous.map((message) =>
              message.id === localId &&
              message.messageId === messageId &&
              message.scope?.session_key === scope.session_key
                ? { ...message, ...patch }
                : message,
            ),
          );
      };
      updateFeedback({ feedbackPending: true, feedbackFailure: undefined });
      try {
        await GuestApi.submitFeedback(
          messageId,
          rating,
          scope,
          controller.signal,
        );
        updateFeedback({ rating });
      } catch (error) {
        updateFeedback({ feedbackFailure: failureReason(error) });
      } finally {
        if (feedbackRequests.current.get(key) === controller)
          feedbackRequests.current.delete(key);
        updateFeedback({ feedbackPending: false });
      }
    },
    [],
  );

  const handleSend = () => {
    if (
      !input.trim() ||
      activeRequest.current ||
      !guestRef.current ||
      guestLoading
    )
      return;
    void processMessage(input);
    setInput("");
    setVoiceDraftNotice(false);
  };

  const handleStopReceiving = () => {
    activeRequest.current?.controller.abort();
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      handleSend();
    }
  };

  const handleVoiceTranscription = useCallback(
    (text: string) => {
      if (!mounted.current || !text.trim() || !guestRef.current) return;
      if (activeRequest.current) {
        setInput((previous) =>
          previous ? `${previous} ${text.trim()}` : text.trim(),
        );
        setVoiceDraftNotice(true);
        return;
      }
      void processMessage(text, true);
    },
    [processMessage],
  );

  const handleVoiceError = useCallback((message: string) => {
    if (mounted.current) setVoiceError(message);
  }, []);

  const getStatusIcon = () => {
    switch (apiStatus) {
      case "checking":
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case "online":
        return <Wifi className="h-4 w-4 text-green-500" />;
      case "offline":
        return <WifiOff className="h-4 w-4 text-red-500" />;
    }
  };

  const getStatusText = () => {
    const dict = isEn ? STATUS_TEXT_EN : STATUS_TEXT_ZH;
    return dict[apiStatus];
  };

  const voiceLabels = isEn ? VOICE_LABELS_EN : VOICE_LABELS_ZH;
  const statusBar = isEn ? STATUS_BAR_EN : STATUS_BAR_ZH;
  const placeholder = isEn ? PLACEHOLDER_EN : PLACEHOLDER_ZH;
  const voiceLanguage = isEn ? "en-US" : "zh-CN";

  return (
    <div
      className={
        showExecutionChain ? "grid gap-4 lg:grid-cols-[1.15fr_1fr]" : undefined
      }
    >
      <Card
        className={
          showExecutionChain
            ? "flex h-[699px] flex-col lg:h-[calc(100dvh-12rem)] lg:min-h-[699px]"
            : undefined
        }
      >
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>
              {isEn
                ? "Agent Store · AI Shop Assistant"
                : "Agent Store · AI 店员"}
            </CardTitle>
            <div className="flex items-center gap-4">
              {/* Voice mode toggle */}
              <div className="flex items-center gap-2 text-sm">
                <Button
                  variant={voiceMode === "manual" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setVoiceMode("manual")}
                  className="h-7"
                >
                  <Mic className="mr-1 h-3 w-3" />
                  {voiceLabels.manual}
                </Button>
                <Button
                  variant={voiceMode === "auto" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setVoiceMode("auto")}
                  className="h-7"
                >
                  <Radio className="mr-1 h-3 w-3" />
                  {voiceLabels.auto}
                </Button>
              </div>

              {/* API status */}
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                {getStatusIcon()}
                <span>{getStatusText()}</span>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent
          className={
            showExecutionChain ? "flex min-h-0 flex-1 flex-col" : undefined
          }
        >
          {guestLoading && (
            <p role="status" className="mb-3 text-sm">
              {isEn
                ? "Opening your visitor conversation…"
                : "正在打开您的访客会话…"}
            </p>
          )}
          {guestError && (
            <p role="alert" className="mb-3 text-sm text-red-600">
              {guestError === "expired"
                ? isEn
                  ? "Your visitor session expired. Start again; previous requests will not be resent."
                  : "访客会话已失效。请重新开始，原请求不会自动重发。"
                : isEn
                  ? "Your visitor conversation could not be verified. No shared conversation was used."
                  : "无法核实访客会话，未使用共享会话。"}
            </p>
          )}
          {!guest && !guestLoading && (
            <Button
              variant="outline"
              className="mb-3"
              disabled={isLoading}
              onClick={restartGuest}
            >
              {guest
                ? isEn
                  ? "Reopen visitor conversation"
                  : "重新打开访客会话"
                : isEn
                  ? "Start a new visitor session"
                  : "重新开始访客会话"}
            </Button>
          )}

          <div
            className={
              showExecutionChain
                ? "flex min-h-0 flex-1 flex-col rounded-lg border"
                : "flex h-[500px] flex-col rounded-lg border"
            }
          >
            {/* Messages */}
            <div
              ref={messagesScrollRef}
              onScroll={handleMessagesScroll}
              className="flex-1 overflow-y-auto p-4 space-y-4"
            >
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  data-message-id={message.id}
                  data-message-status={
                    message.role === "assistant" ? message.status : undefined
                  }
                  aria-busy={message.status === "streaming" || undefined}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {message.role === "assistant" && (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary">
                      <Bot className="h-4 w-4 text-primary-foreground" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-2 ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      {message.isVoice && message.role === "user" && (
                        <Mic className="mt-0.5 h-3 w-3 shrink-0 opacity-70" />
                      )}
                      {message.isWelcome ? (
                        <p className="whitespace-pre-wrap text-sm">
                          {isEn ? WELCOME_EN : WELCOME_ZH}
                        </p>
                      ) : message.role === "assistant" ? (
                        <RichText text={stripDataChannels(message.content)} />
                      ) : (
                        <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                      )}
                    </div>
                    {/* Guardrail blocked indicator */}
                    {message.role === "assistant" &&
                      message.guardrailBlocked && (
                        <div className="mt-1 flex items-center gap-1 text-xs text-orange-600">
                          <Shield className="h-3 w-3" />
                          <span>
                            {isEn
                              ? `Blocked by safety guardrail: ${message.guardrailViolations?.join(", ")}`
                              : `已被安全门禁拦截: ${message.guardrailViolations?.join(", ")}`}
                          </span>
                        </div>
                      )}
                    {/* Intent metadata */}
                    {message.role === "assistant" &&
                      message.intent &&
                      !message.guardrailBlocked && (
                        <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                          {message.intent !== "unknown" && (
                            <span className="rounded bg-muted px-1.5 py-0.5">
                              {message.intent}
                            </span>
                          )}
                        </div>
                      )}
                    {message.role === "assistant" &&
                      !message.guardrailBlocked && (
                        <SourceCitations
                          citations={message.citations ?? []}
                          isEn={isEn}
                        />
                      )}
                    {message.role === "assistant" &&
                      message.status === "streaming" && (
                        <p
                          role="status"
                          className="mt-2 text-xs text-muted-foreground"
                        >
                          {isEn ? "Receiving reply…" : "正在接收回复…"}
                        </p>
                      )}
                    {message.role === "assistant" &&
                      message.status === "completed" && (
                        <p
                          role="status"
                          className="mt-2 text-xs text-muted-foreground"
                        >
                          {isEn ? "Reply received" : "回复接收完成"}
                        </p>
                      )}
                    {message.role === "assistant" &&
                      message.status === "unknown" &&
                      !message.isWelcome &&
                      !message.unconfirmed && (
                        <p className="mt-2 text-xs text-muted-foreground">
                          {isEn
                            ? "History · completion status unknown"
                            : "历史记录 · 完成状态未知"}
                        </p>
                      )}
                    {message.role === "assistant" && message.unconfirmed && (
                      <p className="mt-2 text-xs text-muted-foreground">
                        {isEn
                          ? "Reply delivered, but the server did not confirm saving it. Verify the outcome before relying on it."
                          : "回复已送达，但服务端未确认保存结果。请先核实后再依赖该回复。"}
                      </p>
                    )}
                    {message.role === "assistant" &&
                      message.status === "failed" && (
                        <p
                          role="alert"
                          className="mt-2 text-xs text-red-600 dark:text-red-400"
                        >
                          {
                            FAILURE_TEXT[isEn ? "en" : "zh"][
                              message.failure ?? "unknown"
                            ]
                          }{" "}
                          {isEn
                            ? "Business processing status is unconfirmed. No automatic retry was made. Verify the status before resubmitting."
                            : "业务处理状态尚未确认，本次未自动重试。请先核实处理状态，再决定是否重新提交。"}
                        </p>
                      )}
                    {isFeedbackEligible(message) && (
                      <div className="mt-2">
                        {guest ? (
                          <div
                            role="group"
                            aria-label={
                              isEn ? "Rate this reply" : "评价这条回复"
                            }
                            aria-busy={Boolean(message.feedbackPending)}
                            className="flex items-center gap-1"
                          >
                            <button
                              type="button"
                              aria-label={isEn ? "Helpful" : "有帮助"}
                              title={isEn ? "Helpful" : "有帮助"}
                              aria-pressed={message.rating === 1}
                              disabled={
                                message.feedbackPending || message.rating === 1
                              }
                              onClick={() => {
                                void handleFeedback(
                                  message.id,
                                  message.messageId,
                                  1,
                                );
                              }}
                              className={`rounded p-1 ${message.rating === 1 ? "text-green-600" : "text-muted-foreground hover:text-green-600"} disabled:cursor-default`}
                            >
                              <ThumbsUp className="h-4 w-4" />
                            </button>
                            <button
                              type="button"
                              aria-label={isEn ? "Not helpful" : "没帮助"}
                              title={isEn ? "Not helpful" : "没帮助"}
                              aria-pressed={message.rating === -1}
                              disabled={
                                message.feedbackPending || message.rating === -1
                              }
                              onClick={() => {
                                void handleFeedback(
                                  message.id,
                                  message.messageId,
                                  -1,
                                );
                              }}
                              className={`rounded p-1 ${message.rating === -1 ? "text-red-600" : "text-muted-foreground hover:text-red-600"} disabled:cursor-default`}
                            >
                              <ThumbsDown className="h-4 w-4" />
                            </button>
                          </div>
                        ) : (
                          <p
                            role={
                              message.feedbackFailure === "authentication"
                                ? "alert"
                                : undefined
                            }
                            className="text-xs text-muted-foreground"
                          >
                            {
                              FEEDBACK_FAILURE_TEXT[isEn ? "en" : "zh"]
                                .authentication
                            }
                          </p>
                        )}
                        {message.feedbackPending && (
                          <p
                            role="status"
                            className="text-xs text-muted-foreground"
                          >
                            {isEn ? "Saving feedback…" : "正在保存评价…"}
                          </p>
                        )}
                        {message.feedbackFailure &&
                          !(
                            message.feedbackFailure === "authentication" &&
                            !guest
                          ) && (
                            <p
                              role="alert"
                              className="text-xs text-red-600 dark:text-red-400"
                            >
                              {
                                FEEDBACK_FAILURE_TEXT[isEn ? "en" : "zh"][
                                  message.feedbackFailure
                                ]
                              }
                            </p>
                          )}
                      </div>
                    )}
                    {message.timestamp && (
                      <p className="mt-1 text-xs opacity-70">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </p>
                    )}
                  </div>
                  {message.role === "user" && !message.isVoice && (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary">
                      <User className="h-4 w-4 text-primary-foreground" />
                    </div>
                  )}
                </motion.div>
              ))}
              {approvals.length > 0 && (
                <div data-approval-section className="flex justify-start">
                  <div className="space-y-2">
                    {approvals.map((entry) => (
                      <ApprovalCard
                        key={entry.key}
                        approval={entry.data}
                        status={entry.status}
                        busy={isLoading}
                        now={approvalNow}
                        isEn={isEn}
                        onConfirm={handleApprovalConfirm}
                        onCancel={handleApprovalCancel}
                      />
                    ))}
                  </div>
                </div>
              )}
              {watches.length > 0 && (
                <div data-watch-section className="flex justify-start">
                  <div className="space-y-2">
                    {watches.map((entry) => (
                      <WatchCard
                        key={entry.key}
                        watch={entry.data}
                        status={entry.status}
                        busy={isLoading}
                        isEn={isEn}
                        onOrder={handleWatchOrder}
                        onDismiss={handleWatchDismiss}
                      />
                    ))}
                  </div>
                </div>
              )}
              {isLoading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex gap-3"
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary">
                    <Bot className="h-4 w-4 text-primary-foreground" />
                  </div>
                  <div className="flex items-center gap-2 rounded-lg bg-muted px-4 py-2">
                    {turnStatusKey ? (
                      <span
                        data-testid="turn-status"
                        className="text-sm text-muted-foreground"
                      >
                        {turnStatusPhrase(turnStatusKey, isEn)}
                        <span className="animate-pulse">…</span>
                      </span>
                    ) : (
                      <>
                        <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground" />
                        <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground delay-100" />
                        <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground delay-200" />
                      </>
                    )}
                  </div>
                </motion.div>
              )}
            </div>

            {/* Input */}
            <div className="border-t p-4">
              <div className="flex items-center gap-2">
                {/* Voice Button */}
                <VoiceButton
                  mode={voiceMode}
                  sessionId={guest?.session.id}
                  stream={true}
                  language={voiceLanguage}
                  disabled={isLoading || guestLoading || !guest}
                  onTranscription={handleVoiceTranscription}
                  onError={handleVoiceError}
                  className="shrink-0"
                />

                {/* Text Input */}
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  aria-label={isEn ? "Message" : "消息"}
                  placeholder={
                    apiStatus === "online"
                      ? placeholder.online
                      : placeholder.offline
                  }
                  className="flex-1 rounded-md border bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                />

                {/* Send Button */}
                <Button
                  onClick={handleSend}
                  disabled={
                    isLoading || guestLoading || !guest || !input.trim()
                  }
                  aria-label={isEn ? "Send message" : "发送消息"}
                  size="icon"
                >
                  <Send className="h-4 w-4" />
                </Button>
                {isLoading && (
                  <Button
                    onClick={handleStopReceiving}
                    variant="outline"
                    size="sm"
                    aria-label={isEn ? "Stop receiving" : "停止接收"}
                  >
                    {isEn ? "Stop receiving" : "停止接收"}
                  </Button>
                )}
              </div>

              {voiceDraftNotice && (
                <p role="status" className="mt-2 text-xs text-muted-foreground">
                  {isLoading
                    ? isEn
                      ? "A reply is still arriving. Your speech was saved as a draft."
                      : "上条回复接收中，语音已保留为草稿。"
                    : isEn
                      ? "Your speech was saved as a draft. Review it and send when ready."
                      : "语音已保留为草稿，请确认后手动发送。"}
                </p>
              )}
              {voiceError && (
                <p role="alert" className="mt-2 text-xs text-red-600">
                  {isEn ? "Voice error: " : "语音错误："}
                  {voiceError}
                </p>
              )}

              {/* Status bar */}
              <div className="mt-3 flex items-center justify-between">
                <p className="text-xs text-muted-foreground">
                  {voiceMode === "manual" ? statusBar.manual : statusBar.auto}
                </p>
                {apiStatus === "offline" && (
                  <p className="text-xs text-orange-600">
                    <strong>{isEn ? "API unavailable" : "API 不可用"}</strong> -{" "}
                    {statusBar.offline.replace(/^.*- /, "")}
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Connection notice is independent of per-request failures. */}
          {apiStatus === "offline" && (
            <div className="mt-4 rounded-lg border border-orange-200 bg-orange-50 p-3">
              <p className="text-xs text-orange-800">
                <strong>{isEn ? "Note:" : "注意:"}</strong>{" "}
                {statusBar.notice.replace(/^注意: |^Note: /, "")}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Live execution-chain panel (demo cockpit mode only) */}
      {showExecutionChain && (
        <ExecutionChainPanel
          stages={chainStages}
          stats={chainStats}
          turn={chainTurn}
        />
      )}
    </div>
  );
}
