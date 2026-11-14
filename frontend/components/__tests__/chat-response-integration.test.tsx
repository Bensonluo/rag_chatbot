/** Real React -> ApiClient -> SSE parser, with synthetic fetch responses only.
 * No live server, browser speech service, or ApiClient method is replaced.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { ReadableStream as NativeReadableStream } from "node:stream/web";
import { TextDecoder, TextEncoder } from "node:util";
import { ChatInterfaceUpdated } from "@/components/ChatInterfaceUpdated";
import { LanguageProvider } from "@/lib/i18n/context";

Object.assign(globalThis, {
  IS_REACT_ACT_ENVIRONMENT: true,
  TextDecoder,
  TextEncoder,
});

interface ControlledStream {
  response: Response;
  send: (text: string) => void;
  sendBytes: (bytes: Uint8Array) => void;
  close: () => void;
  fail: (error: Error) => void;
  cancel: jest.Mock;
}

function controlledStream(): ControlledStream {
  let controller: ReadableStreamDefaultController<Uint8Array>;
  let closed = false;
  const cancel = jest.fn(() => {
    closed = true;
  });
  const body = new NativeReadableStream<Uint8Array>({
    start(value) {
      controller = value;
    },
    cancel,
  });
  return {
    response: {
      ok: true,
      status: 200,
      statusText: "OK",
      headers: {
        get: (name: string) =>
          name.toLowerCase() === "content-type"
            ? "text/event-stream; charset=utf-8"
            : null,
      },
      body,
    } as unknown as Response,
    send(text) {
      if (!closed) controller.enqueue(new TextEncoder().encode(text));
    },
    sendBytes(bytes) {
      if (!closed) controller.enqueue(bytes);
    },
    close() {
      if (!closed) {
        closed = true;
        controller.close();
      }
    },
    fail(error) {
      if (!closed) {
        closed = true;
        controller.error(error);
      }
    },
    cancel,
  };
}

function jsonResponse(value: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 401 ? "Unauthorized" : "Request failed",
    json: async () => value,
    text: async () => JSON.stringify(value),
  } as Response;
}

let container: HTMLDivElement;
let root: Root | undefined;
let fetchMock: jest.Mock;
let chatReply: (init?: RequestInit) => Response | Promise<Response>;
let streams: ControlledStream[];
const originalFetch = globalThis.fetch;

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem("portfolio-locale", "en");
  sessionStorage.clear();
  streams = [];
  container = document.createElement("div");
  document.body.appendChild(container);
  chatReply = () => {
    throw new Error("Each test must provide its chat transport");
  };
  fetchMock = jest.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input), "http://localhost").pathname;
    if (path === "/health") return jsonResponse({ status: "healthy" });
    if (path === "/api/v1/guest/bootstrap")
      return jsonResponse({
        visitor_id: "22222222-2222-4222-8222-222222222222",
        expires_at: "2099-01-01T00:00:00Z",
        session: {
          id: 317,
          session_key: "11111111-1111-4111-8111-111111111111",
        },
      });
    if (path === "/api/v1/guest/history")
      return jsonResponse({
        session_id: 317,
        session_key: "11111111-1111-4111-8111-111111111111",
        messages: [],
      });
    if (path === "/api/v1/guest/chat/stream") return chatReply(init);
    if (path === "/api/v1/guest/feedback")
      return jsonResponse({ success: true });
    throw new Error(`Unexpected request: ${path}`);
  });
  globalThis.fetch = fetchMock;
  jest.spyOn(console, "warn").mockImplementation(() => {});
  jest.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(async () => {
  await act(async () => {
    root?.unmount();
    root = undefined;
  });
  await act(async () => streams.forEach((stream) => stream.close()));
  container.remove();
  globalThis.fetch = originalFetch;
  Reflect.deleteProperty(window, "SpeechRecognition");
  Reflect.deleteProperty(window, "webkitSpeechRecognition");
});

function streamReply(): ControlledStream {
  const stream = controlledStream();
  streams.push(stream);
  chatReply = () => stream.response;
  return stream;
}

function chatRequests(): RequestInit[] {
  return fetchMock.mock.calls
    .filter(
      ([url]) =>
        new URL(String(url), "http://localhost").pathname ===
        "/api/v1/guest/chat/stream",
    )
    .map(([, init]) => init as RequestInit);
}

async function mount() {
  await act(async () => {
    root = createRoot(container);
    root.render(
      <LanguageProvider>
        <ChatInterfaceUpdated />
      </LanguageProvider>,
    );
  });
}

function input(): HTMLInputElement {
  const field = container.querySelector<HTMLInputElement>('input[type="text"]');
  if (!field) throw new Error("Chat input missing");
  return field;
}

function sendButton(): HTMLButtonElement {
  const button =
    container.querySelector<HTMLButtonElement>(
      'button[aria-label="Send message"]',
    ) ?? (input().nextElementSibling as HTMLButtonElement | null);
  if (!button) throw new Error("Send button missing");
  return button;
}

async function type(message: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )?.set?.call(input(), message);
    input().dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function send(message: string) {
  await type(message);
  await act(async () => sendButton().click());
}

function feedbackButtons(): HTMLButtonElement[] {
  return Array.from(
    container.querySelectorAll<HTMLButtonElement>(
      'button[title="Helpful"], button[title="Not helpful"]',
    ),
  );
}

function failure(): Element | null {
  return container.querySelector('[role="alert"]');
}

function completedReplies(): Element[] {
  return Array.from(container.querySelectorAll('[role="status"]')).filter(
    (node) => node.textContent === "Reply received",
  );
}

function paragraphs(): string[] {
  return Array.from(container.querySelectorAll("p.whitespace-pre-wrap")).map(
    (node) => node.textContent ?? "",
  );
}

let completionMessageId = 500;

function completionFrame(): string {
  completionMessageId += 1;
  return (
    "event: completion\ndata: " +
    JSON.stringify({
      message_id: completionMessageId,
      session_id: 317,
      session_key: "11111111-1111-4111-8111-111111111111",
      status: "completed",
    }) +
    "\n\n"
  );
}

const citationFrame =
  "event: citations\ndata: " +
  JSON.stringify({
    citations: [
      {
        source_type: "document",
        document_id: "policy-1",
        version_id: "version-1",
        title: "Original policy",
        excerpt: "REFERENCE EXCERPT",
        section_path: [],
        source_document_ids: [],
        source_version_ids: [],
        source_chunk_ids: [],
      },
    ],
  }) +
  "\n\n";

test.each([401, 429, 503])(
  "healthy endpoint does not turn chat HTTP %i into a simulated answer",
  async (status) => {
    chatReply = () => jsonResponse({ detail: "Chat request rejected" }, status);
    await mount();
    expect(container.textContent).toContain("Connected to Agent Store");
    await send("I want a refund");
    expect(chatRequests()).toHaveLength(1);
    expect(failure()?.textContent).toBeTruthy();
    expect(feedbackButtons()).toHaveLength(0);
    // Also catch delayed fallback timers from the original implementation.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 550));
    });
    expect(paragraphs()).not.toContain(
      "Sure, I will help you with a refund.\n\nPlease provide your order ID.",
    );
    expect(failure()?.textContent).toBeTruthy();
  },
);

test("server stream error preserves real partial text and never exposes its citations or feedback", async () => {
  const stream = streamReply();
  await mount();
  await send("What happened to my request?");
  await act(async () =>
    stream.send("data: REAL PARTIAL ANSWER\n\n" + citationFrame),
  );
  expect(paragraphs()).toContain("REAL PARTIAL ANSWER");
  expect(feedbackButtons()).toHaveLength(0);
  await act(async () => {
    stream.send("data: [STREAM_ERROR]\n\ndata: [DONE]\n\n");
    stream.close();
  });
  expect(paragraphs()).toContain("REAL PARTIAL ANSWER");
  expect(failure()?.textContent).toBeTruthy();
  expect(container.textContent).not.toContain("REFERENCE EXCERPT");
  expect(feedbackButtons()).toHaveLength(0);
});

test.each(["data: TRUNCATED ANSWER\n\n", "data: TRUNCATED ANSWER"])(
  "EOF without DONE fails even when text was received (%s)",
  async (wire) => {
    const stream = streamReply();
    await mount();
    await send("Tell me the policy");
    await act(async () => {
      stream.send(wire);
      stream.close();
    });
    expect(failure()?.textContent).toBeTruthy();
    if (wire.endsWith("\n\n"))
      expect(paragraphs()).toContain("TRUNCATED ANSWER");
    expect(feedbackButtons()).toHaveLength(0);
  },
);

test("DONE completes fragmented UTF-8 content but cannot invent a server feedback ID", async () => {
  const stream = streamReply();
  await mount();
  await send("Show the return policy");
  const utf8 = new TextEncoder().encode("data: 有依据的答复\n\n");
  await act(async () => {
    stream.sendBytes(utf8.slice(0, 7)); // Inside the first Chinese code point.
    stream.sendBytes(utf8.slice(7));
    stream.send(citationFrame + completionFrame() + "data: [DO");
  });
  expect(paragraphs()).toContain("有依据的答复");
  expect(completedReplies()).toHaveLength(0);
  expect(container.textContent).toContain("Receiving reply");
  expect(feedbackButtons()).toHaveLength(0);
  await act(async () => {
    stream.send("NE]\n\n");
    stream.close();
  });
  expect(failure()).toBeNull();
  expect(paragraphs()).toContain("有依据的答复");
  expect(container.textContent).toContain("REFERENCE EXCERPT");
  expect(completedReplies()).toHaveLength(1);
  // A valid server receipt makes the persisted reply feedback-eligible.
  expect(feedbackButtons()).toHaveLength(2);
});

test("synchronous Enter repeats are one request; the next completed turn owns its own answer", async () => {
  const first = streamReply();
  await mount();
  await type("FIRST QUESTION");
  await act(async () => {
    for (let n = 0; n < 2; n += 1)
      input().dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
  });
  expect(chatRequests()).toHaveLength(1);
  await type("SECOND QUESTION");
  await act(async () =>
    input().dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    ),
  );
  expect(chatRequests()).toHaveLength(1);
  expect(input().value).toBe("SECOND QUESTION");
  await act(async () => first.send("data: FIRST-A\n\n"));
  expect(paragraphs()).toContain("FIRST-A");
  await act(async () => {
    first.send("data: FIRST-C\n\n" + completionFrame() + "data: [DONE]\n\n");
    first.close();
  });
  const second = streamReply();
  await act(async () =>
    input().dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    ),
  );
  expect(chatRequests()).toHaveLength(2);
  await act(async () => {
    second.send("data: SECOND-B\n\n" + completionFrame() + "data: [DONE]\n\n");
    second.close();
  });
  expect(paragraphs().slice(-4)).toEqual([
    "FIRST QUESTION",
    "FIRST-AFIRST-C",
    "SECOND QUESTION",
    "SECOND-B",
  ]);
  expect(
    chatRequests().map((request) => JSON.parse(String(request.body)).message),
  ).toEqual(["FIRST QUESTION", "SECOND QUESTION"]);
  expect(completedReplies()).toHaveLength(2);
  expect(feedbackButtons()).toHaveLength(4);
});

test("a failed turn keeps its partial answer when the next independent request completes", async () => {
  const first = streamReply();
  await mount();
  await send("FIRST FAILED QUESTION");
  await act(async () => {
    first.send("data: FIRST PARTIAL\n\ndata: [STREAM_ERROR]\n\n");
    first.close();
  });
  expect(failure()?.textContent).toBeTruthy();
  const second = streamReply();
  await send("SECOND RETRY QUESTION");
  await act(async () => {
    second.send(
      "data: SECOND COMPLETE\n\n" + completionFrame() + "data: [DONE]\n\n",
    );
    second.close();
  });
  expect(chatRequests()).toHaveLength(2);
  expect(paragraphs().slice(-4)).toEqual([
    "FIRST FAILED QUESTION",
    "FIRST PARTIAL",
    "SECOND RETRY QUESTION",
    "SECOND COMPLETE",
  ]);
  expect(container.querySelectorAll('[role="alert"]')).toHaveLength(1);
  expect(completedReplies()).toHaveLength(1);
  expect(feedbackButtons()).toHaveLength(2);
});

class LocalRecognition {
  static instances: LocalRecognition[] = [];
  onstart: (() => void) | null = null;
  onend: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onresult:
    | ((event: {
        resultIndex: number;
        results: { isFinal: boolean; 0: { transcript: string } }[];
      }) => void)
    | null = null;
  start() {
    this.onstart?.();
  }
  stop() {}
  abort() {}
  constructor() {
    LocalRecognition.instances.push(this);
  }
}

test("a voice result queued with text submission cannot start a competing stream", async () => {
  LocalRecognition.instances = [];
  Object.defineProperty(window, "SpeechRecognition", {
    configurable: true,
    value: LocalRecognition,
  });
  const stream = streamReply();
  await mount();
  const record = container.querySelector<HTMLButtonElement>(
    'button[title="Click to start recording"]',
  );
  expect(record?.disabled).toBe(false);
  await act(async () => record?.click());
  await act(async () =>
    LocalRecognition.instances[0].onresult?.({
      resultIndex: 0,
      results: [
        { isFinal: true, 0: { transcript: "COMPETING VOICE QUESTION" } },
      ],
    }),
  );
  await type("TEXT QUESTION");
  const stop = container.querySelector<HTMLButtonElement>(
    'button[title="Click to stop"]',
  );
  await act(async () => {
    sendButton().click();
    stop?.click();
  });
  expect(chatRequests()).toHaveLength(1);
  expect(JSON.parse(String(chatRequests()[0].body)).message).toBe(
    "TEXT QUESTION",
  );
  expect(paragraphs()).not.toContain("COMPETING VOICE QUESTION");
  await act(async () => {
    stream.send("data: TEXT ANSWER\n\ndata: [DONE]\n\n");
    stream.close();
  });
  expect(paragraphs().slice(-2)).toEqual(["TEXT QUESTION", "TEXT ANSWER"]);
});

test("unmount aborts the actual in-flight ApiClient fetch", async () => {
  const stream = streamReply();
  let signal: AbortSignal | null | undefined;
  chatReply = (init) => {
    signal = init?.signal;
    signal?.addEventListener(
      "abort",
      () => stream.fail(new DOMException("View unmounted", "AbortError")),
      { once: true },
    );
    return stream.response;
  };
  await mount();
  await send("Pending request");
  await act(async () => stream.send("data: BEFORE UNMOUNT\n\n"));
  expect(signal?.aborted).toBe(false);
  await act(async () => {
    root?.unmount();
    root = undefined;
  });
  expect(signal?.aborted).toBe(true);
  expect(container.textContent).toBe("");
});

test.each([false, true])(
  "Stop receiving aborts the real stream, preserves partial=%s, and permits only a manual next request",
  async (withPartial) => {
    const first = streamReply();
    let signal: AbortSignal | null | undefined;
    const privateTransportDetail = "upstream-private-session-detail";
    chatReply = (init) => {
      signal = init?.signal;
      signal?.addEventListener(
        "abort",
        () => {
          first.fail(new DOMException(privateTransportDetail, "AbortError"));
        },
        { once: true },
      );
      return first.response;
    };
    await mount();
    await send("STOPPABLE QUESTION");
    if (withPartial)
      await act(async () => first.send("data: PRESERVED PARTIAL\n\n"));
    const stop = Array.from(
      container.querySelectorAll<HTMLButtonElement>("button"),
    ).find(
      (button) =>
        button.getAttribute("aria-label") === "Stop receiving" ||
        button.textContent === "Stop receiving",
    );
    expect(stop).toBeDefined();
    expect(signal?.aborted).toBe(false);
    await act(async () => stop?.click());
    expect(signal?.aborted).toBe(true);
    expect(failure()?.textContent).toMatch(/stop/i);
    expect(failure()?.textContent).toMatch(/receiv|reception/i);
    expect(container.textContent).not.toContain(privateTransportDetail);
    expect(container.textContent).not.toContain("The request was cancelled.");
    if (withPartial) expect(paragraphs()).toContain("PRESERVED PARTIAL");
    expect(completedReplies()).toHaveLength(0);
    expect(feedbackButtons()).toHaveLength(0);
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 550));
    });
    expect(chatRequests()).toHaveLength(1);

    const second = streamReply();
    await type("MANUAL NEXT QUESTION");
    expect(sendButton().disabled).toBe(false);
    expect(chatRequests()).toHaveLength(1);
    await act(async () => sendButton().click());
    expect(chatRequests()).toHaveLength(2);
    await act(async () => {
      second.send(
        "data: INDEPENDENT NEXT ANSWER\n\n" +
          completionFrame() +
          "data: [DONE]\n\n",
      );
      second.close();
    });
    expect(paragraphs().slice(-2)).toEqual([
      "MANUAL NEXT QUESTION",
      "INDEPENDENT NEXT ANSWER",
    ]);
    if (withPartial) expect(paragraphs()).toContain("PRESERVED PARTIAL");
    expect(container.querySelectorAll('[role="alert"]')).toHaveLength(1);
    expect(completedReplies()).toHaveLength(1);
    expect(feedbackButtons()).toHaveLength(2);
  },
);
