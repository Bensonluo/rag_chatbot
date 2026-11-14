import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { ChatInterfaceUpdated } from "@/components/ChatInterfaceUpdated";
import { LanguageProvider } from "@/lib/i18n/context";
import { type SourceCitation, type TraceEvent } from "@/lib/api";
import { GuestApi as ApiClient } from "@/lib/guest-api";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

let mockVoiceProps: {
  disabled?: boolean;
  onTranscription?: (text: string) => void;
};
jest.mock("@/components/VoiceButton", () => ({
  VoiceButton: (props: typeof mockVoiceProps) => {
    mockVoiceProps = props;
    return <button disabled={props.disabled} aria-label="Test microphone" />;
  },
}));

let container: HTMLDivElement;
let root: Root | undefined;

beforeEach(() => {
  localStorage.clear();
  container = document.createElement("div");
  document.body.appendChild(container);
  jest.spyOn(ApiClient, "healthCheck").mockResolvedValue({ status: "healthy" });
  jest
    .spyOn(ApiClient, "bootstrap")
    .mockResolvedValue({
      visitor_id: "22222222-2222-4222-8222-222222222222",
      expires_at: "2099-01-01T00:00:00Z",
      session: { id: 317, session_key: "11111111-1111-4111-8111-111111111111" },
    });
  jest
    .spyOn(ApiClient, "getChatHistory")
    .mockResolvedValue({
      session_id: 317,
      session_key: "11111111-1111-4111-8111-111111111111",
      messages: [],
    });
});

afterEach(async () => {
  await act(async () => root?.unmount());
  root = undefined;
  container.remove();
});

async function mount(locale: "zh" | "en" = "zh", chain = false) {
  localStorage.setItem("portfolio-locale", locale);
  await act(async () => {
    root = createRoot(container);
    root.render(
      <LanguageProvider>
        <ChatInterfaceUpdated showExecutionChain={chain} />
      </LanguageProvider>,
    );
  });
}

async function draft(text: string) {
  const input =
    container.querySelector<HTMLInputElement>('input[type="text"]')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )?.set?.call(input, text);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  return input;
}

async function send(text: string) {
  const input = await draft(text);
  await act(async () =>
    (input.nextElementSibling as HTMLButtonElement).click(),
  );
}

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function failure(kind: string, status?: number) {
  return Object.assign(new Error("SECRET_RAW_SERVER_ERROR"), { kind, status });
}

const citation: SourceCitation = {
  source_type: "document",
  title: "Current reference",
  excerpt: "Policy excerpt",
  document_id: "policy",
  version_id: "v1",
  chunk_id: "c1",
  section_path: [],
  source_document_ids: [],
  source_version_ids: [],
  source_chunk_ids: [],
};

test.each([
  ["zh", "http", 401, "访客会话已失效"],
  ["en", "http", 401, "visitor session expired"],
  ["zh", "http", 429, "请求过于频繁"],
  ["en", "http", 429, "Too many requests"],
  ["zh", "server_error", undefined, "服务未能完成回复"],
  ["en", "server_error", undefined, "The service could not finish the reply"],
] as const)(
  "%s %s/%s retains partial text and reports this turn failed without a simulated answer",
  async (locale, kind, status, label) => {
    jest
      .spyOn(ApiClient, "streamChat")
      .mockImplementation(
        async function* (_message, _session, _trace, onCitations) {
          onCitations?.([citation]);
          yield "真实部分 Partial reply";
          throw failure(kind, status);
        },
      );
    await mount(locale);
    await send("我要退款");
    if (status === 401) {
      expect(container.textContent).not.toContain("真实部分 Partial reply");
      expect(
        container.querySelector('[data-message-status="failed"]'),
      ).toBeNull();
    } else {
      expect(container.textContent).toContain("真实部分 Partial reply");
      expect(
        container.querySelector('[data-message-status="failed"]'),
      ).not.toBeNull();
    }
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      label,
    );
    expect(container.textContent).not.toContain("好的，我来帮你处理退款");
    expect(container.textContent).not.toContain("SECRET_RAW_SERVER_ERROR");
    expect(container.textContent).not.toContain("Current reference");
    expect(
      container.querySelector(
        'button[title="有帮助"], button[title="Helpful"]',
      ),
    ).toBeNull();
    expect(container.textContent).toContain(
      locale === "zh" ? "已连接 Agent Store" : "Connected to Agent Store",
    );
  },
);

test("a reply without a server receipt stays unconfirmed and never invents a message ID for feedback", async () => {
  const done = deferred();
  jest.spyOn(ApiClient, "streamChat").mockImplementation(async function* () {
    yield "Received partial";
    await done.promise;
    yield " complete";
  });
  await mount();
  expect(container.querySelector('button[title="有帮助"]')).toBeNull();
  await send("question");
  expect(
    container.querySelector('[data-message-status="streaming"]'),
  ).not.toBeNull();
  expect(container.querySelector('button[title="有帮助"]')).toBeNull();
  await act(async () => done.resolve());
  expect(
    container.querySelector(
      '[data-message-status="unknown"]:not([data-message-id="welcome"])',
    ),
  ).not.toBeNull();
  expect(container.textContent).toContain("Received partial complete");
  expect(container.textContent).not.toContain("回复接收完成");
  expect(container.textContent).toContain("未确认保存结果");
  expect(container.querySelector('button[title="有帮助"]')).toBeNull();
});

test("busy click, Enter, and queued speech cannot submit or erase a draft", async () => {
  const done = deferred();
  const stream = jest
    .spyOn(ApiClient, "streamChat")
    .mockImplementation(async function* () {
      yield "First answer";
      await done.promise;
    });
  await mount();
  await send("first question");
  const firstId = container
    .querySelector('[data-message-status="streaming"]')
    ?.getAttribute("data-message-id");
  const input = await draft("keep this draft");
  await act(async () => {
    (input.nextElementSibling as HTMLButtonElement).click();
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    mockVoiceProps.onTranscription?.("queued second speech");
  });
  expect(stream).toHaveBeenCalledTimes(1);
  expect(input.value).toBe("keep this draft queued second speech");
  expect(mockVoiceProps.disabled).toBe(true);
  await act(async () => done.resolve());
  expect(
    container
      .querySelector(`[data-message-id="${firstId}"]`)
      ?.getAttribute("data-message-status"),
  ).toBe("unknown");
  await act(async () =>
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    ),
  );
  expect(stream).toHaveBeenCalledTimes(2);
});

test("same-event speech callbacks share the immediate submission lock", async () => {
  const done = deferred();
  const stream = jest
    .spyOn(ApiClient, "streamChat")
    .mockImplementation(async function* () {
      await done.promise;
      yield "One answer";
    });
  await mount();
  await act(async () => {
    mockVoiceProps.onTranscription?.("first speech");
    mockVoiceProps.onTranscription?.("second speech");
  });
  expect(stream).toHaveBeenCalledTimes(1);
  expect(container.textContent).not.toContain("second speech");
  await act(async () => done.resolve());
});

test("callbacks retained from a completed request cannot alter the next reply or traces", async () => {
  let oldCitations: ((value: SourceCitation[]) => void) | undefined;
  let oldTrace: ((event: TraceEvent) => void) | undefined;
  const secondDone = deferred();
  jest
    .spyOn(ApiClient, "streamChat")
    .mockImplementationOnce(
      async function* (_message, _session, trace, citations) {
        oldTrace = trace;
        oldCitations = citations;
        yield "First answer";
      },
    )
    .mockImplementationOnce(async function* () {
      yield "Second partial";
      await secondDone.promise;
    });
  await mount("zh", true);
  await send("first");
  await send("second");
  await act(async () => {
    oldCitations?.([{ ...citation, title: "STALE CITATION" }]);
    oldTrace?.({
      stage: "cs.handoff",
      status: "detail",
      detail: { ticket_id: 999 },
    });
  });
  expect(container.textContent).not.toContain("STALE CITATION");
  expect(container.textContent).not.toContain("999");
  expect(container.textContent).toContain("Second partial");
  await act(async () => secondDone.resolve());
});

test("unmount aborts the active request and ignores its late callbacks and completion", async () => {
  let signal: AbortSignal | undefined;
  let oldCitations: ((value: SourceCitation[]) => void) | undefined;
  const done = deferred();
  jest
    .spyOn(ApiClient, "streamChat")
    .mockImplementation(
      async function* (_message, _session, _trace, citations, requestSignal) {
        signal = requestSignal;
        oldCitations = citations;
        yield "Partial before navigation";
        await done.promise;
        yield " stale completion";
      },
    );
  await mount();
  await send("question");
  expect(signal?.aborted).toBe(false);
  await act(async () => {
    root?.unmount();
    root = undefined;
  });
  expect(signal?.aborted).toBe(true);
  await mount();
  await act(async () => {
    oldCitations?.([citation]);
    done.resolve();
  });
  expect(container.textContent).not.toContain("stale completion");
  expect(container.textContent).not.toContain("Current reference");
});

test("a failed health probe never enables a simulated business response", async () => {
  jest
    .mocked(ApiClient.healthCheck)
    .mockRejectedValue(new Error("health unavailable"));
  jest.spyOn(ApiClient, "streamChat").mockImplementation(async function* () {
    throw failure("network");
  });
  await mount();
  await send("我要退款");
  expect(container.querySelector('[role="alert"]')?.textContent).toContain(
    "网络连接中断",
  );
  expect(container.textContent).not.toContain("演示响应");
  expect(container.textContent).not.toContain("好的，我来帮你处理退款");
});

test.each([
  ["http", 403, "无权访问"],
  ["http", 404, "会话不可用"],
  ["network", undefined, "网络连接中断"],
  ["interrupted", undefined, "回复在完成前中断"],
  ["invalid_response", undefined, "未返回有效回复"],
  ["aborted", undefined, "已停止接收回复。"],
  ["unexpected", undefined, "无法确认处理状态"],
] as const)(
  "%s/%s reports a fixed safe reason and never retries the business action",
  async (kind, status, reason) => {
    const stream = jest
      .spyOn(ApiClient, "streamChat")
      .mockImplementation(async function* () {
        throw failure(kind, status);
      });
    await mount();
    await send("确认退款");
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      reason,
    );
    expect(container.textContent).not.toContain("SECRET_RAW_SERVER_ERROR");
    expect(stream).toHaveBeenCalledTimes(1);
  },
);

test.each(["", "   "])(
  "an empty completed transport is a failed reply rather than business success",
  async (content) => {
    jest.spyOn(ApiClient, "streamChat").mockImplementation(async function* () {
      yield content;
    });
    await mount();
    await send("question");
    expect(
      container.querySelector('[data-message-status="failed"]'),
    ).not.toBeNull();
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      "未返回有效回复",
    );
    expect(container.textContent).not.toContain("回复接收完成");
  },
);

test("legacy history has unknown completion and no fabricated feedback identity", async () => {
  jest
    .mocked(ApiClient.getChatHistory)
    .mockResolvedValue({
      session_id: 317,
      session_key: "11111111-1111-4111-8111-111111111111",
      messages: [{ role: "assistant", content: "Historical partial" }],
    });
  await mount();
  expect(
    container.querySelector('[data-message-status="unknown"]'),
  ).not.toBeNull();
  expect(container.textContent).toContain("完成状态未知");
  expect(container.querySelector('button[title="有帮助"]')).toBeNull();
});

test.each([
  ["zh", "", "上条回复接收中，语音已保留为草稿"],
  ["zh", "Typed draft", "上条回复接收中，语音已保留为草稿"],
  ["en", "", "A reply is still arriving. Your speech was saved as a draft."],
  [
    "en",
    "Typed draft",
    "A reply is still arriving. Your speech was saved as a draft.",
  ],
] as const)(
  "%s busy speech preserves %s and waits for an explicit send",
  async (locale, existingDraft, notice) => {
    const done = deferred();
    const stream = jest
      .spyOn(ApiClient, "streamChat")
      .mockImplementationOnce(async function* () {
        yield "First partial";
        await done.promise;
      })
      .mockImplementation(async function* () {
        yield "Manually requested reply";
      });
    await mount(locale);
    await send("first question");
    const input = await draft(existingDraft);
    await act(async () =>
      mockVoiceProps.onTranscription?.("Recognized speech"),
    );
    expect(input.value).toBe(
      existingDraft
        ? `${existingDraft} Recognized speech`
        : "Recognized speech",
    );
    expect(
      [...container.querySelectorAll('[role="status"]')].some((element) =>
        element.textContent?.includes(notice),
      ),
    ).toBe(true);
    expect(stream).toHaveBeenCalledTimes(1);
    await act(async () => done.resolve());
    expect(stream).toHaveBeenCalledTimes(1);
    expect(input.value).toContain("Recognized speech");
    const savedNotice =
      locale === "zh"
        ? "语音已保留为草稿，请确认后手动发送。"
        : "Your speech was saved as a draft. Review it and send when ready.";
    expect(container.textContent).toContain(savedNotice);
    await act(async () =>
      (input.nextElementSibling as HTMLButtonElement).click(),
    );
    expect(stream).toHaveBeenCalledTimes(2);
    expect(stream.mock.calls[1][0]).toBe(
      existingDraft
        ? `${existingDraft} Recognized speech`
        : "Recognized speech",
    );
    expect(input.value).toBe("");
    expect(container.textContent).not.toContain(notice);
    expect(container.textContent).not.toContain(savedNotice);
  },
);

test("a queued speech callback from an unmounted conversation cannot alter the next draft", async () => {
  const done = deferred();
  const stream = jest
    .spyOn(ApiClient, "streamChat")
    .mockImplementation(async function* () {
      await done.promise;
      yield "Late reply";
    });
  await mount();
  await send("first question");
  const oldTranscription = mockVoiceProps.onTranscription;
  await act(async () => {
    root?.unmount();
    root = undefined;
  });
  await mount();
  const input = await draft("Current draft");
  await act(async () => {
    oldTranscription?.("stale speech");
    done.resolve();
  });
  expect(input.value).toBe("Current draft");
  expect(container.textContent).not.toContain("stale speech");
  expect(container.textContent).not.toContain("语音已保留为草稿");
  expect(stream).toHaveBeenCalledTimes(1);
});

test.each([
  ["zh", "停止接收", "已停止接收回复。"],
  ["en", "Stop receiving", "Stopped receiving the reply."],
] as const)(
  "%s stopping reception preserves partial text and unlocks only manual next submission",
  async (locale, stopLabel, stoppedReason) => {
    let activeSignal: AbortSignal | undefined;
    const stream = jest
      .spyOn(ApiClient, "streamChat")
      .mockImplementationOnce(
        async function* (_message, _session, _trace, citations, signal) {
          activeSignal = signal;
          citations?.([citation]);
          yield "Preserved partial";
          await new Promise<void>((_resolve, reject) =>
            signal?.addEventListener(
              "abort",
              () => reject(failure("aborted")),
              { once: true },
            ),
          );
        },
      )
      .mockImplementation(async function* () {
        yield "Next manual answer";
      });
    await mount(locale);
    await send("first question");
    const input = await draft("Manual next question");
    const stop = container.querySelector<HTMLButtonElement>(
      `button[aria-label="${stopLabel}"]`,
    );
    expect(stop).not.toBeNull();
    expect(stop?.disabled).toBe(false);
    await act(async () => stop?.click());
    expect(activeSignal?.aborted).toBe(true);
    expect(
      container.querySelector('[data-message-status="failed"]')?.textContent,
    ).toContain("Preserved partial");
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      stoppedReason,
    );
    expect(container.textContent).not.toContain("Current reference");
    expect(container.textContent).not.toContain("退款取消");
    expect(stream).toHaveBeenCalledTimes(1);
    expect(input.value).toBe("Manual next question");
    expect(
      container.querySelector(`button[aria-label="${stopLabel}"]`),
    ).toBeNull();
    expect((input.nextElementSibling as HTMLButtonElement).disabled).toBe(
      false,
    );
    await act(async () =>
      (input.nextElementSibling as HTMLButtonElement).click(),
    );
    expect(stream).toHaveBeenCalledTimes(2);
    expect(stream.mock.calls[1][0]).toBe("Manual next question");
    expect(container.textContent).toContain("Preserved partial");
    expect(container.textContent).toContain("Next manual answer");
  },
);

test("reception can be stopped before the first token without submitting a saved draft", async () => {
  let activeSignal: AbortSignal | undefined;
  const stream = jest
    .spyOn(ApiClient, "streamChat")
    .mockImplementation(
      async function* (_message, _session, _trace, _citations, signal) {
        activeSignal = signal;
        await new Promise<void>((_resolve, reject) =>
          signal?.addEventListener("abort", () => reject(failure("aborted")), {
            once: true,
          }),
        );
        yield "Never received";
      },
    );
  await mount();
  await send("first question");
  const input = await draft("Saved draft");
  const stop = container.querySelector<HTMLButtonElement>(
    'button[aria-label="停止接收"]',
  );
  expect(stop).not.toBeNull();
  await act(async () => stop?.click());
  expect(activeSignal?.aborted).toBe(true);
  expect(container.querySelector('[role="alert"]')?.textContent).toContain(
    "已停止接收回复。",
  );
  expect(container.textContent).not.toContain("Never received");
  expect(input.value).toBe("Saved draft");
  expect(stream).toHaveBeenCalledTimes(1);
  expect((input.nextElementSibling as HTMLButtonElement).disabled).toBe(false);
});

test("recognized speech uses the same dynamically issued visitor session as text", async () => {
  const stream = jest
    .spyOn(ApiClient, "streamChat")
    .mockImplementation(async function* () {
      yield "Voice reply";
    });
  await mount("en");
  await act(async () => mockVoiceProps.onTranscription?.("Visitor speech"));
  expect(stream).toHaveBeenCalledTimes(1);
  expect(stream.mock.calls[0][1]).toEqual({
    session_id: 317,
    session_key: "11111111-1111-4111-8111-111111111111",
  });
});

test("an expired visitor ignores a late speech callback instead of restoring private draft text", async () => {
  const stream = jest
    .spyOn(ApiClient, "streamChat")
    .mockImplementation(async function* () {
      throw failure("http", 401);
    });
  await mount("en");
  await send("Question before expiry");
  expect(container.textContent).toContain("visitor session expired");
  await act(async () =>
    mockVoiceProps.onTranscription?.("LATE PRIVATE SPEECH"),
  );
  expect(
    container.querySelector<HTMLInputElement>('input[type="text"]')?.value,
  ).toBe("");
  expect(container.textContent).not.toContain("LATE PRIVATE SPEECH");
  expect(stream).toHaveBeenCalledTimes(1);
});
