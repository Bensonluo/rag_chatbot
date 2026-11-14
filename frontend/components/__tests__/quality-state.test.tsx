import { act, StrictMode, type ReactNode } from "react";
import { createRoot, hydrateRoot, type Root } from "react-dom/client";
import { LanguageProvider, useLanguage } from "@/lib/i18n/context";
import { VoiceButton } from "@/components/VoiceButton";
import { ChatInterfaceUpdated } from "@/components/ChatInterfaceUpdated";
import {
  GuestApi as ApiClient,
  type GuestHistory as ChatHistoryResponse,
} from "@/lib/guest-api";

const { renderToString } =
  require("react-dom/server.node") as typeof import("react-dom/server");

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

function LocaleControls() {
  const { locale, setLocale } = useLanguage();
  return (
    <button
      data-locale={locale}
      onClick={() => setLocale(locale === "zh" ? "en" : "zh")}
    >
      Change locale
    </button>
  );
}

function App({ children }: { children: ReactNode }) {
  return (
    <StrictMode>
      <LanguageProvider>
        <LocaleControls />
        {children}
      </LanguageProvider>
    </StrictMode>
  );
}

let container: HTMLDivElement;
let root: Root | undefined;

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
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
  Reflect.deleteProperty(window, "SpeechRecognition");
  Reflect.deleteProperty(window, "webkitSpeechRecognition");
  jest.useRealTimers();
});

async function mount(children: ReactNode) {
  await act(async () => {
    root = createRoot(container);
    root.render(<App>{children}</App>);
  });
}

async function changeLocale() {
  const button = container.querySelector<HTMLButtonElement>("[data-locale]");
  expect(button).not.toBeNull();
  await act(async () => button?.click());
}

function mockSystemLanguage(language: string) {
  jest.spyOn(window.navigator, "language", "get").mockReturnValue(language);
  jest
    .spyOn(window.navigator, "languages", "get")
    .mockReturnValue(language ? [language] : []);
}

async function sendMessage(content: string) {
  const input = container.querySelector<HTMLInputElement>('input[type="text"]');
  expect(input).not.toBeNull();
  await act(async () => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )?.set?.call(input, content);
    input?.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const send = input?.nextElementSibling as HTMLButtonElement | null;
  expect(send?.disabled).toBe(false);
  await act(async () => send?.click());
}

test.each([
  ["zh-CN", "zh"],
  ["zh-TW", "zh"],
  ["en-US", "en"],
  ["fr-FR", "en"],
  ["", "en"],
] as const)(
  "system language %s selects %s when no locale is saved",
  async (systemLanguage, expectedLocale) => {
    mockSystemLanguage(systemLanguage);
    await mount(<div />);
    expect(
      container.querySelector("[data-locale]")?.getAttribute("data-locale"),
    ).toBe(expectedLocale);
  },
);

test("saved locale takes precedence over the system language", async () => {
  mockSystemLanguage("en-US");
  localStorage.setItem("portfolio-locale", "zh");
  await mount(<div />);
  expect(
    container.querySelector("[data-locale]")?.getAttribute("data-locale"),
  ).toBe("zh");
});

test.each(["manual", "auto"] as const)(
  "%s voice support and unsupported copy follow the locale",
  async (mode) => {
    localStorage.setItem("portfolio-locale", "zh");
    await mount(<VoiceButton mode={mode} />);
    const voice = container.querySelector<HTMLButtonElement>(
      "button:not([data-locale])",
    );
    expect(voice?.disabled).toBe(true);
    expect(container.textContent).toContain("浏览器不支持语音识别");
    await changeLocale();
    expect(container.textContent).toContain(
      "Browser does not support speech recognition",
    );
    expect(container.textContent).not.toContain("浏览器不支持语音识别");
  },
);

test.each(["manual", "auto"] as const)(
  "%s voice enables when Web Speech is supported",
  async (mode) => {
    Object.defineProperty(window, "SpeechRecognition", {
      configurable: true,
      value: jest.fn(),
    });
    await mount(<VoiceButton mode={mode} />);
    expect(
      container.querySelector<HTMLButtonElement>("button:not([data-locale])")
        ?.disabled,
    ).toBe(false);
    expect(container.textContent).not.toContain("浏览器不支持语音识别");
    await changeLocale();
    expect(
      container.querySelector<HTMLButtonElement>("button:not([data-locale])")
        ?.disabled,
    ).toBe(false);
  },
);

test("stored locale hydrates with server markup and updates document language", async () => {
  localStorage.setItem("portfolio-locale", "en");
  const app = (
    <App>
      <VoiceButton />
    </App>
  );
  container.innerHTML = renderToString(app);
  expect(
    container.querySelector("[data-locale]")?.getAttribute("data-locale"),
  ).toBe("zh");
  const errors: unknown[] = [];
  await act(async () => {
    root = hydrateRoot(container, app, {
      onRecoverableError: (error) => errors.push(error),
    });
  });
  expect(errors).toEqual([]);
  expect(
    container.querySelector("[data-locale]")?.getAttribute("data-locale"),
  ).toBe("en");
  expect(document.documentElement.lang).toBe("en");
  await changeLocale();
  expect(localStorage.getItem("portfolio-locale")).toBe("zh");
  expect(document.documentElement.lang).toBe("zh-CN");
});

test("welcome text follows locale without a state-sync effect", async () => {
  localStorage.setItem("portfolio-locale", "zh");
  jest.spyOn(ApiClient, "streamChat").mockImplementation(async function* () {
    yield "Current reply";
  });
  await mount(<ChatInterfaceUpdated />);
  expect(container.textContent).toContain("欢迎来到 Agent Store");
  await changeLocale();
  expect(container.textContent).toContain("Welcome to Agent Store");
  expect(container.textContent).not.toContain("欢迎来到 Agent Store");
  await sendMessage("New question");
  expect(container.textContent).toContain("Current reply");
  expect(container.textContent).toContain("Welcome to Agent Store");
  expect(container.textContent).not.toContain("欢迎来到 Agent Store");
});

test("late history does not overwrite a newly started conversation", async () => {
  let resolveHistory: (value: ChatHistoryResponse) => void = () => {};
  const pendingHistory = new Promise<ChatHistoryResponse>((resolve) => {
    resolveHistory = resolve;
  });
  jest.mocked(ApiClient.getChatHistory).mockReturnValue(pendingHistory);
  jest.spyOn(ApiClient, "streamChat").mockImplementation(async function* () {
    yield "New answer";
  });
  await mount(<ChatInterfaceUpdated />);
  await sendMessage("New question");
  await act(async () =>
    resolveHistory({
      session_id: 317,
      session_key: "11111111-1111-4111-8111-111111111111",
      messages: [{ role: "assistant", content: "Late old answer" }],
    }),
  );
  expect(container.textContent).toContain("New question");
  expect(container.textContent).toContain("New answer");
  expect(container.textContent).not.toContain("Late old answer");
});

test("a single historical answer and its citation survive locale changes", async () => {
  jest.mocked(ApiClient.getChatHistory).mockResolvedValue({
    session_id: 317,
    session_key: "11111111-1111-4111-8111-111111111111",
    messages: [
      {
        role: "assistant",
        content: "保留回答时的内容",
        citations: [
          {
            source_type: "document",
            title: "回答时的资料",
            version_id: "old-version",
            excerpt: "原版本片段",
            section_path: [],
            source_document_ids: [],
            source_version_ids: [],
            source_chunk_ids: [],
          },
        ],
      },
    ],
  });
  await mount(<ChatInterfaceUpdated />);
  expect(container.textContent).toContain("保留回答时的内容");
  await changeLocale();
  expect(container.textContent).toContain("保留回答时的内容");
  expect(container.textContent).toContain("原版本片段");
  expect(container.querySelector("details")?.open).toBe(false);
  expect(container.textContent).not.toContain(
    "Hello! I am the AI customer service assistant",
  );
});

class FakeRecognition {
  static instances: FakeRecognition[] = [];
  static emitStart = true;
  static throwStarts = 0;
  onstart: (() => void) | null = null;
  onend: (() => void) | null = null;
  onerror: ((event: { error: string }) => void) | null = null;
  onresult:
    | ((event: {
        resultIndex: number;
        results: { isFinal: boolean; 0: { transcript: string } }[];
      }) => void)
    | null = null;
  start = jest.fn(() => {
    if (FakeRecognition.throwStarts > 0) {
      FakeRecognition.throwStarts -= 1;
      throw new Error("Recognition engine busy");
    }
    if (FakeRecognition.emitStart) this.onstart?.();
  });
  stop = jest.fn();
  abort = jest.fn();
  constructor() {
    FakeRecognition.instances.push(this);
  }
}

function installRecognition() {
  FakeRecognition.instances = [];
  FakeRecognition.emitStart = true;
  FakeRecognition.throwStarts = 0;
  Object.defineProperty(window, "SpeechRecognition", {
    configurable: true,
    value: FakeRecognition,
  });
}

function voiceButton() {
  const button = container.querySelector<HTMLButtonElement>(
    "button:not([data-locale])",
  );
  if (!button) throw new Error("Voice button not rendered");
  return button;
}

test.each(["manual", "auto"] as const)(
  "%s releases recognition and ignores queued callbacks on unmount",
  async (mode) => {
    jest.useFakeTimers();
    installRecognition();
    const onTranscription = jest.fn();
    const onError = jest.fn();
    await mount(
      <VoiceButton
        mode={mode}
        onTranscription={onTranscription}
        onError={onError}
      />,
    );
    await act(async () => voiceButton().click());
    const recognition = FakeRecognition.instances[0];
    await act(async () =>
      recognition.onresult?.({
        resultIndex: 0,
        results: [{ isFinal: true, 0: { transcript: "pending speech" } }],
      }),
    );
    const queuedResult = recognition.onresult;
    const queuedEnd = recognition.onend;
    const queuedStart = recognition.onstart;
    const queuedError = recognition.onerror;
    await act(async () => {
      root?.unmount();
      root = undefined;
    });
    expect(recognition.abort).toHaveBeenCalledTimes(1);
    expect(recognition.onresult).toBeNull();
    expect(recognition.onend).toBeNull();
    await act(async () => {
      queuedResult?.({
        resultIndex: 0,
        results: [{ isFinal: true, 0: { transcript: "stale speech" } }],
      });
      queuedEnd?.();
      queuedStart?.();
      queuedError?.({ error: "not-allowed" });
      jest.advanceTimersByTime(3000);
    });
    expect(onTranscription).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
    expect(recognition.start).toHaveBeenCalledTimes(1);
  },
);

test("switching mode cancels an automatic restart queued by onend", async () => {
  jest.useFakeTimers();
  installRecognition();
  await mount(<VoiceButton mode="auto" />);
  await act(async () => voiceButton().click());
  const recognition = FakeRecognition.instances[0];
  await act(async () => recognition.onend?.());
  await act(async () =>
    root?.render(
      <App>
        <VoiceButton mode="manual" />
      </App>,
    ),
  );
  await act(async () => jest.advanceTimersByTime(1000));
  expect(recognition.abort).toHaveBeenCalledTimes(1);
  expect(recognition.start).toHaveBeenCalledTimes(1);
  expect(FakeRecognition.instances).toHaveLength(1);
});

test("a stopped automatic session cannot restart its replacement", async () => {
  jest.useFakeTimers();
  installRecognition();
  await mount(<VoiceButton mode="auto" />);
  await act(async () => voiceButton().click());
  const first = FakeRecognition.instances[0];
  await act(async () => first.onend?.());
  await act(async () => voiceButton().click());
  await act(async () => voiceButton().click());
  const second = FakeRecognition.instances[1];
  await act(async () => jest.advanceTimersByTime(1000));
  expect(first.stop).toHaveBeenCalledTimes(1);
  expect(first.start).toHaveBeenCalledTimes(1);
  expect(second.start).toHaveBeenCalledTimes(1);
  await act(async () => second.onend?.());
  await act(async () => jest.advanceTimersByTime(500));
  expect(second.start).toHaveBeenCalledTimes(2);
});

test("manual mode can cancel its pending start without creating another recognizer", async () => {
  installRecognition();
  FakeRecognition.emitStart = false;
  await mount(<VoiceButton mode="manual" />);
  await act(async () => voiceButton().click());
  await act(async () => voiceButton().click());
  expect(FakeRecognition.instances).toHaveLength(1);
  expect(FakeRecognition.instances[0].start).toHaveBeenCalledTimes(1);
  expect(FakeRecognition.instances[0].stop).toHaveBeenCalledTimes(1);
  expect(FakeRecognition.instances[0].onstart).toBeNull();
});

test.each(["manual", "auto"] as const)(
  "%s remains stoppable when its parent disables new recognition",
  async (mode) => {
    jest.useFakeTimers();
    installRecognition();
    await mount(<VoiceButton mode={mode} />);
    await act(async () => voiceButton().click());
    const recognition = FakeRecognition.instances[0];
    await act(async () =>
      root?.render(
        <App>
          <VoiceButton mode={mode} disabled />
        </App>,
      ),
    );
    expect(voiceButton().disabled).toBe(false);
    await act(async () => voiceButton().click());
    expect(recognition.stop).toHaveBeenCalledTimes(1);
    expect(voiceButton().disabled).toBe(true);
    await act(async () => jest.advanceTimersByTime(1000));
    expect(recognition.start).toHaveBeenCalledTimes(1);
  },
);

test.each(["manual", "auto"] as const)(
  "%s recovers after a synchronous recognition start failure",
  async (mode) => {
    installRecognition();
    FakeRecognition.throwStarts = 1;
    const onError = jest.fn();
    await mount(<VoiceButton mode={mode} onError={onError} />);
    await act(async () => voiceButton().click());
    expect(onError).toHaveBeenCalledTimes(1);
    expect(FakeRecognition.instances[0].abort).toHaveBeenCalledTimes(1);
    expect(FakeRecognition.instances[0].onend).toBeNull();
    expect(voiceButton().disabled).toBe(false);
    await act(async () => voiceButton().click());
    expect(FakeRecognition.instances).toHaveLength(2);
    expect(FakeRecognition.instances[1].start).toHaveBeenCalledTimes(1);
    expect(container.textContent).not.toContain("语音识别未能启动");
  },
);

test("automatic restart failure ends the old session and permits a new one", async () => {
  jest.useFakeTimers();
  installRecognition();
  const onError = jest.fn();
  await mount(<VoiceButton mode="auto" onError={onError} />);
  await act(async () => voiceButton().click());
  const recognition = FakeRecognition.instances[0];
  FakeRecognition.throwStarts = 1;
  await act(async () => recognition.onend?.());
  await act(async () => jest.advanceTimersByTime(500));
  expect(onError).toHaveBeenCalledTimes(1);
  expect(recognition.abort).toHaveBeenCalledTimes(1);
  await act(async () => jest.advanceTimersByTime(2000));
  expect(recognition.start).toHaveBeenCalledTimes(2);
  await act(async () => voiceButton().click());
  expect(FakeRecognition.instances).toHaveLength(2);
});
