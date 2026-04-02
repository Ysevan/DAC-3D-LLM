import { FormEvent, KeyboardEvent, memo, useEffect, useRef, useState, useLayoutEffect } from "react";
import { streamChat } from "./api";
import type { AssistantPayload, ChatHistoryTurn, MessageRecord } from "./types";
import { prepareWithSegments, layoutWithLines } from "@chenglou/pretext";

type StreamRenderState = {
  assistantId: string | null;
  queue: string[];
  timerId: number | null;
  cadenceMs: number;
  lastChunkAt: number | null;
  finalPayload: AssistantPayload | null;
  pendingPrompt: string | null;
  streamedAnswer: string;
};

const EMPTY_DETAILS: AssistantPayload = {
  intent: "query",
  answer: "",
  sources: [],
  command_preview: null,
  status_summary: null,
  parsed_result: null,
};

function App() {
  const [messages, setMessages] = useState<MessageRecord[]>([]);
  const [history, setHistory] = useState<ChatHistoryTurn[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  const endRef = useRef<HTMLDivElement | null>(null);
  const manualThemeOverrideRef = useRef(false);
  const lastAutoDarkRef = useRef<boolean | null>(null);
  const streamRenderRef = useRef<StreamRenderState>({
    assistantId: null,
    queue: [],
    timerId: null,
    cadenceMs: 18,
    lastChunkAt: null,
    finalPayload: null,
    pendingPrompt: null,
    streamedAnswer: "",
  });

  // Time-based lighting
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const hours = now.getHours();
      const isDark = hours < 6 || hours >= 18;

      const previousAutoDark = lastAutoDarkRef.current;
      const crossedNaturalBoundary = previousAutoDark !== null && previousAutoDark !== isDark;
      lastAutoDarkRef.current = isDark;

      if (manualThemeOverrideRef.current && !crossedNaturalBoundary) {
        return;
      }

      if (manualThemeOverrideRef.current && crossedNaturalBoundary) {
        manualThemeOverrideRef.current = false;
      }

      setIsDarkMode(isDark);
      if (isDark) {
        document.body.classList.add("dark");
      } else {
        document.body.classList.remove("dark");
      }
    };
    updateTime();
    const intervalId = window.setInterval(updateTime, 60000);
    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    const updateConversationGeometry = () => {
      const root = document.documentElement;
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const gutter = viewportWidth < 640 ? 14 : 24;

      if (viewportWidth < 900) {
        const mobileWidth = Math.max(0, viewportWidth - gutter * 2);
        root.style.setProperty("--conversation-width", `${mobileWidth}px`);
        root.style.setProperty("--conversation-offset", `${gutter}px`);
        return;
      }

      const desiredWidth = viewportWidth * 0.5;
      const maxWidthByHeight = viewportHeight > viewportWidth ? viewportWidth * 0.52 : viewportWidth * 0.7;
      const maxWidthByViewport = viewportWidth - gutter * 2;
      const safeWidth = Math.max(420, Math.min(desiredWidth, maxWidthByHeight, maxWidthByViewport));

      const remainingSpace = Math.max(0, viewportWidth - safeWidth - gutter * 2);
      const visualOffset = gutter + remainingSpace * 0.382;
      const offset = Math.min(Math.max(gutter, visualOffset), viewportWidth - safeWidth - gutter);

      root.style.setProperty("--conversation-width", `${safeWidth}px`);
      root.style.setProperty("--conversation-offset", `${offset}px`);
    };

    updateConversationGeometry();
    window.addEventListener("resize", updateConversationGeometry);
    return () => window.removeEventListener("resize", updateConversationGeometry);
  }, []);

  const toggleLighting = () => {
    manualThemeOverrideRef.current = true;
    document.body.classList.add('animation-ready');
    document.body.classList.toggle('dark');
    setIsDarkMode(document.body.classList.contains('dark'));
  };

  useEffect(() => {
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.code === 'Space' && event.target === document.body) {
        event.preventDefault();
        toggleLighting();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => () => clearStreamTimer(), []);

  async function handleSubmit(event?: FormEvent<HTMLFormElement>): Promise<void> {
    event?.preventDefault();
    const prompt = input.trim();
    if (!prompt || isSending) return;

    const timestamp = Date.now();
    const userId = `user-${timestamp}`;
    const assistantId = `assistant-${timestamp}`;
    resetStreamState(assistantId, prompt);
    
    setMessages((current) => [
      ...current,
      { id: userId, role: "user", content: prompt, status: "ready" },
      { id: assistantId, role: "assistant", content: "", status: "streaming" },
    ]);
    setInput("");
    setIsSending(true);

    try {
      await streamChat(
        { message: prompt, history },
        {
          onDelta: (chunk) => enqueueStreamChunk(assistantId, prompt, chunk),
          onDone: (payload) => finishStream(assistantId, prompt, payload),
          onError: (message) => {
            const errorText = `Error: ${message}`;
            setMessages((current) =>
              current.map((item) =>
                item.id === assistantId
                  ? { ...item, content: errorText, status: "error" }
                  : item
              )
            );
            setIsSending(false);
          },
        }
      );
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, content: `Error: ${errorText}`, status: "error" }
            : message
        )
      );
      setIsSending(false);
    }
  }

  function resetStreamState(assistantId: string, prompt: string): void {
    clearStreamTimer();
    streamRenderRef.current = {
      assistantId,
      queue: [],
      timerId: null,
      cadenceMs: 18,
      lastChunkAt: null,
      finalPayload: null,
      pendingPrompt: prompt,
      streamedAnswer: "",
    };
  }

  function clearStreamTimer(): void {
    const timerId = streamRenderRef.current.timerId;
    if (timerId !== null) {
      window.clearTimeout(timerId);
      streamRenderRef.current.timerId = null;
    }
  }

  function enqueueStreamChunk(assistantId: string, prompt: string, chunk: string): void {
    if (!chunk) return;

    const state = streamRenderRef.current;
    if (state.assistantId !== assistantId) {
      resetStreamState(assistantId, prompt);
    }

    const activeState = streamRenderRef.current;
    activeState.pendingPrompt = prompt;
    activeState.streamedAnswer += chunk;
    const characters = Array.from(chunk);
    const now = performance.now();

    if (activeState.lastChunkAt !== null && characters.length > 0) {
      const observedCadence = (now - activeState.lastChunkAt) / Math.max(characters.length, 1);
      const backlogBoost = activeState.queue.length > 36 ? 0.72 : activeState.queue.length > 12 ? 0.84 : 1;
      activeState.cadenceMs = Math.min(42, Math.max(10, Math.round((activeState.cadenceMs * 0.45 + observedCadence * 0.55) * backlogBoost)));
    }
    activeState.lastChunkAt = now;

    if (activeState.queue.length === 0 && characters.length > 0) {
      const immediateCharacter = characters.shift()!;
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, content: `${message.content}${immediateCharacter}`, status: "streaming" }
            : message
        )
      );
    }

    activeState.queue.push(...characters);
    scheduleNextCharacter(assistantId);
  }

  function finishStream(assistantId: string, prompt: string, payload: AssistantPayload): void {
    const state = streamRenderRef.current;
    if (state.assistantId !== assistantId) return;

    state.finalPayload = payload;
    state.pendingPrompt = prompt;
    if (state.queue.length === 0) {
      finalizeRenderedStream(assistantId);
    }
  }

  function scheduleNextCharacter(assistantId: string): void {
    const state = streamRenderRef.current;
    if (state.assistantId !== assistantId || state.timerId !== null || state.queue.length === 0) return;

    state.timerId = window.setTimeout(() => {
      state.timerId = null;
      flushNextCharacter(assistantId);
    }, state.cadenceMs);
  }

  function flushNextCharacter(assistantId: string): void {
    const state = streamRenderRef.current;
    if (state.assistantId !== assistantId) return;

    const nextCharacter = state.queue.shift();
    if (!nextCharacter) {
      if (state.finalPayload) finalizeRenderedStream(assistantId);
      return;
    }

    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? { ...message, content: `${message.content}${nextCharacter}`, status: "streaming" }
          : message
      )
    );

    if (state.queue.length > 0) {
      scheduleNextCharacter(assistantId);
      return;
    }

    if (state.finalPayload) finalizeRenderedStream(assistantId);
  }

  function finalizeRenderedStream(assistantId: string): void {
    const state = streamRenderRef.current;
    if (state.assistantId !== assistantId || !state.finalPayload) return;

    const payload = state.finalPayload;
    const prompt = state.pendingPrompt;
    const payloadAnswer = payload.answer || "";
    const streamedAnswer = state.streamedAnswer || "";
    const finalAnswer = mergeStreamedAndFinalAnswer(streamedAnswer, payloadAnswer);
    
    clearStreamTimer();
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId ? { ...message, content: finalAnswer, status: "ready", payload: { ...payload, answer: finalAnswer } } : message
      )
    );
    if (prompt) {
      setHistory((current) => [...current, { user: prompt, assistant: finalAnswer }]);
    }
    setIsSending(false);
    streamRenderRef.current = {
      assistantId: null, queue: [], timerId: null, cadenceMs: 18, lastChunkAt: null, finalPayload: null, pendingPrompt: null, streamedAnswer: "",
    };
  }

  function mergeStreamedAndFinalAnswer(streamedAnswer: string, payloadAnswer: string): string {
    if (!streamedAnswer) return payloadAnswer;
    if (!payloadAnswer) return streamedAnswer;
    if (streamedAnswer === payloadAnswer) return streamedAnswer;
    if (payloadAnswer.startsWith(streamedAnswer)) return payloadAnswer;
    if (streamedAnswer.startsWith(payloadAnswer)) return streamedAnswer;

    const maxOverlap = Math.min(streamedAnswer.length, payloadAnswer.length);
    for (let size = maxOverlap; size > 0; size -= 1) {
      if (streamedAnswer.slice(-size) === payloadAnswer.slice(0, size)) {
        return streamedAnswer + payloadAnswer.slice(size);
      }
    }

    return payloadAnswer;
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSubmit();
    }
  }

  return (
    <>
      <div id="dappled-light">
        <div id="glow" />
        <div id="glow-bounce" />
        <div className="perspective">
          <div id="leaves">
            <svg style={{ width: 0, height: 0, position: "absolute" }}>
              <defs>
                <filter id="wind" x="-20%" y="-20%" width="140%" height="140%">
                  <feTurbulence type="fractalNoise" numOctaves="2" seed="1">
                    <animate
                      attributeName="baseFrequency"
                      dur="16s"
                      keyTimes="0;0.33;0.66;1"
                      values="0.005 0.003;0.01 0.009;0.008 0.004;0.005 0.003"
                      repeatCount="indefinite"
                    />
                  </feTurbulence>
                  <feDisplacementMap in="SourceGraphic">
                    <animate
                      attributeName="scale"
                      dur="20s"
                      keyTimes="0;0.25;0.5;0.75;1"
                      values="45;55;75;55;45"
                      repeatCount="indefinite"
                    />
                  </feDisplacementMap>
                </filter>
              </defs>
            </svg>
          </div>
          <div id="blinds">
            <div className="shutters">
              {[...Array(23)].map((_, i) => <div key={i} className="shutter" />)}
            </div>
            <div className="vertical">
              <div className="bar" />
              <div className="bar" />
            </div>
          </div>
        </div>
        <div id="progressive-blur">
          <div />
          <div />
          <div />
          <div />
        </div>
      </div>

      <div className="app-container">
        <div className="conversation-column">
          <div className="chat-log">
            <div style={{ flex: 1, minHeight: 0 }} />
            {messages.map((message) => (
              <MessageRow key={message.id} message={message} />
            ))}
            <div ref={endRef} />
          </div>

          <div className="composer-container">
            <form className="composer-form" onSubmit={(event) => void handleSubmit(event)}>
              <span className="composer-prompt">&gt;</span>
              <textarea
                className="composer-input"
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder="Type a message..."
                rows={1}
                value={input}
                autoFocus
              />
            </form>
            <div className="controls" aria-label="lighting control">
              <button 
                type="button"
                className="sunlit-toggle"
                onClick={toggleLighting}
              >
                [toggle the sun]
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

const MessageRow = memo(function MessageRow({ message }: { message: MessageRecord }) {
  return (
    <div className={`log-entry role-${message.role}`}>
      <div className="log-role">{message.role === "user" ? "USER" : "DAC-3D"}</div>
      <div className="log-content">
        <PretextLayout content={message.content || "..."} showCursor={message.status === "streaming"} />
      </div>
    </div>
  );
});

function PretextLayout({ content, showCursor }: { content: string, showCursor: boolean }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [lines, setLines] = useState<{text: string, width: number}[]>([]);

  useLayoutEffect(() => {
    if (!containerRef.current) return;
    
    const updateLayout = () => {
      if (!containerRef.current) return;
      const width = containerRef.current.clientWidth || 600;
      try {
        const prepared = prepareWithSegments(content, '14px "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", monospace', { whiteSpace: 'pre-wrap' });
        const layoutResult = layoutWithLines(prepared, width, 21);
        setLines(layoutResult.lines);
      } catch (e) {
        setLines([{ text: content, width: width }]);
      }
    };

    updateLayout();
    window.addEventListener('resize', updateLayout);
    return () => window.removeEventListener('resize', updateLayout);
  }, [content]);

  if (lines.length === 0) {
    return <div ref={containerRef}>{content}{showCursor && <span className="cursor-blink" />}</div>;
  }

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {lines.map((line, i) => (
        <div key={i} style={{ height: '21px', whiteSpace: 'pre-wrap' }}>
          {line.text}
          {showCursor && i === lines.length - 1 && <span className="cursor-blink" />}
        </div>
      ))}
    </div>
  );
}

export default App;
