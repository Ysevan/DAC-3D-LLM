import type {
  AssistantPayload,
  ChatRequest,
  KnowledgeBaseSummary,
  RuntimeSummary,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

type StreamHandlers = {
  onMeta?: (payload: Omit<AssistantPayload, "answer">) => void;
  onStatus?: (payload: { stage: string; label: string }) => void;
  onDelta?: (chunk: string) => void;
  onDone?: (payload: AssistantPayload) => void;
  onError?: (message: string) => void;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}

export function fetchRuntimeSummary(): Promise<RuntimeSummary> {
  return requestJson<RuntimeSummary>("/api/runtime");
}

export function fetchKnowledgeBaseSummary(): Promise<KnowledgeBaseSummary> {
  return requestJson<KnowledgeBaseSummary>("/api/knowledge-base/summary");
}

export function sendChat(request: ChatRequest): Promise<AssistantPayload> {
  return requestJson<AssistantPayload>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export async function streamChat(request: ChatRequest, handlers: StreamHandlers): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok || !response.body) {
    throw new Error(await response.text());
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    buffer = processSseBuffer(buffer, handlers);
  }

  buffer += decoder.decode();
  buffer = processSseBuffer(buffer, handlers);

  if (buffer.trim()) {
    processSseEvent(buffer.trim(), handlers);
  }
}

export async function buildKnowledgeBase(files: File[]): Promise<{
  runtime: RuntimeSummary;
  knowledge_base: KnowledgeBaseSummary;
}> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  return requestJson("/api/knowledge-base/build", {
    method: "POST",
    body: formData,
  });
}

function processSseBuffer(buffer: string, handlers: StreamHandlers): string {
  buffer = buffer.replace(/\r\n/g, "\n");
  let cursor = buffer.indexOf("\n\n");
  while (cursor !== -1) {
    const rawEvent = buffer.slice(0, cursor).trim();
    buffer = buffer.slice(cursor + 2);
    if (rawEvent) {
      processSseEvent(rawEvent, handlers);
    }
    cursor = buffer.indexOf("\n\n");
  }
  return buffer;
}

function processSseEvent(rawEvent: string, handlers: StreamHandlers): void {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of rawEvent.split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (!dataLines.length) {
    return;
  }

  const payload = JSON.parse(dataLines.join("\n")) as
    | AssistantPayload
    | Omit<AssistantPayload, "answer">
    | { stage: string; label: string }
    | { chunk: string }
    | { message: string };

  if (eventName === "meta") {
    handlers.onMeta?.(payload as Omit<AssistantPayload, "answer">);
    return;
  }
  if (eventName === "delta") {
    handlers.onDelta?.((payload as { chunk: string }).chunk);
    return;
  }
  if (eventName === "status") {
    handlers.onStatus?.(payload as { stage: string; label: string });
    return;
  }
  if (eventName === "done") {
    handlers.onDone?.(payload as AssistantPayload);
    return;
  }
  if (eventName === "error") {
    handlers.onError?.((payload as { message: string }).message);
  }
}
