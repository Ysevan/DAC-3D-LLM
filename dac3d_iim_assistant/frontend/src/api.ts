import type {
  AssistantPayload,
  ChatRequest,
  KnowledgeBaseSummary,
  MachineAgentPayload,
  MachineSnapshot,
  RuntimeSummary,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const SESSION_STORAGE_KEY = "dac3d.session_id";
const OPERATOR_STORAGE_KEY = "dac3d.operator_id";

type StreamHandlers = {
  onMeta?: (payload: Omit<AssistantPayload, "answer">) => void;
  onStatus?: (payload: { stage: string; label: string }) => void;
  onDelta?: (chunk: string) => void;
  onDone?: (payload: AssistantPayload) => void;
  onError?: (message: string) => void;
};

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  options: { includeOperator?: boolean; sessionId?: string } = {},
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, withSecurityHeaders(init, options));
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
  const sessionId = request.session_id ?? getSessionId();
  return requestJson<AssistantPayload>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...request, session_id: sessionId }),
  }, { sessionId });
}

export function fetchMachineSnapshot(): Promise<MachineSnapshot> {
  return requestJson<MachineSnapshot>("/api/machine-agent/snapshot");
}

export function sendMachineAgentChat(message: string): Promise<MachineAgentPayload> {
  return requestJson<MachineAgentPayload>("/api/machine-agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: getSessionId() }),
  });
}

export async function streamChat(request: ChatRequest, handlers: StreamHandlers): Promise<void> {
  const sessionId = request.session_id ?? getSessionId();
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: withSecurityHeaders({ headers: { "Content-Type": "application/json" } }, { sessionId }).headers,
    body: JSON.stringify({ ...request, session_id: sessionId }),
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
  }, { includeOperator: true });
}

function withSecurityHeaders(
  init?: RequestInit,
  options: { includeOperator?: boolean; sessionId?: string } = {},
): RequestInit {
  const headers = new Headers(init?.headers);
  headers.set("X-DAC3D-Session-ID", options.sessionId ?? getSessionId());
  if (options.includeOperator) {
    headers.set("X-DAC3D-Operator-ID", getOperatorId());
    headers.set("X-DAC3D-Roles", "operator");
  }
  return { ...init, headers };
}

function getSessionId(): string {
  return getOrCreateBrowserId(SESSION_STORAGE_KEY, "web-session");
}

function getOperatorId(): string {
  return getOrCreateBrowserId(OPERATOR_STORAGE_KEY, "web-operator");
}

function getOrCreateBrowserId(storageKey: string, prefix: string): string {
  const storage = window.localStorage;
  const existing = storage.getItem(storageKey);
  if (existing) return existing;
  const randomId =
    typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const value = `${prefix}-${randomId}`;
  storage.setItem(storageKey, value);
  return value;
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
