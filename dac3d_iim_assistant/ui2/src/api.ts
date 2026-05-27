import type {
  AgentGoalActionResult,
  AgentGoalListResult,
  AgentWorkflowPreview,
  AgentWorkspace,
  ApproveCommandRequest,
  AssistantPayload,
  ChatRequest,
  CodexHandoffResult,
  CommandConfirmation,
  EvalDraftListResult,
  EvalRunResult,
  KnowledgeBaseSummary,
  MemoryPatchActionResult,
  MemoryPatchListResult,
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
    throw new Error(await extractApiErrorMessage(response));
  }
  return (await response.json()) as T;
}

export function fetchRuntimeSummary(): Promise<RuntimeSummary> {
  return requestJson<RuntimeSummary>("/api/runtime");
}

export function fetchAgentWorkspace(): Promise<AgentWorkspace> {
  return requestJson<AgentWorkspace>("/api/agent/workspace");
}

export function previewAgentWorkflow(task: string, sessionId: string): Promise<AgentWorkflowPreview> {
  return requestJson<AgentWorkflowPreview>("/api/agent/workflow/preview", withSecurityHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, session_id: sessionId }),
  }, { sessionId }));
}

export function fetchAgentGoals(status = "active"): Promise<AgentGoalListResult> {
  return requestJson<AgentGoalListResult>(`/api/goals?status=${encodeURIComponent(status)}`);
}

export function createAgentGoal(objective: string, sessionId: string): Promise<AgentGoalActionResult> {
  return requestJson<AgentGoalActionResult>("/api/goals", withSecurityHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ objective, session_id: sessionId }),
  }, { includeOperator: true, sessionId }));
}

export function appendAgentGoalProgress(goalId: string, note: string): Promise<AgentGoalActionResult> {
  return requestJson<AgentGoalActionResult>(`/api/goals/${encodeURIComponent(goalId)}/progress`, withSecurityHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note }),
  }, { includeOperator: true }));
}

export function completeAgentGoal(goalId: string, note = ""): Promise<AgentGoalActionResult> {
  return requestJson<AgentGoalActionResult>(`/api/goals/${encodeURIComponent(goalId)}/complete`, withSecurityHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note }),
  }, { includeOperator: true }));
}

export function fetchKnowledgeBaseSummary(): Promise<KnowledgeBaseSummary> {
  return requestJson<KnowledgeBaseSummary>("/api/knowledge-base/summary");
}

export function sendChat(request: ChatRequest): Promise<AssistantPayload> {
  const sessionId = request.session_id ?? getSessionId();
  return requestJson<AssistantPayload>("/api/chat", withSecurityHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...request, session_id: sessionId }),
  }, { sessionId }));
}

export function approvePendingCommand(request: ApproveCommandRequest): Promise<AssistantPayload> {
  return requestJson<AssistantPayload>("/api/commands/approve", withSecurityHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  }, { includeOperator: true, sessionId: request.session_id }));
}

export function runAgentEvals(categories?: string[]): Promise<EvalRunResult> {
  return requestJson<EvalRunResult>("/api/evals/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ categories }),
  });
}

export function fetchEvalDrafts(): Promise<EvalDraftListResult> {
  return requestJson<EvalDraftListResult>("/api/evals/drafts");
}

export function generateEvalDrafts(limit = 5): Promise<EvalDraftListResult> {
  return requestJson<EvalDraftListResult>("/api/evals/drafts", withSecurityHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit }),
  }, { includeOperator: true }));
}

export function generateCodexHandoff(recentTraceLimit = 8): Promise<CodexHandoffResult> {
  return requestJson<CodexHandoffResult>("/api/evals/codex-handoff", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ recent_trace_limit: recentTraceLimit }),
  });
}

export function fetchMemoryPatches(status = "pending"): Promise<MemoryPatchListResult> {
  return requestJson<MemoryPatchListResult>(`/api/memory/patches?status=${encodeURIComponent(status)}`);
}

export function approveMemoryPatch(patchId: string): Promise<MemoryPatchActionResult> {
  return requestJson<MemoryPatchActionResult>(`/api/memory/patches/${encodeURIComponent(patchId)}/approve`, withSecurityHeaders({
    method: "POST",
  }, { includeOperator: true, role: "admin" }));
}

export function rejectMemoryPatch(patchId: string, reason = "ui_rejected"): Promise<MemoryPatchActionResult> {
  return requestJson<MemoryPatchActionResult>(`/api/memory/patches/${encodeURIComponent(patchId)}/reject`, withSecurityHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason }),
  }, { includeOperator: true, role: "admin" }));
}

export async function streamChat(request: ChatRequest, handlers: StreamHandlers): Promise<void> {
  const sessionId = request.session_id ?? getSessionId();
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: withSecurityHeaders(
      { headers: { "Content-Type": "application/json" } },
      { sessionId },
    ).headers,
    body: JSON.stringify({ ...request, session_id: sessionId }),
  });
  if (!response.ok || !response.body) {
    throw new Error(await extractApiErrorMessage(response));
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

export function previewCommand(request: ChatRequest): Promise<AssistantPayload> {
  const sessionId = request.session_id ?? getSessionId();
  return requestJson<AssistantPayload>("/api/commands/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...request,
      session_id: sessionId,
      operator_id: getOperatorId(),
      roles: ["operator"],
    }),
  }, { includeOperator: true, sessionId });
}

export function confirmCommand(
  confirmation: CommandConfirmation,
  sessionId: string,
): Promise<AssistantPayload> {
  return requestJson<AssistantPayload>("/api/commands/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      preview_id: confirmation.preview_id,
      preview_hash: confirmation.preview_hash,
      confirmation_token: confirmation.confirmation_token,
      session_id: sessionId,
      operator_id: getOperatorId(),
      roles: ["operator"],
    }),
  }, { includeOperator: true, sessionId });
}

function withSecurityHeaders(
  init?: RequestInit,
  options: { includeOperator?: boolean; role?: string; sessionId?: string } = {},
): RequestInit {
  const headers = new Headers(init?.headers);
  headers.set("X-DAC3D-Session-ID", options.sessionId ?? headers.get("X-DAC3D-Session-ID") ?? getSessionId());
  if (options.includeOperator || options.role) {
    headers.set("X-DAC3D-Operator-ID", getOperatorId());
    headers.set("X-DAC3D-Roles", options.role ?? "operator");
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

async function extractApiErrorMessage(response: Response): Promise<string> {
  const rawText = await response.text();
  try {
    const payload = JSON.parse(rawText) as { error?: { code?: string; message?: string; trace_id?: string } };
    if (payload.error) {
      const code = payload.error.code ? `${payload.error.code}: ` : "";
      const trace = payload.error.trace_id ? ` trace_id=${payload.error.trace_id}` : "";
      return `${code}${payload.error.message ?? "请求失败。"}${trace}`;
    }
  } catch {
    // Keep the sanitized server text below.
  }
  return rawText || `HTTP ${response.status}`;
}
