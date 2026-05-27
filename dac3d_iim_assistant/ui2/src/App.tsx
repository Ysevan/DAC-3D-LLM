import { FormEvent, KeyboardEvent, ReactNode, memo, useCallback, useEffect, useMemo, useRef, useState, useLayoutEffect } from "react";
import { createPortal } from "react-dom";

import {
  appendAgentGoalProgress,
  approveMemoryPatch,
  approvePendingCommand,
  buildKnowledgeBase,
  completeAgentGoal,
  createAgentGoal,
  fetchAgentGoals,
  fetchAgentWorkspace,
  fetchEvalDrafts,
  fetchKnowledgeBaseSummary,
  fetchMemoryPatches,
  fetchRuntimeSummary,
  generateCodexHandoff,
  generateEvalDrafts,
  previewAgentWorkflow,
  rejectMemoryPatch,
  runAgentEvals,
  streamChat,
} from "./api";
import type {
  AgentGoal,
  AgentGoalListResult,
  AgentWorkflowPreview,
  AgentWorkspace,
  AssistantPayload,
  ChatHistoryTurn,
  CodexHandoffResult,
  EvalDraftListResult,
  EvalRunResult,
  KnowledgeBaseSummary,
  MemoryPatch,
  MemoryPatchListResult,
  MessageRecord,
  RuntimeSummary,
} from "./types";

type PanelMode = "hidden" | "details" | "settings";
type ThemeMode = "auto" | "light" | "dark";
type ToolGatewayToolView = {
  name: string;
  riskLevel: string;
  requiresConfirmation: boolean;
  readOnlyHint: boolean;
  destructiveHint: boolean;
  idempotentHint: boolean;
  openWorldHint: boolean;
};

type ToolGatewayView = {
  enabled: boolean;
  toolCount: number;
  workflow: string;
  metadataPolicy: string;
  policyEngineEnabled: boolean;
  policyMode: string;
  policyEnforcement: string;
  tools: ToolGatewayToolView[];
};

type AgentWorkspaceView = {
  entryAgent: string;
  specialistCount: number;
  skillCount: number;
  contextNodeCount: number;
  memoryTraceCount: number;
  goalCount: number;
  workflow: string[];
  agentNames: string[];
  contextKinds: Array<{ name: string; count: number }>;
};

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

function createChatSessionId(): string {
  return `web-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function App() {
  const [themeMode, setThemeMode] = useState<ThemeMode>("auto");
  const [systemTheme, setSystemTheme] = useState<"light" | "dark">("light");
  const [messages, setMessages] = useState<MessageRecord[]>([]);
  const [history, setHistory] = useState<ChatHistoryTurn[]>([]);
  const [sessionId, setSessionId] = useState(createChatSessionId);
  const [input, setInput] = useState("");
  const [panelMode, setPanelMode] = useState<PanelMode>("hidden");
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [requestStage, setRequestStage] = useState<string | null>(null);
  const [detailsPayload, setDetailsPayload] = useState<AssistantPayload>(EMPTY_DETAILS);
  const [runtimeSummary, setRuntimeSummary] = useState<RuntimeSummary | null>(null);
  const [knowledgeBaseSummary, setKnowledgeBaseSummary] = useState<KnowledgeBaseSummary | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [approvalInFlight, setApprovalInFlight] = useState<string | null>(null);
  const [approvedPreviewIds, setApprovedPreviewIds] = useState<string[]>([]);
  const [isBuilding, setIsBuilding] = useState(false);
  const [buildStatus, setBuildStatus] = useState("");
  const [evalResult, setEvalResult] = useState<EvalRunResult | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalStatus, setEvalStatus] = useState("");
  const [evalDrafts, setEvalDrafts] = useState<EvalDraftListResult | null>(null);
  const [isGeneratingEvalDrafts, setIsGeneratingEvalDrafts] = useState(false);
  const [evalDraftStatus, setEvalDraftStatus] = useState("");
  const [codexHandoff, setCodexHandoff] = useState<CodexHandoffResult | null>(null);
  const [isGeneratingCodexHandoff, setIsGeneratingCodexHandoff] = useState(false);
  const [codexHandoffStatus, setCodexHandoffStatus] = useState("");
  const [memoryPatches, setMemoryPatches] = useState<MemoryPatchListResult | null>(null);
  const [memoryPatchStatus, setMemoryPatchStatus] = useState("");
  const [memoryPatchBusyId, setMemoryPatchBusyId] = useState<string | null>(null);
  const [agentWorkspace, setAgentWorkspace] = useState<AgentWorkspace | null>(null);
  const [workflowTask, setWorkflowTask] = useState("选择 pre_fusion_images 下的图片进行离线检测");
  const [workflowPreview, setWorkflowPreview] = useState<AgentWorkflowPreview | null>(null);
  const [workflowStatus, setWorkflowStatus] = useState("");
  const [isPreviewingWorkflow, setIsPreviewingWorkflow] = useState(false);
  const [agentGoals, setAgentGoals] = useState<AgentGoalListResult | null>(null);
  const [goalInput, setGoalInput] = useState("持续升级 DAC-Agent Runtime 功能");
  const [goalStatus, setGoalStatus] = useState("");
  const [goalBusyId, setGoalBusyId] = useState<string | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const endRef = useRef<HTMLDivElement | null>(null);
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
  const hasConversation = messages.length > 0;
  const theme = themeMode === "auto" ? systemTheme : themeMode;
  const dac3dRuntime = useMemo(() => buildDac3dRuntimeView(runtimeSummary), [runtimeSummary]);
  const toolGateway = useMemo(() => buildToolGatewayView(runtimeSummary), [runtimeSummary]);
  const agentWorkspaceView = useMemo(() => buildAgentWorkspaceView(agentWorkspace), [agentWorkspace]);

  useEffect(() => {
    void refreshSidebarData();
  }, []);

  useEffect(() => {
    const updateSystemTheme = () => {
      const hour = new Date().getHours();
      const isDayTime = hour >= 6 && hour < 18;
      setSystemTheme(isDayTime ? "light" : "dark");
    };
    updateSystemTheme();
    const intervalId = window.setInterval(updateSystemTheme, 60000);
    return () => window.clearInterval(intervalId);
  }, []);

  const toggleThemeMode = useCallback(() => {
    setThemeMode((prev) => (prev === "auto" ? "dark" : prev === "dark" ? "light" : "auto"));
  }, []);

  useEffect(() => {
    // Panel mode effect removed as menuOpen is gone
  }, [panelMode]);

  useEffect(() => {
    if (!hasConversation) {
      return;
    }
    const lastMessage = messages[messages.length - 1];
    endRef.current?.scrollIntoView({
      behavior: lastMessage?.status === "streaming" ? "auto" : "smooth",
      block: "end",
    });
  }, [hasConversation, messages]);

  useEffect(() => {
    if (!copiedMessageId) {
      return;
    }
    const timeoutId = window.setTimeout(() => setCopiedMessageId(null), 1400);
    return () => window.clearTimeout(timeoutId);
  }, [copiedMessageId]);

  useEffect(() => () => clearStreamTimer(), []);

  async function refreshSidebarData(): Promise<void> {
    try {
      const [runtime, knowledgeBase, patches, drafts, workspace, goals] = await Promise.all([
        fetchRuntimeSummary(),
        fetchKnowledgeBaseSummary(),
        fetchMemoryPatches().catch(() => null),
        fetchEvalDrafts().catch(() => null),
        fetchAgentWorkspace().catch(() => null),
        fetchAgentGoals().catch(() => null),
      ]);
      setRuntimeSummary(runtime);
      setKnowledgeBaseSummary(knowledgeBase);
      if (patches) {
        setMemoryPatches(patches);
      }
      if (drafts) {
        setEvalDrafts(drafts);
      }
      if (workspace) {
        setAgentWorkspace(workspace);
      }
      if (goals) {
        setAgentGoals(goals);
      }
    } catch (error) {
      setBuildStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleSubmit(event?: FormEvent<HTMLFormElement>): Promise<void> {
    event?.preventDefault();
    const prompt = input.trim();
    if (!prompt || isSending) {
      return;
    }

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
    setRequestStage("请求已发送");
    setDetailsPayload(EMPTY_DETAILS);

    try {
      await streamChat(
        { message: prompt, history, session_id: sessionId },
        {
          onMeta: (payload) => {
            setDetailsPayload((current) => ({
              ...current,
              ...payload,
              answer: current.answer,
            }));
            setMessages((current) =>
              current.map((item) =>
                item.id === assistantId
                  ? { ...item, payload: { ...(item.payload || EMPTY_DETAILS), ...payload, answer: item.content } }
                  : item,
              ),
            );
          },
          onStatus: (payload) => {
            setRequestStage(payload.label);
          },
          onDelta: (chunk) => {
            enqueueStreamChunk(assistantId, prompt, chunk);
          },
          onDone: (payload) => {
            finishStream(assistantId, prompt, payload);
          },
          onError: (message) => {
            const errorText = `系统处理失败: ${message}`;
            setMessages((current) =>
              current.map((item) =>
                item.id === assistantId
                  ? {
                      ...item,
                      content: errorText,
                      status: "error",
                      payload: {
                        ...EMPTY_DETAILS,
                        intent: "error",
                        answer: errorText,
                      },
                    }
                  : item,
              ),
            );
            setRequestStage(null);
            setDetailsPayload({
              ...EMPTY_DETAILS,
              intent: "error",
              answer: errorText,
            });
            setIsSending(false);
          },
        },
      );
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: `系统处理失败: ${errorText}`,
                status: "error",
                payload: {
                  ...EMPTY_DETAILS,
                  intent: "error",
                  answer: `系统处理失败: ${errorText}`,
                },
              }
            : message,
        ),
      );
      setDetailsPayload({
        ...EMPTY_DETAILS,
        intent: "error",
        answer: `系统处理失败: ${errorText}`,
      });
      setRequestStage(null);
      setIsSending(false);
    }
  }

  async function handleApproveCommand(message: MessageRecord): Promise<void> {
    const approval = getApprovalRequest(message.payload);
    if (!approval || isSending || approvalInFlight) {
      return;
    }

    const timestamp = Date.now();
    const userId = `approval-user-${timestamp}`;
    const assistantId = `approval-assistant-${timestamp}`;
    setMessages((current) => [
      ...current,
      { id: userId, role: "user", content: "批准执行", status: "ready" },
      { id: assistantId, role: "assistant", content: "正在提交已批准的 DAC-3D 命令…", status: "streaming" },
    ]);
    setIsSending(true);
    setApprovalInFlight(approval.previewId);
    setRequestStage("提交批准中");
    setDetailsPayload(message.payload ?? EMPTY_DETAILS);

    try {
      const payload = await approvePendingCommand({
        session_id: sessionId,
        preview_id: approval.previewId,
        confirmation_token: approval.confirmationToken,
      });
      const answer = payload.answer || "命令已提交。";
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId
            ? { ...item, content: answer, status: "ready", payload: { ...payload, answer } }
            : item,
        ),
      );
      setHistory((current) => [...current, { user: "批准执行", assistant: answer }]);
      setDetailsPayload({ ...payload, answer });
      setApprovedPreviewIds((current) =>
        current.includes(approval.previewId) ? current : [...current, approval.previewId],
      );
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      const answer = `批准执行失败: ${errorText}`;
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                content: answer,
                status: "error",
                payload: { ...EMPTY_DETAILS, intent: "operation", answer },
              }
            : item,
        ),
      );
      setDetailsPayload({ ...EMPTY_DETAILS, intent: "operation", answer });
    } finally {
      setIsSending(false);
      setApprovalInFlight(null);
      setRequestStage(null);
      void refreshSidebarData();
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
    if (!chunk) {
      return;
    }

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
      activeState.cadenceMs = clampCadence((activeState.cadenceMs * 0.45 + observedCadence * 0.55) * backlogBoost);
    }
    activeState.lastChunkAt = now;

    if (activeState.queue.length === 0 && characters.length > 0) {
      const immediateCharacter = characters.shift()!;
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, content: `${message.content}${immediateCharacter}`, status: "streaming" }
            : message,
        ),
      );
    }

    activeState.queue.push(...characters);
    scheduleNextCharacter(assistantId);
  }

  function finishStream(assistantId: string, prompt: string, payload: AssistantPayload): void {
    const state = streamRenderRef.current;
    if (state.assistantId !== assistantId) {
      return;
    }

    state.finalPayload = payload;
    state.pendingPrompt = prompt;
    if (state.queue.length === 0) {
      finalizeRenderedStream(assistantId);
    }
  }

  function scheduleNextCharacter(assistantId: string): void {
    const state = streamRenderRef.current;
    if (state.assistantId !== assistantId || state.timerId !== null || state.queue.length === 0) {
      return;
    }

    state.timerId = window.setTimeout(() => {
      state.timerId = null;
      flushNextCharacter(assistantId);
    }, state.cadenceMs);
  }

  function flushNextCharacter(assistantId: string): void {
    const state = streamRenderRef.current;
    if (state.assistantId !== assistantId) {
      return;
    }

    const nextCharacter = state.queue.shift();
    if (!nextCharacter) {
      if (state.finalPayload) {
        finalizeRenderedStream(assistantId);
      }
      return;
    }

    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? { ...message, content: `${message.content}${nextCharacter}`, status: "streaming" }
          : message,
      ),
    );

    if (state.queue.length > 0) {
      scheduleNextCharacter(assistantId);
      return;
    }

    if (state.finalPayload) {
      finalizeRenderedStream(assistantId);
    }
  }

  function finalizeRenderedStream(assistantId: string): void {
    const state = streamRenderRef.current;
    if (state.assistantId !== assistantId || !state.finalPayload) {
      return;
    }

    const payload = state.finalPayload;
    const prompt = state.pendingPrompt;
    const payloadAnswer = payload.answer || "";
    const streamedAnswer = state.streamedAnswer || "";
    const finalAnswer =
      streamedAnswer.length >= payloadAnswer.length ? streamedAnswer : payloadAnswer;
    clearStreamTimer();
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId ? { ...message, content: finalAnswer, status: "ready", payload: { ...payload, answer: finalAnswer } } : message,
      ),
    );
    if (prompt) {
      setHistory((current) => [...current, { user: prompt, assistant: finalAnswer }]);
    }
    setDetailsPayload({ ...payload, answer: finalAnswer });
    setRequestStage(null);
    setIsSending(false);
    void refreshSidebarData();
    streamRenderRef.current = {
      assistantId: null,
      queue: [],
      timerId: null,
      cadenceMs: 18,
      lastChunkAt: null,
      finalPayload: null,
      pendingPrompt: null,
      streamedAnswer: "",
    };
  }

  async function handleBuildKnowledgeBase(filesOverride?: File[]): Promise<void> {
    if (isBuilding) {
      return;
    }

    const filesToBuild = filesOverride ?? selectedFiles;
    setBuildStatus(filesToBuild.length ? "已收到文档，正在自动重建知识库..." : "正在重建当前知识库...");
    setIsBuilding(true);
    setPanelMode("settings");
    try {
      const result = await buildKnowledgeBase(filesToBuild);
      setRuntimeSummary(result.runtime);
      setKnowledgeBaseSummary(result.knowledge_base);
      setBuildStatus(result.knowledge_base.status_message ?? "知识库构建完成。");
      setSelectedFiles([]);
    } catch (error) {
      setBuildStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsBuilding(false);
    }
  }

  async function handleRunEvals(): Promise<void> {
    if (isEvaluating) {
      return;
    }
    setIsEvaluating(true);
    setEvalStatus("正在运行本地 Agent 评测...");
    setPanelMode("settings");
    try {
      const result = await runAgentEvals();
      setEvalResult(result);
      setEvalStatus(`评测完成：${result.passed}/${result.case_count} 通过`);
      void refreshSidebarData();
    } catch (error) {
      setEvalStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsEvaluating(false);
    }
  }

  async function handleGenerateEvalDrafts(): Promise<void> {
    if (isGeneratingEvalDrafts) {
      return;
    }
    setIsGeneratingEvalDrafts(true);
    setEvalDraftStatus("正在从最近 trace 生成评测草稿...");
    setPanelMode("settings");
    try {
      const result = await generateEvalDrafts(5);
      setEvalDrafts(result);
      setEvalDraftStatus(`已生成 ${result.count} 条评测草稿，尚未加入正式回归集。`);
      void refreshSidebarData();
    } catch (error) {
      setEvalDraftStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsGeneratingEvalDrafts(false);
    }
  }

  async function handleGenerateCodexHandoff(): Promise<void> {
    if (isGeneratingCodexHandoff) {
      return;
    }
    setIsGeneratingCodexHandoff(true);
    setCodexHandoffStatus("正在生成 Codex handoff...");
    setPanelMode("settings");
    try {
      const result = await generateCodexHandoff(8);
      setCodexHandoff(result);
      setCodexHandoffStatus(`已生成 handoff：${result.failed_count} 个失败 / ${result.trace_count} 条 trace。`);
      void refreshSidebarData();
    } catch (error) {
      setCodexHandoffStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsGeneratingCodexHandoff(false);
    }
  }

  async function handlePreviewWorkflow(): Promise<void> {
    const task = workflowTask.trim();
    if (!task || isPreviewingWorkflow) {
      return;
    }
    setIsPreviewingWorkflow(true);
    setWorkflowStatus("正在预览 Agent 工作流...");
    setPanelMode("settings");
    try {
      const result = await previewAgentWorkflow(task, sessionId);
      setWorkflowPreview(result);
      setWorkflowStatus(`已选择 ${result.agent_path.length} 个 Agent / ${result.tool_candidates.length} 个候选工具。`);
    } catch (error) {
      setWorkflowStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsPreviewingWorkflow(false);
    }
  }

  async function handleCreateGoal(): Promise<void> {
    const objective = goalInput.trim();
    if (!objective || goalBusyId) {
      return;
    }
    setGoalBusyId("create");
    setGoalStatus("正在创建 Agent 目标...");
    setPanelMode("settings");
    try {
      await createAgentGoal(objective, sessionId);
      const result = await fetchAgentGoals();
      setAgentGoals(result);
      setGoalStatus(`已记录目标：${objective}`);
    } catch (error) {
      setGoalStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setGoalBusyId(null);
    }
  }

  async function handleGoalProgress(goal: AgentGoal): Promise<void> {
    if (!goal.id || goalBusyId) {
      return;
    }
    setGoalBusyId(goal.id);
    setGoalStatus("正在追加目标进度...");
    try {
      await appendAgentGoalProgress(goal.id, "已在当前 Agent 工作台继续推进。");
      const result = await fetchAgentGoals();
      setAgentGoals(result);
      setGoalStatus(`已更新目标进度：${goal.id}`);
    } catch (error) {
      setGoalStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setGoalBusyId(null);
    }
  }

  async function handleGoalComplete(goal: AgentGoal): Promise<void> {
    if (!goal.id || goalBusyId) {
      return;
    }
    setGoalBusyId(goal.id);
    setGoalStatus("正在完成 Agent 目标...");
    try {
      await completeAgentGoal(goal.id, "用户在 Agent 工作台标记完成。");
      const result = await fetchAgentGoals();
      setAgentGoals(result);
      setGoalStatus(`已完成目标：${goal.id}`);
      void refreshSidebarData();
    } catch (error) {
      setGoalStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setGoalBusyId(null);
    }
  }

  async function handleRefreshMemoryPatches(): Promise<void> {
    setMemoryPatchStatus("正在读取待审核记忆...");
    try {
      const result = await fetchMemoryPatches();
      setMemoryPatches(result);
      setMemoryPatchStatus(`待审核记忆：${result.count} 条`);
    } catch (error) {
      setMemoryPatchStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleApproveMemoryPatch(patch: MemoryPatch): Promise<void> {
    if (!patch.id || memoryPatchBusyId) {
      return;
    }
    setMemoryPatchBusyId(patch.id);
    setMemoryPatchStatus("正在批准记忆补丁...");
    try {
      await approveMemoryPatch(patch.id);
      const result = await fetchMemoryPatches();
      setMemoryPatches(result);
      setMemoryPatchStatus(`已批准记忆补丁：${patch.id}`);
      void refreshSidebarData();
    } catch (error) {
      setMemoryPatchStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setMemoryPatchBusyId(null);
    }
  }

  async function handleRejectMemoryPatch(patch: MemoryPatch): Promise<void> {
    if (!patch.id || memoryPatchBusyId) {
      return;
    }
    setMemoryPatchBusyId(patch.id);
    setMemoryPatchStatus("正在拒绝记忆补丁...");
    try {
      await rejectMemoryPatch(patch.id);
      const result = await fetchMemoryPatches();
      setMemoryPatches(result);
      setMemoryPatchStatus(`已拒绝记忆补丁：${patch.id}`);
      void refreshSidebarData();
    } catch (error) {
      setMemoryPatchStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setMemoryPatchBusyId(null);
    }
  }

  function handleKnowledgeBaseFilesChange(files: FileList | null): void {
    const pickedFiles = Array.from(files ?? []);
    setSelectedFiles(pickedFiles);
    if (pickedFiles.length > 0) {
      void handleBuildKnowledgeBase(pickedFiles);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSubmit();
    }
  }

  const openPanel = useCallback((mode: Exclude<PanelMode, "hidden">): void => {
    setPanelMode(mode);
  }, []);

  const resetConversation = useCallback((): void => {
    setMessages([]);
    setHistory([]);
    setSessionId(createChatSessionId());
    setInput("");
    setDetailsPayload(EMPTY_DETAILS);
    setApprovedPreviewIds([]);
    setPanelMode("hidden");
  }, []);

  const handleCopyMessage = useCallback(async (message: MessageRecord): Promise<void> => {
    const content = getDisplayContent(message).trim();
    if (!content) {
      return;
    }

    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(message.id);
    } catch {
      setInput(content);
      setCopiedMessageId(message.id);
    }
  }, []);

  const handleMessageUtilityAction = useCallback((message: MessageRecord): void => {
    if (message.role === "assistant") {
      if (message.payload) {
        setDetailsPayload(message.payload);
      }
      openPanel("details");
      return;
    }
    setInput(message.content);
  }, [openPanel]);

  return (
    <div className={`app-container theme-${theme} ${hasConversation ? "state-active" : "state-idle"}`}>
      <a className="ui-switch-link" href="http://127.0.0.1:7860" title="切换到备用前端">
        备用前端
      </a>

      <aside className="left-sidebar">
        <div className="sidebar-header">
          <div className="brand">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
            DAC-3D
          </div>
          <button className="theme-toggle" onClick={toggleThemeMode} title="切换显示模式">
            <span className="theme-toggle-icon">{theme === "light" ? "☼" : "☽"}</span>
            <span className="theme-toggle-label">{themeMode === "auto" ? "自动" : theme === "light" ? "日间" : "夜间"}</span>
          </button>
        </div>
        <div className="sidebar-actions">
          <button className="btn-new-chat" onClick={resetConversation}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
            新建会话
          </button>
        </div>
        <Dac3dRuntimeCard runtime={dac3dRuntime} onRefresh={() => void refreshSidebarData()} />
        <nav className="sidebar-nav">
          <button className={`nav-item ${panelMode === "details" ? "active" : ""}`} onClick={() => openPanel("details")}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            检测详情
          </button>
          <button className={`nav-item ${panelMode === "settings" ? "active" : ""}`} onClick={() => openPanel("settings")}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
            系统设置
          </button>
        </nav>
        <div className="sidebar-footer">
          <div className="status-indicator">
            <span className="status-dot"></span>
            系统在线
          </div>
        </div>
      </aside>

      <main className="main-content">
        {!hasConversation ? (
          <div className="hero-section">
            <div className="hero-logo">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
            </div>
            <h1 className="hero-title">有什么我可以帮您的？</h1>
            <div className="hero-suggestions">
              <button className="suggestion-card" onClick={() => setInput("扫描 10mm × 10mm 区域")}>
                <div className="suggestion-title">执行扫描</div>
                <div className="suggestion-desc">扫描 10mm × 10mm 区域</div>
              </button>
              <button className="suggestion-card" onClick={() => setInput("这个参数是什么意思？")}>
                <div className="suggestion-title">解释参数</div>
                <div className="suggestion-desc">了解当前设置的含义</div>
              </button>
              <button className="suggestion-card" onClick={() => setInput("当前检测状态是什么？")}>
                <div className="suggestion-title">查询状态</div>
                <div className="suggestion-desc">获取最新检测进度</div>
              </button>
              <button className="suggestion-card" onClick={() => setInput("样品太反光了应该怎么办？")}>
                <div className="suggestion-title">操作指导</div>
                <div className="suggestion-desc">解决反光等常见问题</div>
              </button>
            </div>
          </div>
        ) : (
          <div className="chat-log">
            {messages.map((message) => (
              <MessageRow
                key={message.id}
                message={message}
                copiedMessageId={copiedMessageId}
                requestStage={requestStage}
                approvalInFlight={approvalInFlight}
                approvedPreviewIds={approvedPreviewIds}
                onCopy={handleCopyMessage}
                onAction={handleMessageUtilityAction}
                onApproveCommand={handleApproveCommand}
              />
            ))}
            <div ref={endRef} />
          </div>
        )}

        <div className="composer-container">
          <form className="composer-form" onSubmit={(event) => void handleSubmit(event)}>
            <textarea
              className="composer-input"
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="给 DAC-3D 助手发送消息…"
              rows={1}
              value={input}
            />
            <button
              className="btn-submit"
              disabled={isSending || !input.trim()}
              type="submit"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
          </form>
          <div className="composer-footer">
            助手可能会产生不准确的信息，请以实际检测结果为准。
          </div>
        </div>
      </main>

      <div
        className={`panel-scrim ${panelMode !== "hidden" ? "open" : ""}`}
        onClick={() => setPanelMode("hidden")}
        role="presentation"
      />

      <aside className={`side-panel ${panelMode !== "hidden" ? "open" : ""}`}>
        <div className="panel-header">
          <h2>{panelMode === "settings" ? "系统设置" : "检测详情"}</h2>
          <button className="btn-close" onClick={() => setPanelMode("hidden")}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          </button>
        </div>

        {panelMode === "details" ? (
          <div className="panel-body" key="details-panel">
            <section className="data-section">
              <h3>参考来源</h3>
              {detailsPayload.sources.length ? (
                <div className="source-list">
                  {detailsPayload.sources.map((source, index) => (
                    <div className="source-card" key={`${source.source}-${index}`}>
                      <div className="source-file">{source.source}</div>
                      <div className="source-meta">
                        <span>章节：{source.section || "未标注"}</span>
                        <span>类型：{source.document_type || "未知"}</span>
                        <span>相关度：{formatScore(source.score)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="data-row">
                  <span className="data-label">状态</span>
                  <span className="data-value">暂无检索证据</span>
                </div>
              )}
            </section>

            <section className="data-section">
              <h3>当前结果</h3>
              <div className="data-grid">
                <DetailRow label="意图" value={detailsPayload.intent || "未知"} />
                <DetailRow label="命令预览" value={formatPresence(detailsPayload.command_preview)} />
                <DetailRow label="状态摘要" value={formatPresence(detailsPayload.status_summary)} />
                <DetailRow label="解析结果" value={formatPresence(detailsPayload.parsed_result)} />
              </div>
            </section>

            <section className="data-section">
              <h3>结构化数据</h3>
              <details className="json-block" open>
                <summary>命令预览</summary>
                <pre>{formatJson(detailsPayload.command_preview)}</pre>
              </details>
              <details className="json-block">
                <summary>状态摘要</summary>
                <pre>{formatJson(detailsPayload.status_summary)}</pre>
              </details>
              <details className="json-block">
                <summary>解析结果</summary>
                <pre>{formatJson(detailsPayload.parsed_result)}</pre>
              </details>
            </section>
          </div>
        ) : null}

        {panelMode === "settings" ? (
          <div className="panel-body" key="settings-panel">
            <section className="data-section">
              <h3>运行时概览</h3>
              <div className="data-grid">
                <DetailRow label="模型提供方" value={stringValue(runtimeSummary?.provider)} />
                <DetailRow label="模型名称" value={stringValue(runtimeSummary?.model)} />
                <DetailRow label="检索 Top-K" value={stringValue(runtimeSummary?.retrieval_top_k)} />
                <DetailRow label="向量库" value={stringValue(runtimeSummary?.vector_store)} />
                <DetailRow
                  label="知识库规模"
                  value={`${stringValue(runtimeSummary?.document_count)} 份文档 / ${stringValue(runtimeSummary?.chunk_count)} 个分块`}
                />
                <DetailRow label="最近构建" value={formatTimestamp(runtimeSummary?.latest_build_at)} />
                <DetailRow label="模拟模式" value={runtimeSummary?.mock_mode ? "已开启" : "未开启"} />
              </div>
            </section>

            <section className="data-section">
              <Dac3dRuntimeDetails runtime={dac3dRuntime} onRefresh={() => void refreshSidebarData()} />
            </section>

            <section className="data-section">
              <h3>Agent 工作区</h3>
              <div className="data-grid">
                <DetailRow label="入口 Agent" value={agentWorkspaceView.entryAgent} />
                <DetailRow label="专家 Agent" value={`${agentWorkspaceView.specialistCount}`} />
                <DetailRow label="技能数量" value={`${agentWorkspaceView.skillCount}`} />
                <DetailRow label="Context 节点" value={`${agentWorkspaceView.contextNodeCount}`} />
                <DetailRow label="记忆 Trace" value={`${agentWorkspaceView.memoryTraceCount}`} />
                <DetailRow label="Agent 目标" value={`${agentWorkspaceView.goalCount}`} />
              </div>
              {agentWorkspaceView.agentNames.length ? (
                <div className="agent-chip-list">
                  {agentWorkspaceView.agentNames.map((name) => (
                    <span className="agent-chip" key={name}>{name}</span>
                  ))}
                </div>
              ) : null}
              {agentWorkspaceView.contextKinds.length ? (
                <div className="context-kind-list">
                  {agentWorkspaceView.contextKinds.map((item) => (
                    <span key={item.name}>{item.name}: {item.count}</span>
                  ))}
                </div>
              ) : null}
              <div className="workflow-preview-form">
                <input
                  aria-label="Agent 工作流任务"
                  onChange={(event) => setWorkflowTask(event.target.value)}
                  value={workflowTask}
                />
                <button
                  className="btn-run-evals"
                  disabled={isPreviewingWorkflow || !workflowTask.trim()}
                  onClick={() => void handlePreviewWorkflow()}
                  type="button"
                >
                  {isPreviewingWorkflow ? "预览中..." : "预览工作流"}
                </button>
              </div>
              {workflowStatus ? <div className="status-msg">{workflowStatus}</div> : null}
              {workflowPreview ? (
                <div className="workflow-preview-card">
                  <div className="workflow-path">
                    {workflowPreview.agent_path.map((agent, index) => (
                      <span key={`${agent}-${index}`}>{agent}</span>
                    ))}
                  </div>
                  <div className="workflow-node-grid">
                    {workflowPreview.nodes.map((node) => (
                      <div className={`workflow-node ${node.status}`} key={node.id}>
                        <strong>{node.label}</strong>
                        <span>{node.kind}{typeof node.count === "number" ? ` / ${node.count}` : ""}</span>
                      </div>
                    ))}
                  </div>
                  <div className="workflow-tool-list">
                    {workflowPreview.tool_candidates.map((tool) => (
                      <span key={tool}>{tool}</span>
                    ))}
                  </div>
                  {workflowPreview.context_tree_matches.length ? (
                    <div className="workflow-match-list">
                      {workflowPreview.context_tree_matches.slice(0, 4).map((match, index) => {
                        const node = asRecord(match.node);
                        return (
                          <div className="workflow-match-item" key={`${stringValue(node?.id)}-${index}`}>
                            <strong>{stringValue(node?.title)}</strong>
                            <span>
                              {stringValue(node?.kind)} / {formatScore(Number.isFinite(Number(match.score)) ? Number(match.score) : undefined)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </section>

            <section className="data-section">
              <h3>Agent 目标</h3>
              <div className="goal-create-form">
                <input
                  aria-label="Agent 目标"
                  onChange={(event) => setGoalInput(event.target.value)}
                  value={goalInput}
                />
                <button
                  className="btn-run-evals"
                  disabled={Boolean(goalBusyId) || !goalInput.trim()}
                  onClick={() => void handleCreateGoal()}
                  type="button"
                >
                  {goalBusyId === "create" ? "记录中..." : "记录目标"}
                </button>
              </div>
              {goalStatus ? <div className="status-msg">{goalStatus}</div> : null}
              {agentGoals?.goals.length ? (
                <div className="goal-list">
                  {agentGoals.goals.map((goal) => (
                    <div className="goal-item" key={goal.id}>
                      <div className="goal-meta">
                        <span>{goal.status || "active"}</span>
                        <span>{goal.session_id || "web"}</span>
                      </div>
                      <strong>{goal.objective}</strong>
                      <small>
                        {goal.progress?.length ? `${goal.progress.length} 条进度` : "暂无进度"} / {formatTimestamp(goal.updated_at)}
                      </small>
                      <div className="goal-actions">
                        <button
                          disabled={Boolean(goalBusyId)}
                          onClick={() => void handleGoalProgress(goal)}
                          type="button"
                        >
                          {goalBusyId === goal.id ? "处理中..." : "追加进度"}
                        </button>
                        <button
                          disabled={Boolean(goalBusyId)}
                          onClick={() => void handleGoalComplete(goal)}
                          type="button"
                        >
                          完成
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-note">暂无活跃目标。</div>
              )}
            </section>

            <section className="data-section">
              <h3>Tool Gateway 风险提示</h3>
              <div className="data-grid">
                <DetailRow label="网关状态" value={toolGateway.enabled ? "已开启" : "未开启"} />
                <DetailRow label="工具数量" value={`${toolGateway.toolCount}`} />
                <DetailRow label="工作流" value={toolGateway.workflow} />
                <DetailRow label="提示策略" value={toolGateway.metadataPolicy} />
                <DetailRow label="策略引擎" value={toolGateway.policyEngineEnabled ? "已开启" : "未开启"} />
                <DetailRow label="策略模式" value={toolGateway.policyMode} />
                <DetailRow label="执行约束" value={toolGateway.policyEnforcement} />
              </div>
              {toolGateway.tools.length ? (
                <div className="tool-hint-list">
                  {toolGateway.tools.map((tool) => (
                    <div className="tool-hint-item" key={tool.name}>
                      <div className="tool-hint-title">
                        <strong>{tool.name}</strong>
                        <span>{tool.riskLevel}</span>
                      </div>
                      <div className="tool-hint-flags">
                        <span className={tool.readOnlyHint ? "active" : ""}>
                          readOnly {tool.readOnlyHint ? "是" : "否"}
                        </span>
                        <span className={tool.destructiveHint ? "danger" : ""}>
                          destructive {tool.destructiveHint ? "是" : "否"}
                        </span>
                        <span className={tool.idempotentHint ? "active" : ""}>
                          idempotent {tool.idempotentHint ? "是" : "否"}
                        </span>
                        <span className={tool.openWorldHint ? "danger" : ""}>
                          openWorld {tool.openWorldHint ? "是" : "否"}
                        </span>
                        <span className={tool.requiresConfirmation ? "danger" : ""}>
                          confirm {tool.requiresConfirmation ? "是" : "否"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
            </section>

            <section className="data-section">
              <h3>Agent 评测</h3>
              <div className="eval-summary-card">
                <div>
                  <span>本地回归集</span>
                  <strong>
                    {evalResult
                      ? `${evalResult.passed}/${evalResult.case_count} 通过`
                      : "尚未运行"}
                  </strong>
                </div>
                <button
                  className="btn-run-evals"
                  disabled={isEvaluating}
                  onClick={() => void handleRunEvals()}
                  type="button"
                >
                  {isEvaluating ? "评测中…" : "运行评测"}
                </button>
              </div>
              {evalStatus ? <div className="status-msg">{evalStatus}</div> : null}
              {evalResult ? (
                <div className="eval-result-list">
                  {evalResult.results.map((result) => (
                    <div className={`eval-result-item ${result.passed ? "passed" : "failed"}`} key={result.id}>
                      <span>{result.category}</span>
                      <strong>{result.id}</strong>
                    </div>
                  ))}
                </div>
              ) : null}
              <div className="eval-draft-card">
                <div>
                  <span>Trace 评测草稿</span>
                  <strong>{evalDrafts ? `${evalDrafts.count} 条` : "未生成"}</strong>
                </div>
                <button
                  className="btn-run-evals"
                  disabled={isGeneratingEvalDrafts}
                  onClick={() => void handleGenerateEvalDrafts()}
                  type="button"
                >
                  {isGeneratingEvalDrafts ? "生成中…" : "生成草稿"}
                </button>
              </div>
              {evalDraftStatus ? <div className="status-msg">{evalDraftStatus}</div> : null}
              {evalDrafts?.drafts.length ? (
                <div className="eval-draft-list">
                  {evalDrafts.drafts.slice(-5).map((item) => (
                    <div className="eval-draft-item" key={item.draft.id}>
                      <span>{item.draft.category}</span>
                      <strong>{item.draft.id}</strong>
                      <p>{item.draft.input}</p>
                    </div>
                  ))}
                </div>
              ) : null}
              <div className="eval-draft-card">
                <div>
                  <span>Codex Handoff</span>
                  <strong>{codexHandoff ? `${codexHandoff.failed_count} 失败` : "未生成"}</strong>
                </div>
                <button
                  className="btn-run-evals"
                  disabled={isGeneratingCodexHandoff}
                  onClick={() => void handleGenerateCodexHandoff()}
                  type="button"
                >
                  {isGeneratingCodexHandoff ? "生成中…" : "生成 Handoff"}
                </button>
              </div>
              {codexHandoffStatus ? <div className="status-msg">{codexHandoffStatus}</div> : null}
              {codexHandoff ? (
                <div className="eval-draft-list">
                  <div className="eval-draft-item">
                    <span>{codexHandoff.workflow || "codex_handoff"}</span>
                    <strong>{codexHandoff.path}</strong>
                    <p>{codexHandoff.recommendations?.[0]?.title || "当前评测通过，可从最近 trace 扩展回归覆盖。"}</p>
                  </div>
                </div>
              ) : null}
            </section>

            <section className="data-section">
              <h3>记忆补丁审核</h3>
              <div className="memory-review-header">
                <div>
                  <span>待审核候选</span>
                  <strong>{memoryPatches ? `${memoryPatches.count} 条` : "未读取"}</strong>
                </div>
                <button
                  className="btn-run-evals"
                  disabled={Boolean(memoryPatchBusyId)}
                  onClick={() => void handleRefreshMemoryPatches()}
                  type="button"
                >
                  刷新
                </button>
              </div>
              {memoryPatchStatus ? <div className="status-msg">{memoryPatchStatus}</div> : null}
              {memoryPatches?.patches.length ? (
                <div className="memory-patch-list">
                  {memoryPatches.patches.map((patch) => (
                    <div className="memory-patch-item" key={patch.id}>
                      <div className="memory-patch-meta">
                        <span>{patch.target || "memory"}</span>
                        <span>{patch.topic || patch.status || "pending"}</span>
                      </div>
                      <p>{patch.content || "空记忆补丁"}</p>
                      <small>{patch.reason || patch.source_trace_id || patch.id}</small>
                      <div className="memory-patch-actions">
                        <button
                          disabled={Boolean(memoryPatchBusyId)}
                          onClick={() => void handleApproveMemoryPatch(patch)}
                          type="button"
                        >
                          {memoryPatchBusyId === patch.id ? "处理中..." : "批准"}
                        </button>
                        <button
                          disabled={Boolean(memoryPatchBusyId)}
                          onClick={() => void handleRejectMemoryPatch(patch)}
                          type="button"
                        >
                          拒绝
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-note">暂无待审核记忆。</div>
              )}
            </section>

            <section className="data-section">
              <h3>知识库构建</h3>
              <div className="upload-area">
                <label className="upload-label">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                  <span>选择文档后自动构建</span>
                  <input
                    multiple
                    onChange={(event) => {
                      handleKnowledgeBaseFilesChange(event.target.files);
                      event.currentTarget.value = "";
                    }}
                    type="file"
                  />
                </label>
              </div>
              {selectedFiles.length ? (
                <div className="file-list">
                  {selectedFiles.map((file) => (
                    <span className="file-chip" key={file.name}>
                      {file.name}
                    </span>
                  ))}
                </div>
              ) : null}
              <button
                className="btn-build"
                disabled={isBuilding}
                onClick={() => void handleBuildKnowledgeBase()}
                type="button"
              >
                {isBuilding ? "正在构建…" : "手动重建当前知识库"}
              </button>
              {buildStatus ? <div className="status-msg">{buildStatus}</div> : null}
            </section>

            <section className="data-section">
              <h3>知识库摘要</h3>
              <div className="data-grid">
                <DetailRow label="文档数量" value={stringValue(knowledgeBaseSummary?.document_count)} />
                <DetailRow label="分块数量" value={stringValue(knowledgeBaseSummary?.chunk_count)} />
                <DetailRow label="最近构建" value={formatTimestamp(knowledgeBaseSummary?.latest_build_at)} />
                <DetailRow label="存储后端" value={stringValue(knowledgeBaseSummary?.storage_backend)} />
              </div>
              {knowledgeBaseSummary?.documents?.length ? (
                <div className="file-list" style={{ marginTop: '12px' }}>
                  {knowledgeBaseSummary.documents.slice(0, 8).map((name) => (
                    <span className="file-chip" key={name}>
                      {name}
                    </span>
                  ))}
                </div>
              ) : null}
            </section>
          </div>
        ) : null}
      </aside>
    </div>
  );
}

type Dac3dRuntimeView = {
  state: string;
  stateLabel: string;
  stateClassName: string;
  progress: number;
  message: string;
  step: string;
  mode: string;
  endpoint: string;
  source: string;
  updatedAt: string;
  trayId: string;
  offline: string;
  lastCommandAction: string;
  latestResult: string;
  resultHistoryCount: number;
  rawStatus: Record<string, unknown> | null;
  rawRuntime: Record<string, unknown> | null;
};

function Dac3dRuntimeCard(props: { runtime: Dac3dRuntimeView; onRefresh: () => void }) {
  const { runtime, onRefresh } = props;
  return (
    <section className={`dac3d-runtime-card state-${runtime.stateClassName}`} aria-label="DAC-3D 运行状态">
      <div className="dac3d-runtime-topline">
        <span>DAC-3D 运行状态</span>
        <button className="btn-runtime-refresh" onClick={onRefresh} title="刷新 DAC-3D 状态" type="button">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"></path></svg>
        </button>
      </div>
      <div className="dac3d-runtime-state">
        <span className="runtime-state-dot" />
        <strong>{runtime.stateLabel}</strong>
        <span>{runtime.progress}%</span>
      </div>
      <div className="runtime-progress-track">
        <span style={{ width: `${runtime.progress}%` }} />
      </div>
      <p className="runtime-message">{runtime.message}</p>
      <div className="runtime-meta-line">
        <span>{runtime.mode}</span>
        <span>{runtime.updatedAt}</span>
      </div>
    </section>
  );
}

function Dac3dRuntimeDetails(props: { runtime: Dac3dRuntimeView; onRefresh: () => void }) {
  const { runtime, onRefresh } = props;
  return (
    <>
      <div className="section-title-row">
        <h3>DAC-3D 运行状态</h3>
        <button className="btn-panel-refresh" onClick={onRefresh} type="button">
          刷新
        </button>
      </div>
      <div className="data-grid">
        <DetailRow label="状态" value={`${runtime.stateLabel} / ${runtime.progress}%`} />
        <DetailRow label="阶段" value={runtime.step} />
        <DetailRow label="消息" value={runtime.message} />
        <DetailRow label="更新时间" value={runtime.updatedAt} />
        <DetailRow label="来源" value={runtime.source} />
        <DetailRow label="桥接模式" value={runtime.mode} />
        <DetailRow label="端点" value={runtime.endpoint} />
        <DetailRow label="托盘/批次" value={runtime.trayId} />
        <DetailRow label="离线模式" value={runtime.offline} />
        <DetailRow label="最近命令" value={runtime.lastCommandAction} />
        <DetailRow label="最新结果" value={runtime.latestResult} />
        <DetailRow label="历史结果" value={`${runtime.resultHistoryCount} 条`} />
      </div>
      <details className="json-block runtime-json-block">
        <summary>原始 DAC-3D 状态 JSON</summary>
        <pre>{formatJson(runtime.rawStatus ?? runtime.rawRuntime)}</pre>
      </details>
    </>
  );
}

function DetailRow(props: { label: string; value: string }) {
  return (
    <div className="data-row">
      <span className="data-label">{props.label}</span>
      <span className="data-value">{props.value}</span>
    </div>
  );
}

const MessageRow = memo(function MessageRow(props: {
  message: MessageRecord;
  copiedMessageId: string | null;
  requestStage: string | null;
  approvalInFlight: string | null;
  approvedPreviewIds: string[];
  onCopy: (message: MessageRecord) => Promise<void>;
  onAction: (message: MessageRecord) => void;
  onApproveCommand: (message: MessageRecord) => Promise<void>;
}) {
  const {
    message,
    copiedMessageId,
    requestStage,
    approvalInFlight,
    approvedPreviewIds,
    onCopy,
    onAction,
    onApproveCommand,
  } = props;
  const displayContent = getDisplayContent(message);
  const approvalCandidate =
    message.role === "assistant" && message.status === "ready" ? getApprovalRequest(message.payload) : null;
  const approval =
    approvalCandidate && !approvedPreviewIds.includes(approvalCandidate.previewId) ? approvalCandidate : null;

  return (
    <div
      className={`log-entry role-${message.role} ${message.status === "error" ? "status-error" : ""} ${message.status === "streaming" ? "status-streaming" : ""}`}
    >
      <div className="log-avatar">
        {message.role === "user" ? (
          <div className="avatar user-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
          </div>
        ) : (
          <div className="avatar assistant-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
          </div>
        )}
      </div>
      <div className="log-body">
        <div className="log-meta">
          <span className="log-role">{message.role === "user" ? "您" : "DAC-3D 助手"}</span>
        </div>
        {message.role === "assistant" && message.status === "streaming" && requestStage ? (
          <div className="request-stage">
            <span className="request-stage-dot" />
            <span>{requestStage}</span>
          </div>
        ) : null}
        <div className="log-content">
          <MessageContent
            content={displayContent || "正在准备响应…"}
            renderMode="rich"
            showCursor={message.status === "streaming" && displayContent.length > 0}
          />
        </div>
        {message.role === "assistant" && message.payload ? <MessageFooter payload={message.payload} /> : null}
        {approval ? (
            <CommandApprovalCard
              approval={approval}
              disabled={Boolean(approvalInFlight) || approval.expired}
              isSubmitting={approvalInFlight === approval.previewId}
              onApprove={() => void onApproveCommand(message)}
            />
        ) : null}
        <div className="log-actions">
          <button onClick={() => void onCopy(message)} title="复制内容">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            {copiedMessageId === message.id ? "已复制" : "复制"}
          </button>
          <button onClick={() => onAction(message)} title={message.role === "assistant" ? "查看详情" : "再次使用"}>
            {message.role === "assistant" ? (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                详情
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10"></polyline><polyline points="23 20 23 14 17 14"></polyline><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"></path></svg>
                重用
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}, areMessageRowPropsEqual);

function getDisplayContent(message: MessageRecord): string {
  return message.content || message.payload?.answer || "";
}

type ApprovalRequest = {
  previewId: string;
  confirmationToken: string;
  action: string;
  message: string;
  expiresAt: string;
  ttlSeconds: number | null;
  lifecycleState: string;
  expired: boolean;
};

function getApprovalRequest(payload?: AssistantPayload): ApprovalRequest | null {
  const parsedResult = asRecord(payload?.parsed_result);
  const gatewayResult = asRecord(parsedResult?.tool_gateway);
  if (gatewayResult?.tool === "submit_command" || gatewayResult?.blocked === true) {
    return null;
  }
  const validation = asRecord(gatewayResult?.validation);
  if (validation?.can_submit === false) {
    return null;
  }

  const commandPreview = asRecord(payload?.command_preview);
  const gateway = asRecord(commandPreview?.gateway);
  if (!gateway || gateway.confirmation_required !== true) {
    return null;
  }

  const previewId = stringValue(gateway.preview_id);
  const confirmationToken = stringValue(gateway.confirmation_token);
  if (previewId === "未知" || confirmationToken === "未知") {
    return null;
  }

  return {
    previewId,
    confirmationToken,
    action: stringValue(commandPreview?.action),
    expiresAt: stringValue(gateway.confirmation_expires_at),
    ttlSeconds:
      typeof gateway.confirmation_ttl_seconds === "number"
        ? gateway.confirmation_ttl_seconds
        : Number.isFinite(Number(gateway.confirmation_ttl_seconds))
          ? Number(gateway.confirmation_ttl_seconds)
          : null,
    lifecycleState: stringValue(gateway.lifecycle_state),
    expired: isPastTimestamp(gateway.confirmation_expires_at),
    message:
      stringValue(gateway.confirmation_message) === "未知"
        ? "该命令需要人工批准后才能继续下发。"
        : stringValue(gateway.confirmation_message),
  };
}

function CommandApprovalCard(props: {
  approval: ApprovalRequest;
  disabled: boolean;
  isSubmitting: boolean;
  onApprove: () => void;
}) {
  const { approval, disabled, isSubmitting, onApprove } = props;
  return (
    <div className="command-approval-card">
      <div className="command-approval-copy">
        <strong>等待批准</strong>
        <span>{approval.action}</span>
        <p>{approval.message}</p>
        <div className="command-approval-meta">
          <span>状态 {approval.lifecycleState}</span>
          {approval.ttlSeconds !== null ? <span>TTL {approval.ttlSeconds}s</span> : null}
          <span>过期 {formatTimestamp(approval.expiresAt)}</span>
        </div>
      </div>
      <button
        className="btn-approve-command"
        disabled={disabled}
        onClick={onApprove}
        type="button"
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 6 9 17l-5-5" />
        </svg>
        {approval.expired ? "已过期" : isSubmitting ? "提交中…" : "批准执行"}
      </button>
    </div>
  );
}

function MessageContent(props: { content: string; renderMode: "plain" | "rich"; showCursor: boolean }) {
  const richContent = useMemo(() => renderRichText(props.content, props.showCursor), [props.content, props.showCursor]);
  if (props.renderMode === "plain") {
    return (
      <span>
        {props.content}
        {props.showCursor ? <span className="cursor-blink" /> : null}
      </span>
    );
  }
  return <>{richContent}</>;
}

function areMessageRowPropsEqual(
  previous: {
    message: MessageRecord;
  copiedMessageId: string | null;
  requestStage: string | null;
  approvalInFlight: string | null;
  approvedPreviewIds: string[];
  onCopy: (message: MessageRecord) => Promise<void>;
  onAction: (message: MessageRecord) => void;
  onApproveCommand: (message: MessageRecord) => Promise<void>;
  },
  next: {
    message: MessageRecord;
    copiedMessageId: string | null;
    requestStage: string | null;
    approvalInFlight: string | null;
    approvedPreviewIds: string[];
    onCopy: (message: MessageRecord) => Promise<void>;
    onAction: (message: MessageRecord) => void;
    onApproveCommand: (message: MessageRecord) => Promise<void>;
  },
): boolean {
  if (previous.message !== next.message) {
    return false;
  }
  if (
    previous.onCopy !== next.onCopy ||
    previous.onAction !== next.onAction ||
    previous.onApproveCommand !== next.onApproveCommand
  ) {
    return false;
  }
  if (previous.approvalInFlight !== next.approvalInFlight) {
    return false;
  }
  if (previous.approvedPreviewIds !== next.approvedPreviewIds) {
    return false;
  }

  const previousCopied = previous.copiedMessageId === previous.message.id;
  const nextCopied = next.copiedMessageId === next.message.id;
  if (previousCopied !== nextCopied) {
    return false;
  }

  const previousStage = previous.message.role === "assistant" && previous.message.status === "streaming" ? previous.requestStage : null;
  const nextStage = next.message.role === "assistant" && next.message.status === "streaming" ? next.requestStage : null;
  return previousStage === nextStage;
}

function renderRichText(content: string, showCursor: boolean): ReactNode[] {
  const lines = content.split(/\r?\n/);
  const lastVisibleLineIndex = findLastVisibleLineIndex(lines);
  return lines.map((line, index) => {
    const key = `line-${index}`;
    const withCursor = showCursor && index === lastVisibleLineIndex;
    if (!line.trim()) {
      return <div className="rich-spacer" key={key} />;
    }
    if (line.startsWith("### ")) {
      return <h3 className="rich-h3" key={key}>{renderInline(line.slice(4), key, withCursor)}</h3>;
    }
    if (line.startsWith("## ")) {
      return <h2 className="rich-h2" key={key}>{renderInline(line.slice(3), key, withCursor)}</h2>;
    }
    if (line.startsWith("# ")) {
      return <h1 className="rich-h1" key={key}>{renderInline(line.slice(2), key, withCursor)}</h1>;
    }
    if (line.startsWith("> ")) {
      return <blockquote className="rich-quote" key={key}>{renderInline(line.slice(2), key, withCursor)}</blockquote>;
    }
    if (/^[-*]\s+/.test(line)) {
      return <div className="rich-list-item" key={key}>{renderInline(line.replace(/^[-*]\s+/, ""), key, withCursor)}</div>;
    }
    return <p className="rich-p" key={key}>{renderInline(line, key, withCursor)}</p>;
  });
}

function renderInline(text: string, keyPrefix: string, showCursor: boolean): ReactNode[] {
  const nodes = text.split(/(\*\*.*?\*\*)/g).filter(Boolean).map((segment, index) => {
    const key = `${keyPrefix}-${index}`;
    if (segment.startsWith("**") && segment.endsWith("**") && segment.length >= 4) {
      return <strong key={key}>{segment.slice(2, -2)}</strong>;
    }
    return <span key={key}>{segment}</span>;
  });
  if (showCursor) {
    nodes.push(<span className="cursor-blink" key={`${keyPrefix}-cursor`} />);
  }
  return nodes;
}

function findLastVisibleLineIndex(lines: string[]): number {
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    if (lines[index].trim()) {
      return index;
    }
  }
  return -1;
}

function formatJson(value: unknown): string {
  if (value == null) {
    return "暂无数据";
  }
  return JSON.stringify(value, null, 2);
}

function formatTimestamp(value: unknown): string {
  if (!value) {
    return "尚未构建";
  }
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString("zh-CN", { hour12: false });
}

function formatScore(value: number | null | undefined): string {
  if (typeof value !== "number") {
    return "暂无";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatPresence(value: unknown): string {
  return value ? "有" : "无";
}

function buildDac3dRuntimeView(runtimeSummary: RuntimeSummary | null): Dac3dRuntimeView {
  const runtime = asRecord(runtimeSummary?.dac3d);
  const status = asRecord(runtime?.status);
  const resultHistory = Array.isArray(status?.result_history) ? status.result_history : [];
  const latestResult = asRecord(status?.latest_result);
  const progress = normalizeProgress(status?.progress);
  const state = stringValue(status?.state);
  const endpoint = stringValue(runtime?.endpoint);
  return {
    state,
    stateLabel: formatDac3dState(state),
    stateClassName: normalizeStateClass(state),
    progress,
    message: status?.message == null || status.message === "" ? "暂未读取到 DAC-3D 运行状态。" : String(status.message),
    step: stringValue(status?.step),
    mode: stringValue(runtime?.mode ?? status?.mode),
    endpoint,
    source: stringValue(status?.source),
    updatedAt: formatRuntimeTimestamp(status?.updated_at ?? status?.timestamp ?? status?.time),
    trayId: stringValue(status?.tray_id),
    offline: typeof status?.offline === "boolean" ? (status.offline ? "是" : "否") : stringValue(status?.offline),
    lastCommandAction: stringValue(runtime?.last_command_action),
    latestResult: formatLatestResult(latestResult),
    resultHistoryCount: resultHistory.length,
    rawStatus: status,
    rawRuntime: runtime,
  };
}

function buildAgentWorkspaceView(workspace: AgentWorkspace | null): AgentWorkspaceView {
  const skills = asRecord(workspace?.skills);
  const contextTree = asRecord(workspace?.context_tree);
  const memoryOs = asRecord(workspace?.memory_os);
  const goals = asRecord(workspace?.goals);
  const contextKindsRecord = asRecord(contextTree?.kinds);
  const contextKinds = Object.entries(contextKindsRecord ?? {}).map(([name, count]) => ({
    name,
    count: Number(count) || 0,
  }));
  return {
    entryAgent: stringValue(workspace?.entry_agent),
    specialistCount: Array.isArray(workspace?.specialist_agents) ? workspace.specialist_agents.length : 0,
    skillCount: Number(skills?.skill_count ?? 0) || 0,
    contextNodeCount: Number(contextTree?.node_count ?? 0) || 0,
    memoryTraceCount: Number(memoryOs?.trace_count ?? 0) || 0,
    goalCount: Number(goals?.goal_count ?? 0) || 0,
    workflow: Array.isArray(workspace?.workflow) ? workspace.workflow.map(String) : [],
    agentNames: Array.isArray(workspace?.specialist_agents) ? workspace.specialist_agents.map(String) : [],
    contextKinds,
  };
}

function buildToolGatewayView(runtimeSummary: RuntimeSummary | null): ToolGatewayView {
  const agent = asRecord(runtimeSummary?.agent);
  const gateway = asRecord(agent?.tool_gateway);
  const policyEngine = asRecord(gateway?.policy_engine);
  const rawTools = Array.isArray(gateway?.tools) ? gateway.tools : [];
  const tools = rawTools
    .map((item) => asRecord(item))
    .filter((item): item is Record<string, unknown> => Boolean(item))
    .map((item) => ({
      name: stringValue(item.name),
      riskLevel: stringValue(item.risk_level),
      requiresConfirmation: item.requires_confirmation === true,
      readOnlyHint: item.readOnlyHint === true,
      destructiveHint: item.destructiveHint === true,
      idempotentHint: item.idempotentHint === true,
      openWorldHint: item.openWorldHint === true,
    }));
  return {
    enabled: gateway?.enabled === true,
    toolCount: Number(gateway?.tool_count ?? tools.length) || tools.length,
    workflow: stringValue(gateway?.workflow),
    metadataPolicy: stringValue(gateway?.metadata_policy),
    policyEngineEnabled: policyEngine?.enabled === true,
    policyMode: stringValue(policyEngine?.mode),
    policyEnforcement: stringValue(policyEngine?.enforcement),
    tools,
  };
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function isPastTimestamp(value: unknown): boolean {
  const timestamp = stringValue(value);
  if (timestamp === "未知") {
    return false;
  }
  const parsed = Date.parse(timestamp);
  return Number.isFinite(parsed) && parsed <= Date.now();
}

function normalizeProgress(value: unknown): number {
  const numeric = typeof value === "number" ? value : Number(value ?? 0);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return Math.min(100, Math.max(0, Math.round(numeric)));
}

function formatDac3dState(value: string): string {
  const stateMap: Record<string, string> = {
    idle: "空闲",
    queued: "排队中",
    running: "运行中",
    completed: "已完成",
    stopped: "已停止",
    error: "异常",
    unknown: "未知",
  };
  return stateMap[value.toLowerCase()] ?? value;
}

function normalizeStateClass(value: string): string {
  const normalized = value.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return normalized || "unknown";
}

function formatRuntimeTimestamp(value: unknown): string {
  if (!value) {
    return "暂无更新时间";
  }
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString("zh-CN", { hour12: false });
}

function formatLatestResult(value: Record<string, unknown> | null): string {
  if (!value) {
    return "暂无";
  }
  const position = value.pos ?? value.position ?? value.sample_id;
  const quality = value.quality_label ?? value.sample_quality_label ?? value.quality;
  const defects = value.defects_num ?? value.defection_num;
  const parts = [
    position == null ? "" : `位置 ${position}`,
    quality == null ? "" : `判定 ${quality}`,
    defects == null ? "" : `缺陷 ${defects}`,
  ].filter(Boolean);
  return parts.length ? parts.join("，") : "已有最新结果";
}

function stringValue(value: unknown): string {
  if (value == null || value === "") {
    return "未知";
  }
  return String(value);
}

function clampCadence(value: number): number {
  return Math.min(42, Math.max(10, Math.round(value)));
}

function MessageFooter(props: { payload?: AssistantPayload }) {
  const { payload } = props;
  const [isOpen, setIsOpen] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const [activeSourceIndex, setActiveSourceIndex] = useState<number | null>(null);
  const [showAllSources, setShowAllSources] = useState(false);
  const [popoverStyle, setPopoverStyle] = useState<React.CSSProperties>({});
  const triggerRef = useRef<HTMLElement | null>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const closeTimeoutRef = useRef<number | null>(null);

  if (!payload) {
    return null;
  }

  const hasSources = payload.sources && payload.sources.length > 0;
  const hasDetails = payload.command_preview || payload.status_summary || payload.parsed_result;

  if (!hasSources && !hasDetails) {
    return null;
  }

  const updatePosition = () => {
    if (!triggerRef.current || !popoverRef.current) return;
    const triggerRect = triggerRef.current.getBoundingClientRect();
    const popoverRect = popoverRef.current.getBoundingClientRect();
    
    let top = triggerRect.top - popoverRect.height - 12;
    let left = triggerRect.left;
    let transformOrigin = 'bottom left';

    if (top < 12) {
      top = triggerRect.bottom + 12;
      transformOrigin = 'top left';
    }

    const maxLeft = window.innerWidth - popoverRect.width - 12;
    if (left > maxLeft) {
      left = maxLeft;
    }
    if (left < 12) {
      left = 12;
    }

    setPopoverStyle({
      top: `${top}px`,
      left: `${left}px`,
      transformOrigin,
    });
  };

  useEffect(() => {
    let timeoutId: number;
    if (isOpen) {
      setIsMounted(true);
    } else {
      timeoutId = window.setTimeout(() => {
        setIsMounted(false);
      }, 150);
    }
    return () => window.clearTimeout(timeoutId);
  }, [isOpen]);

  useLayoutEffect(() => {
    if (isMounted && isOpen) {
      updatePosition();
    }
  }, [isMounted, isOpen]);

  useEffect(() => {
    if (isOpen && isMounted) {
      window.addEventListener('scroll', updatePosition, true);
      window.addEventListener('resize', updatePosition);
      return () => {
        window.removeEventListener('scroll', updatePosition, true);
        window.removeEventListener('resize', updatePosition);
      };
    }
  }, [isOpen, isMounted]);

  const handleMouseEnter = () => {
    if (closeTimeoutRef.current !== null) {
      window.clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }
    setIsOpen(true);
  };

  const handleMouseLeave = () => {
    closeTimeoutRef.current = window.setTimeout(() => {
      setIsOpen(false);
    }, 100);
  };

  const handleFocus = () => {
    if (closeTimeoutRef.current !== null) {
      window.clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }
    setIsOpen(true);
  };

  const openForSource = (index: number, element: HTMLElement) => {
    triggerRef.current = element;
    setShowAllSources(false);
    setActiveSourceIndex(index);
    handleMouseEnter();
  };

  const openAllSources = (element: HTMLElement) => {
    triggerRef.current = element;
    setShowAllSources(true);
    setActiveSourceIndex(null);
    handleMouseEnter();
  };

  const displayedSources = showAllSources
    ? payload.sources
    : activeSourceIndex != null
      ? payload.sources.filter((_, index) => index === activeSourceIndex)
      : [];

  const handleBlur = (e: React.FocusEvent) => {
    if (
      popoverRef.current?.contains(e.relatedTarget as Node) ||
      triggerRef.current?.contains(e.relatedTarget as Node)
    ) {
      return;
    }
    setIsOpen(false);
  };

  const popoverContent = isMounted ? (
    <div 
      className={`footer-popover ${isOpen ? 'open' : ''}`} 
      ref={popoverRef}
      style={popoverStyle}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      tabIndex={-1}
    >
      {hasSources && displayedSources.length > 0 && (
        <div className="popover-section">
          <div className="popover-title">{showAllSources ? "全部参考来源" : "来源详情"}</div>
          <div className="popover-source-list">
            {displayedSources.map((source, index) => (
              <div className="popover-source-item" key={`${source.source}-${index}`}>
                <div className="popover-source-header">
                  <span className="popover-source-index">{showAllSources ? payload.sources.indexOf(source) + 1 : (activeSourceIndex ?? 0) + 1}</span>
                  <span className="popover-source-name">{source.title || source.source}</span>
                </div>
                <div className="popover-source-meta">
                  {source.section && <span>章节：{source.section}</span>}
                  {source.document_type && <span>类型：{source.document_type}</span>}
                  {source.score != null && <span>相关度：{formatScore(source.score)}</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {hasDetails && (
        <div className="popover-section">
          <div className="popover-title">结构化数据</div>
          <div className="popover-details-grid">
            {payload.command_preview && (
              <div className="popover-detail-item">
                <div className="popover-detail-label">命令预览</div>
                <pre>{formatJson(payload.command_preview)}</pre>
              </div>
            )}
            {payload.status_summary && (
              <div className="popover-detail-item">
                <div className="popover-detail-label">状态摘要</div>
                <pre>{formatJson(payload.status_summary)}</pre>
              </div>
            )}
            {payload.parsed_result && (
              <div className="popover-detail-item">
                <div className="popover-detail-label">解析结果</div>
                <pre>{formatJson(payload.parsed_result)}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  ) : null;

  return (
    <div className="message-footer">
      <div 
        className="footer-interactive-area" 
        tabIndex={0}
        onMouseLeave={handleMouseLeave}
        onBlur={handleBlur}
      >
        <div className="footer-chips">
          {hasSources && payload.sources.map((source, index) => (
            <button
              className="source-chip"
              key={`${source.source}-${index}`}
              onFocus={(event) => openForSource(index, event.currentTarget)}
              onMouseEnter={(event) => openForSource(index, event.currentTarget)}
              type="button"
            >
              <span className="source-index">{index + 1}</span>
              <span className="source-name">{source.title || source.source}</span>
            </button>
          ))}
          {hasSources ? (
            <button
              className="source-chip source-chip-all"
              onFocus={(event) => openAllSources(event.currentTarget)}
              onMouseEnter={(event) => openAllSources(event.currentTarget)}
              type="button"
            >
              <span className="source-name">全部来源</span>
            </button>
          ) : null}
          {hasDetails && !hasSources && (
            <button className="source-chip" onFocus={handleFocus} onMouseEnter={handleMouseEnter} ref={(node) => { triggerRef.current = node; }} type="button">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
              <span className="source-name">结构化数据</span>
            </button>
          )}
        </div>
      </div>
      {popoverContent && createPortal(popoverContent, document.body)}
    </div>
  );
}

export default App;
