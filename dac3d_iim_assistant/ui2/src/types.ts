export interface ChatHistoryTurn {
  user: string;
  assistant: string;
}

export interface SourceItem {
  source: string;
  title?: string | null;
  section?: string | null;
  document_type?: string | null;
  score?: number | null;
  chunk_id?: number | null;
}

export interface AssistantPayload {
  intent: string;
  answer: string;
  sources: SourceItem[];
  command_preview: Record<string, unknown> | null;
  status_summary: Record<string, unknown> | null;
  parsed_result: Record<string, unknown> | null;
}

export interface RuntimeSummary {
  provider?: string;
  model?: string;
  retrieval_top_k?: number;
  vector_store?: string;
  document_count?: number;
  chunk_count?: number;
  latest_build_at?: string;
  mock_mode?: boolean;
  dac3d?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AgentWorkspace {
  enabled: boolean;
  backend: string;
  entry_agent?: string;
  specialist_agents?: string[];
  tool_groups?: Record<string, string[]>;
  capabilities?: string[];
  skills?: Record<string, unknown>;
  context_tree?: Record<string, unknown>;
  memory_os?: Record<string, unknown>;
  goals?: Record<string, unknown>;
  workflow?: string[];
}

export interface AgentGoal {
  id: string;
  session_id?: string;
  objective: string;
  status: string;
  source?: string;
  created_at?: string;
  updated_at?: string;
  completed_at?: string;
  progress?: Array<{
    id?: string;
    created_at?: string;
    note?: string;
    evidence?: Record<string, unknown>;
  }>;
  metadata?: Record<string, unknown>;
}

export interface AgentGoalListResult {
  enabled: boolean;
  backend?: string;
  path?: string;
  goals: AgentGoal[];
  count: number;
  total_count?: number;
  workflow?: string;
}

export interface AgentGoalActionResult {
  enabled?: boolean;
  goal: AgentGoal;
  created?: boolean;
  duplicate?: boolean;
  completed?: boolean;
  progress?: Record<string, unknown>;
}

export interface AgentWorkflowPreview {
  enabled: boolean;
  backend: string;
  task: string;
  session_id?: string;
  agent_path: string[];
  tool_candidates: string[];
  skill_matches: Record<string, unknown>[];
  context_tree_matches: Record<string, unknown>[];
  memory_hits: Record<string, unknown>[];
  context_sections: Record<string, unknown>[];
  nodes: Array<{
    id: string;
    label: string;
    kind: string;
    status: string;
    count?: number;
  }>;
  workflow?: string;
}

export interface KnowledgeBaseHistoryItem {
  generated_at?: string;
  trigger?: string;
  document_count?: number;
  chunk_count?: number;
  uploaded_files?: string[];
}

export interface KnowledgeBaseSummary {
  status_message?: string;
  document_count?: number;
  chunk_count?: number;
  latest_build_at?: string;
  storage_backend?: string;
  documents?: string[];
  history?: KnowledgeBaseHistoryItem[];
}

export interface EvalCheck {
  name: string;
  passed: boolean;
  expected?: unknown;
  actual?: unknown;
}

export interface EvalCaseResult {
  id: string;
  category: string;
  input: string;
  passed: boolean;
  checks: EvalCheck[];
  intent?: string;
  answer_preview?: string;
  trace_id?: string | null;
  approval_trace_id?: string | null;
}

export interface EvalRunResult {
  backend: string;
  cases_dir: string;
  case_count: number;
  passed: number;
  failed: number;
  pass_rate: number;
  results: EvalCaseResult[];
  trace_logger?: Record<string, unknown>;
}

export interface EvalDraftItem {
  path: string;
  draft: {
    id: string;
    category: string;
    input: string;
    expected: Record<string, unknown>;
    draft?: boolean;
    source_trace_id?: string;
    generated_at?: string;
    review?: Record<string, unknown>;
    notes?: Record<string, unknown>;
  };
}

export interface EvalDraftListResult {
  enabled: boolean;
  backend: string;
  drafts_dir: string;
  count: number;
  drafts: EvalDraftItem[];
  workflow?: string;
  auto_approved?: boolean;
}

export interface CodexHandoffResult {
  enabled: boolean;
  backend: string;
  path: string;
  generated_at?: string;
  failed_count: number;
  trace_count: number;
  eval_summary?: {
    backend?: string;
    case_count?: number;
    passed?: number;
    failed?: number;
    pass_rate?: number;
  };
  recommendations?: Array<{
    title: string;
    reason?: string;
    target_files?: string[];
  }>;
  failed_evals?: Array<Record<string, unknown>>;
  workflow?: string;
  auto_applied?: boolean;
}

export interface MemoryPatch {
  id: string;
  created_at?: string;
  target?: string;
  topic?: string;
  content?: string;
  mode?: string;
  reason?: string;
  source_trace_id?: string;
  status?: string;
  applied_at?: string;
  rejected_at?: string;
  reject_reason?: string;
  metadata?: Record<string, unknown>;
}

export interface MemoryPatchListResult {
  enabled: boolean;
  backend?: string;
  patches_path?: string;
  patches: MemoryPatch[];
  count: number;
}

export interface MemoryPatchActionResult {
  enabled: boolean;
  patch: MemoryPatch;
  applied?: boolean;
  rejected?: boolean;
  result?: Record<string, unknown>;
  message?: string;
}

export interface ChatRequest {
  message: string;
  history: ChatHistoryTurn[];
  session_id?: string;
}

export interface ApproveCommandRequest {
  session_id: string;
  preview_id?: string;
  confirmation_token?: string;
}

export interface MessageRecord {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "ready" | "streaming" | "error";
  payload?: AssistantPayload;
}
