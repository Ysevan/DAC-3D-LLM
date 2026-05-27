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

export interface ChatRequest {
  message: string;
  history: ChatHistoryTurn[];
  session_id?: string;
}

export interface MessageRecord {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "ready" | "streaming" | "error";
  payload?: AssistantPayload;
}

export interface MachineStatus {
  machine_id: string;
  machine_name: string;
  timestamp: string;
  state: string;
  temperature_c: number;
  pressure_mpa: number;
  rpm: number;
  current_a: number;
  output_count: number;
  utilization_pct: number;
  active_alarm_codes: string[];
}

export interface AlarmRecord {
  alarm_id: string;
  timestamp: string;
  code: string;
  alarm_type: string;
  severity: string;
  status: string;
  message: string;
  related_metric: string;
  metric_value: number;
}

export interface ToolCallRecord {
  name: string;
  arguments: Record<string, unknown>;
  reason: string;
  result: Record<string, unknown>;
}

export interface MachineDocumentResult {
  source: string;
  title: string;
  section: string;
  text: string;
  score: number;
}

export interface TimeRangePayload {
  start: string;
  end: string;
  label: string;
}

export interface MachineAgentPayload {
  intent: string;
  answer: string;
  tool_calls: ToolCallRecord[];
  time_range?: TimeRangePayload;
  current_status?: MachineStatus;
  alarm_records?: AlarmRecord[];
  summary_result?: Record<string, unknown> | null;
  abnormal_result?: Record<string, unknown> | null;
  doc_results?: MachineDocumentResult[];
  demo_questions?: string[];
}

export interface MachineSnapshot {
  current_status: MachineStatus;
  alarm_records: AlarmRecord[];
  summary_result: Record<string, unknown>;
  abnormal_result: Record<string, unknown>;
  demo_questions: string[];
  time_range: TimeRangePayload;
}

export interface MachineMessageRecord {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "ready" | "loading" | "error";
  payload?: MachineAgentPayload;
}
