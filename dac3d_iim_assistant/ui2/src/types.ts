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
  confirmation?: CommandConfirmation | null;
  request_id?: string | null;
  trace_id?: string | null;
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

export interface CommandConfirmation {
  required?: boolean;
  preview_id?: string;
  preview_hash?: string;
  confirmation_token?: string;
  expires_at?: number;
  used?: boolean;
}

export interface MessageRecord {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "ready" | "streaming" | "error";
  payload?: AssistantPayload;
}
