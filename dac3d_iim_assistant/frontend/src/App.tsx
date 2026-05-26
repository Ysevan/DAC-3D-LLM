import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { fetchMachineSnapshot, sendMachineAgentChat } from "./api";
import type {
  AlarmRecord,
  MachineAgentPayload,
  MachineMessageRecord,
  MachineSnapshot,
  MachineStatus,
  ToolCallRecord,
} from "./types";

const DEFAULT_DEMO_QUESTIONS = [
  "现在设备状态怎么样？",
  "上个月运行情况怎么样？",
  "最近有哪些异常？",
  "为什么最近温度报警变多了？",
  "E102 错误代码是什么意思？",
  "帮我总结一下这台机器最近三个月的问题。",
];

function App() {
  const [snapshot, setSnapshot] = useState<MachineSnapshot | null>(null);
  const [messages, setMessages] = useState<MachineMessageRecord[]>([
    {
      id: "welcome",
      role: "assistant",
      status: "ready",
      content:
        "工业设备信息管理 AI Agent 已就绪。可以询问实时状态、历史运行、报警原因、错误代码和维护建议。",
    },
  ]);
  const [input, setInput] = useState("现在设备状态怎么样？");
  const [isSending, setIsSending] = useState(false);
  const [selectedPayload, setSelectedPayload] = useState<MachineAgentPayload | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void refreshSnapshot();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const latestPayload = useMemo(() => {
    if (selectedPayload) return selectedPayload;
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const payload = messages[index].payload;
      if (payload) return payload;
    }
    return null;
  }, [messages, selectedPayload]);

  const currentStatus = latestPayload?.current_status ?? snapshot?.current_status ?? null;
  const alarmRecords = latestPayload?.alarm_records?.length
    ? latestPayload.alarm_records
    : snapshot?.alarm_records ?? [];
  const summaryResult = latestPayload?.summary_result ?? snapshot?.summary_result ?? null;
  const abnormalResult = latestPayload?.abnormal_result ?? snapshot?.abnormal_result ?? null;
  const demoQuestions = snapshot?.demo_questions?.length ? snapshot.demo_questions : DEFAULT_DEMO_QUESTIONS;

  async function refreshSnapshot(): Promise<void> {
    try {
      const data = await fetchMachineSnapshot();
      setSnapshot(data);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setMessages((current) => [
        ...current,
        {
          id: `snapshot-error-${Date.now()}`,
          role: "assistant",
          status: "error",
          content: `设备快照加载失败：${message}`,
        },
      ]);
    }
  }

  async function askQuestion(question: string): Promise<void> {
    const prompt = question.trim();
    if (!prompt || isSending) return;

    const timestamp = Date.now();
    const userMessage: MachineMessageRecord = {
      id: `user-${timestamp}`,
      role: "user",
      status: "ready",
      content: prompt,
    };
    const loadingMessage: MachineMessageRecord = {
      id: `assistant-${timestamp}`,
      role: "assistant",
      status: "loading",
      content: "Agent 正在选择工具并读取设备数据...",
    };

    setMessages((current) => [...current, userMessage, loadingMessage]);
    setInput("");
    setIsSending(true);

    try {
      const payload = await sendMachineAgentChat(prompt);
      setSelectedPayload(payload);
      setMessages((current) =>
        current.map((message) =>
          message.id === loadingMessage.id
            ? {
                ...message,
                status: "ready",
                content: payload.answer,
                payload,
              }
            : message
        )
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setMessages((current) =>
        current.map((item) =>
          item.id === loadingMessage.id
            ? { ...item, status: "error", content: `请求失败：${message}` }
            : item
        )
      );
    } finally {
      setIsSending(false);
      void refreshSnapshot();
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void askQuestion(input);
  }

  return (
    <main className="machine-app">
      <section className="workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">Industrial Machine Agent</p>
            <h1>工业设备信息管理 AI Agent</h1>
          </div>
          <button type="button" className="ghost-button" onClick={() => void refreshSnapshot()}>
            刷新状态
          </button>
        </header>

        <div className="main-grid">
          <section className="chat-panel" aria-label="Agent chat">
            <div className="demo-strip">
              {demoQuestions.map((question) => (
                <button
                  key={question}
                  type="button"
                  className="demo-chip"
                  onClick={() => void askQuestion(question)}
                  disabled={isSending}
                >
                  {question}
                </button>
              ))}
            </div>

            <div className="chat-log">
              {messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  onInspect={(payload) => setSelectedPayload(payload)}
                />
              ))}
              <div ref={endRef} />
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="输入设备状态、报警、历史或错误代码问题"
                rows={2}
              />
              <button type="submit" disabled={isSending || !input.trim()}>
                发送
              </button>
            </form>
          </section>

          <aside className="insight-panel" aria-label="Machine insights">
            <StatusCard status={currentStatus} />
            <SummaryCard title="时间段总结" payload={summaryResult} />
            <SummaryCard title="异常模式分析" payload={abnormalResult} />
            <AlarmList alarms={alarmRecords} />
            <ToolTrace toolCalls={latestPayload?.tool_calls ?? []} />
          </aside>
        </div>
      </section>
    </main>
  );
}

function MessageBubble({
  message,
  onInspect,
}: {
  message: MachineMessageRecord;
  onInspect: (payload: MachineAgentPayload) => void;
}) {
  return (
    <article className={`message ${message.role} ${message.status}`}>
      <div className="message-role">{message.role === "user" ? "用户" : "AI Agent"}</div>
      <div className="message-content">{message.content}</div>
      {message.payload && (
        <button type="button" className="inspect-button" onClick={() => onInspect(message.payload!)}>
          查看本轮工具调用
        </button>
      )}
    </article>
  );
}

function StatusCard({ status }: { status: MachineStatus | null }) {
  if (!status) {
    return (
      <section className="panel-card">
        <h2>当前设备状态</h2>
        <p className="muted">等待设备数据...</p>
      </section>
    );
  }

  return (
    <section className="panel-card">
      <div className="card-heading">
        <h2>当前设备状态</h2>
        <span className={`state-badge state-${status.state}`}>{status.state}</span>
      </div>
      <p className="machine-name">{status.machine_name}</p>
      <div className="metric-grid">
        <Metric label="温度" value={`${status.temperature_c}C`} tone={status.temperature_c >= 85 ? "danger" : "normal"} />
        <Metric label="压力" value={`${status.pressure_mpa}MPa`} />
        <Metric label="转速" value={`${status.rpm}rpm`} />
        <Metric label="电流" value={`${status.current_a}A`} tone={status.current_a >= 36 ? "danger" : "normal"} />
        <Metric label="产量" value={`${status.output_count}`} />
        <Metric label="稼动率" value={`${status.utilization_pct}%`} />
      </div>
      <div className="active-alarms">
        <span>未关闭报警</span>
        <strong>{status.active_alarm_codes.length ? status.active_alarm_codes.join(" / ") : "无"}</strong>
      </div>
    </section>
  );
}

function Metric({ label, value, tone = "normal" }: { label: string; value: string; tone?: "normal" | "danger" }) {
  return (
    <div className={`metric metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SummaryCard({ title, payload }: { title: string; payload: Record<string, unknown> | null }) {
  const summaryText = typeof payload?.summary_text === "string" ? payload.summary_text : "";
  return (
    <section className="panel-card">
      <h2>{title}</h2>
      <p className={summaryText ? "summary-text" : "muted"}>{summaryText || "暂无总结。提交 demo 问题后会展示对应分析结果。"}</p>
    </section>
  );
}

function AlarmList({ alarms }: { alarms: AlarmRecord[] }) {
  return (
    <section className="panel-card">
      <div className="card-heading">
        <h2>报警/异常列表</h2>
        <span className="count-badge">{alarms.length}</span>
      </div>
      <div className="alarm-list">
        {alarms.slice(0, 8).map((alarm) => (
          <article key={alarm.alarm_id} className={`alarm-item severity-${alarm.severity}`}>
            <div>
              <strong>{alarm.code}</strong>
              <span>{alarm.alarm_type}</span>
            </div>
            <p>{alarm.message}</p>
            <footer>
              <span>{formatDateTime(alarm.timestamp)}</span>
              <span>{alarm.status}</span>
            </footer>
          </article>
        ))}
        {!alarms.length && <p className="muted">暂无报警记录。</p>}
      </div>
    </section>
  );
}

function ToolTrace({ toolCalls }: { toolCalls: ToolCallRecord[] }) {
  return (
    <section className="panel-card tool-card">
      <div className="card-heading">
        <h2>Agent 工具调用</h2>
        <span className="count-badge">{toolCalls.length}</span>
      </div>
      <div className="tool-list">
        {toolCalls.map((tool, index) => (
          <details key={`${tool.name}-${index}`} className="tool-item" open={index < 3}>
            <summary>
              <span>{index + 1}</span>
              <strong>{tool.name}</strong>
            </summary>
            <p>{tool.reason}</p>
            <pre>{JSON.stringify({ arguments: tool.arguments, result: compactResult(tool.result) }, null, 2)}</pre>
          </details>
        ))}
        {!toolCalls.length && <p className="muted">提问后会显示本轮调用的工具。</p>}
      </div>
    </section>
  );
}

function compactResult(result: Record<string, unknown>): Record<string, unknown> {
  const cloned = { ...result };
  if (Array.isArray(cloned.records) && cloned.records.length > 5) {
    cloned.records = cloned.records.slice(0, 5);
    cloned.records_note = "仅显示前 5 条";
  }
  if (Array.isArray(cloned.documents) && cloned.documents.length > 3) {
    cloned.documents = cloned.documents.slice(0, 3);
  }
  return cloned;
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return `${date.getMonth() + 1}/${date.getDate()} ${date.getHours().toString().padStart(2, "0")}:00`;
}

export default App;
