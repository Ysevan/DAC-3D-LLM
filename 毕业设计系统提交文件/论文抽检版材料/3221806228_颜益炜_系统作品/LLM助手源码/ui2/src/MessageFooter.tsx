import { useState } from "react";
import type { AssistantPayload } from "./types";

export function MessageFooter({ payload }: { payload?: AssistantPayload }) {
  if (!payload) return null;

  const hasSources = payload.sources && payload.sources.length > 0;
  const hasDetails = payload.command_preview || payload.status_summary || payload.parsed_result;

  if (!hasSources && !hasDetails) return null;

  return (
    <div className="message-footer">
      {hasSources && (
        <div className="sources-container">
          <div className="sources-label">参考来源：</div>
          <div className="sources-list">
            {payload.sources.map((source, idx) => (
              <div key={idx} className="source-chip" title={source.title || source.source}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                <span>{source.title || source.source}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {hasDetails && (
        <div className="details-popover-trigger">
          <button className="btn-details-trigger">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            <span>结构化数据</span>
          </button>
          <div className="details-popover">
            {payload.command_preview && (
              <div className="popover-section">
                <div className="popover-title">命令预览</div>
                <pre>{JSON.stringify(payload.command_preview, null, 2)}</pre>
              </div>
            )}
            {payload.status_summary && (
              <div className="popover-section">
                <div className="popover-title">状态摘要</div>
                <pre>{JSON.stringify(payload.status_summary, null, 2)}</pre>
              </div>
            )}
            {payload.parsed_result && (
              <div className="popover-section">
                <div className="popover-title">解析结果</div>
                <pre>{JSON.stringify(payload.parsed_result, null, 2)}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
