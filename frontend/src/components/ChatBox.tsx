import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { MessageList, type ChatMessage } from './MessageList';
import { MessageInput } from './MessageInput';
import { useSSE, type PipelineStage } from '../hooks/useSSE';
import type { Mode } from './ModeToggle';

// Token-stream stage names emitted at high frequency bởi rag-engine SSE.
// Persist từng token sẽ vượt 5-10 MB quota của localStorage cho ReAct (5k+ event
// per message). Ta gộp mỗi run liên tiếp cùng stage + cùng iter thành 1 event
// tổng hợp — trail UI vẫn rebuild được vì buildSteps chỉ cộng dồn token vào buffer.
const TOKEN_STREAM_STAGES = new Set([
  'thinking',
  'answer',
  'hop_thinking',
  'hop_planner_token',
  'hop_research_thinking',
  'hop_research_token',
  'graph_cypher_thinking',
  'graph_cypher_token',
]);

function compactStagesForStorage(stages: PipelineStage[]): PipelineStage[] {
  const out: PipelineStage[] = [];
  let i = 0;
  while (i < stages.length) {
    const s = stages[i];
    if (!TOKEN_STREAM_STAGES.has(s.stage)) {
      out.push(s);
      i++;
      continue;
    }
    // Gom các event cùng stage + cùng iter liên tiếp thành 1 event tổng hợp
    const stageName = s.stage;
    const iter = (s.data as { iter?: number } | undefined)?.iter;
    const startTs = s.timestamp;
    let acc = '';
    let lastAccumulated: string | undefined;
    let j = i;
    while (
      j < stages.length &&
      stages[j].stage === stageName &&
      (stages[j].data as { iter?: number } | undefined)?.iter === iter
    ) {
      const d = stages[j].data;
      if (typeof d === 'string') {
        acc += d;
      } else if (d && typeof d === 'object') {
        const tok = (d as { token?: string }).token;
        if (typeof tok === 'string') acc += tok;
        const accField = (d as { accumulated?: string }).accumulated;
        if (typeof accField === 'string') lastAccumulated = accField;
      }
      j++;
    }
    // `thinking`/`answer`: data là string nguyên gốc → tổng hợp thành 1 string lớn
    // Còn lại: data là object {iter, token, accumulated?} → giữ nguyên schema
    if (stageName === 'thinking' || stageName === 'answer') {
      out.push({ stage: stageName, data: acc, timestamp: startTs });
    } else {
      out.push({
        stage: stageName,
        data: { iter, token: acc, accumulated: lastAccumulated ?? acc },
        timestamp: startTs,
      });
    }
    i = j;
  }
  return out;
}

// Persist history into localStorage; on quota error, drop oldest assistant turn
// và retry. Trường hợp xấu nhất (1 message vượt quota): bỏ qua persist nhưng
// state in-memory vẫn còn nên UI không sập.
function safePersistHistory(key: string, history: ChatMessage[]): void {
  let toStore = history;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      localStorage.setItem(key, JSON.stringify(toStore));
      return;
    } catch (e) {
      if (e instanceof Error && e.name === 'QuotaExceededError' && toStore.length > 1) {
        // Bỏ message cũ nhất rồi thử lại
        toStore = toStore.slice(1);
        continue;
      }
      // Quota vẫn vượt với 1 message hoặc lỗi khác → log và bỏ qua persist
      console.warn('Failed to persist chat history:', e);
      return;
    }
  }
}

interface ChatBoxProps {
  sessionId: string;
  // Optional clickable prompts shown above the input while history is empty
  suggestions?: string[];
}

// Imperative handle the parent can use to read chat state for export, etc.
export interface ChatBoxHandle {
  getHistory: () => ChatMessage[];
  getSettings: () => { mode: Mode; model: string; think: boolean };
  clearHistory: () => void;
}

const MODELS = ['qwen3:14b', 'deepseek-r1:14b', 'llama3:8b', 'mistral:7b', 'gemma3:12b'] as const;

export const ChatBox = forwardRef<ChatBoxHandle, ChatBoxProps>(function ChatBox(
  { sessionId, suggestions }, ref
) {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<Mode>('fixed');
  const [model, setModel] = useState<string>(MODELS[0]);
  // Thinking toggle — when off, reasoning models skip the <think> phase entirely
  const [think, setThink] = useState<boolean>(true);
  const startedAtRef = useRef<number>(0);

  const { answer, thinking, sources, stages, loading, error, send, reset } = useSSE('/api/query');

  // Load chat history from localStorage when sessionId changes (or on mount)
  useEffect(() => {
    const raw = localStorage.getItem(`chat-history-${sessionId}`);
    if (!raw) {
      setHistory([]);
      return;
    }
    try {
      // Compact stages khi load — bản trước có thể đã lưu hàng nghìn token event
      // gây render chậm + nguy cơ quota tiếp khi persist lại
      const parsed = JSON.parse(raw) as ChatMessage[];
      const slimmed = parsed.map((m) =>
        m.stages ? { ...m, stages: compactStagesForStorage(m.stages) } : m,
      );
      setHistory(slimmed);
    } catch {
      // Corrupt entry — fall back to empty history
      setHistory([]);
    }
  }, [sessionId]);

  // Persist history whenever it changes. setItem có thể throw QuotaExceededError
  // khi tổng size vượt 5-10 MB → cần catch để không sập React tree.
  useEffect(() => {
    safePersistHistory(`chat-history-${sessionId}`, history);
  }, [sessionId, history]);

  // Expose imperative API so the parent (App header) can trigger export/clear
  // without lifting state up.
  useImperativeHandle(
    ref,
    () => ({
      getHistory: () => history,
      getSettings: () => ({ mode, model, think }),
      clearHistory: () => {
        setHistory([]);
        localStorage.removeItem(`chat-history-${sessionId}`);
      },
    }),
    [history, mode, model, think, sessionId],
  );

  const handleSend = (message: string) => {
    setHistory((prev) => [...prev, { role: 'user', content: message }]);
    startedAtRef.current = Date.now();
    send({ question: message, mode, model, think });
  };

  // Commit the streaming assistant turn into history when it finishes
  useEffect(() => {
    if (loading) return;
    if (!answer && stages.length === 0) return;
    // Compact token streams trước khi vào history — tránh lưu hàng nghìn event
    // per-token vào localStorage (quota 5-10 MB)
    const compactStages = compactStagesForStorage(stages);
    setHistory((prev) => {
      // Avoid duplicating if effect re-runs
      const last = prev[prev.length - 1];
      if (last?.role === 'assistant' && last.content === answer) return prev;
      return [
        ...prev,
        {
          role: 'assistant',
          content: answer,
          thinking,
          stages: compactStages,
          sources,
          startedAt: startedAtRef.current,
        },
      ];
    });
    reset();
  }, [loading]); // eslint-disable-line react-hooks/exhaustive-deps

  const streaming = loading
    ? { answer, thinking, stages, sources, startedAt: startedAtRef.current }
    : null;

  return (
    <div className="flex flex-col h-full">
      {error && (
        <div className="mx-auto max-w-3xl w-full mt-2 px-4">
          <div className="px-3 py-2 bg-red-50 text-red-600 text-sm rounded-lg border border-red-200">
            {error}
          </div>
        </div>
      )}
      <MessageList messages={history} streaming={streaming} />
      {/* Suggestion chips — only while no conversation has started yet */}
      {suggestions && suggestions.length > 0 && history.length === 0 && !loading && (
        <div className="px-4 pb-1">
          <div className="max-w-3xl mx-auto flex flex-wrap gap-1.5">
            {suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => handleSend(s)}
                className="px-2.5 py-1 text-xs rounded-full border border-gray-200
                           bg-gray-50 hover:bg-gray-100 hover:border-gray-300
                           text-gray-700 text-left transition-colors"
                title={s}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}
      <MessageInput
        onSend={handleSend}
        disabled={loading}
        mode={mode}
        onModeChange={setMode}
        model={model}
        onModelChange={setModel}
        models={MODELS}
        think={think}
        onThinkChange={setThink}
      />
    </div>
  );
});
