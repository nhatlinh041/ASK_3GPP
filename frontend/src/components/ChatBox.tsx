import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { MessageList, type ChatMessage } from './MessageList';
import { MessageInput } from './MessageInput';
import { useSSE } from '../hooks/useSSE';
import type { Mode } from './ModeToggle';

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
      setHistory(JSON.parse(raw) as ChatMessage[]);
    } catch {
      // Corrupt entry — fall back to empty history
      setHistory([]);
    }
  }, [sessionId]);

  // Persist history whenever it changes
  useEffect(() => {
    localStorage.setItem(`chat-history-${sessionId}`, JSON.stringify(history));
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
          stages,
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
