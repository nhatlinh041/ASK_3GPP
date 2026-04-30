import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import type { PipelineStage, Source } from '../hooks/useSSE';
import { useTypewriter } from '../hooks/useTypewriter';
import { ThinkingTrail } from './ThinkingTrail';
import { SourcesFooter } from './SourcesFooter';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  // Reasoning trace from Ollama "think" channel (deepseek-r1, qwen3, …)
  thinking?: string;
  // Pipeline stages and sources captured for this assistant message
  stages?: PipelineStage[];
  sources?: Source[];
  // When the user sent the message that produced this turn (used as elapsed-time anchor)
  startedAt?: number;
}

interface MessageListProps {
  messages: ChatMessage[];
  // The currently-streaming assistant turn (rendered after `messages`)
  streaming: {
    answer: string;
    thinking: string;
    stages: PipelineStage[];
    sources: Source[];
    startedAt: number;
  } | null;
}

export function MessageList({ messages, streaming }: MessageListProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Bật khi user ở gần bottom; tắt khi họ scroll lên xem nội dung cũ
  const followRef = useRef(true);
  const prevMessagesLen = useRef(messages.length);

  // Detect user scroll: nếu cách bottom > 80px thì coi như họ đang đọc nội dung cũ
  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    followRef.current = distance < 80;
  };

  // Khi user gửi tin nhắn mới: luôn reset follow=true và kéo xuống bottom
  useEffect(() => {
    if (messages.length > prevMessagesLen.current) {
      followRef.current = true;
      bottomRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' });
    }
    prevMessagesLen.current = messages.length;
  }, [messages.length]);

  // Trong lúc stream: chỉ auto-scroll khi follow đang bật
  useEffect(() => {
    if (!followRef.current) return;
    bottomRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' });
  }, [streaming?.answer, streaming?.stages.length]);

  const isEmpty = messages.length === 0 && !streaming;

  return (
    <div ref={containerRef} onScroll={handleScroll} className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-4 py-6">
        {isEmpty && (
          <div className="flex flex-col items-center justify-center h-[60vh] text-center">
            <div className="text-2xl font-semibold text-gray-800 mb-2">
              3GPP Knowledge Assistant
            </div>
            <div className="text-sm text-gray-500 max-w-md">
              Ask about 5G architecture, network functions (AMF, SMF, UPF…), procedures, or any
              technical specification.
            </div>
          </div>
        )}

        <div className="space-y-6">
          {messages.map((msg, i) => (
            <MessageRow key={i} msg={msg} streaming={false} />
          ))}

          {streaming && (
            <MessageRow
              msg={{
                role: 'assistant',
                content: streaming.answer,
                thinking: streaming.thinking,
                stages: streaming.stages,
                sources: streaming.sources,
                startedAt: streaming.startedAt,
              }}
              streaming
            />
          )}
        </div>

        <div ref={bottomRef} />
      </div>
    </div>
  );
}

// Single message row — claude.ai style: user has a subtle bubble, assistant flows on background
function MessageRow({ msg, streaming }: { msg: ChatMessage; streaming: boolean }) {
  // Hook luôn được gọi (rule of hooks); khi streaming=false thì hook trả về full text ngay
  const displayedContent = useTypewriter(msg.content, streaming);

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] bg-gray-100 text-gray-900 rounded-2xl rounded-tr-sm px-4 py-2.5 text-[15px] whitespace-pre-wrap leading-relaxed">
          {msg.content}
        </div>
      </div>
    );
  }

  // Assistant: pipeline trail + thinking trace above, markdown body, sources below
  return (
    <div>
      {msg.stages !== undefined && (
        <ThinkingTrail
          stages={msg.stages}
          streaming={streaming}
          startedAt={msg.startedAt ?? Date.now()}
        />
      )}
      {msg.thinking && msg.thinking.length > 0 && (
        <ReasoningPanel thinking={msg.thinking} streaming={streaming && !msg.content} />
      )}
      <div className="md-body text-[15px] leading-relaxed text-gray-900">
        <ReactMarkdown>{displayedContent}</ReactMarkdown>
        {streaming && (
          <span className="inline-block w-1.5 h-4 bg-gray-500 animate-pulse align-text-bottom ml-0.5" />
        )}
      </div>
      {msg.sources && msg.sources.length > 0 && <SourcesFooter sources={msg.sources} />}
    </div>
  );
}

// Collapsible "Thoughts" panel — auto-expanded while the model is still thinking,
// auto-collapsed once the answer starts streaming so the user's eye moves to it.
function ReasoningPanel({ thinking, streaming }: { thinking: string; streaming: boolean }) {
  const [open, setOpen] = React.useState(streaming);
  // While streaming-and-thinking, force-open so the user can watch tokens land
  React.useEffect(() => {
    if (streaming) setOpen(true);
  }, [streaming]);

  return (
    <div className="my-2 border border-gray-200 rounded-lg bg-gray-50 text-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-gray-600 hover:bg-gray-100 rounded-lg"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`transition-transform ${open ? 'rotate-90' : ''}`}
        >
          <path d="M9 18l6-6-6-6" />
        </svg>
        <span className="font-medium">Thoughts</span>
        {streaming && (
          <span className="text-[11px] text-gray-400 italic ml-1">thinking…</span>
        )}
        <span className="ml-auto text-[11px] text-gray-400">{thinking.length} chars</span>
      </button>
      {open && (
        <pre className="px-3 pb-3 pt-1 whitespace-pre-wrap text-[13px] leading-relaxed text-gray-700 font-sans max-h-72 overflow-y-auto">
          {thinking}
        </pre>
      )}
    </div>
  );
}
