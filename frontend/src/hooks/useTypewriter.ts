import { useEffect, useRef, useState } from 'react';

// Reveal text gradually mỗi khi `text` thay đổi — dùng cho payload tĩnh
// (JSON input/output, prompt, cypher hoàn chỉnh) để chúng "type vào" mượt mà
// thay vì pop nguyên cục. Khác useTypewriter ở chỗ luôn reset khi text đổi.
export function useGradualReveal(text: string, charsPerTick = 6): string {
  const [revealed, setRevealed] = useState(0);

  useEffect(() => {
    setRevealed(0);
    if (!text) return;
    const id = window.setInterval(() => {
      setRevealed((cur) => {
        if (cur >= text.length) return cur;
        const backlog = text.length - cur;
        // Reveal đều: tối thiểu charsPerTick, tăng theo backlog để text dài
        // không kéo dài quá. Catch up trong ~24 ticks (~400ms).
        const step = Math.max(charsPerTick, Math.ceil(backlog / 24));
        return Math.min(text.length, cur + step);
      });
    }, 16);
    return () => clearInterval(id);
  }, [text, charsPerTick]);

  return text.slice(0, revealed);
}

// Reveal `text` ký tự theo nhịp đều — backend (Ollama, SSE) thường bắn token theo
// cụm lớn nên render thẳng sẽ giật cục. Hook này buffer text rồi xả ra dần dựa trên
// backlog: stream đang chạy thì pace mượt; stream xong thì flush nhanh phần còn lại.
export function useTypewriter(text: string, streaming: boolean): string {
  // Tin nhắn đã hoàn thành render full ngay; chỉ animate khi đang stream
  const [revealed, setRevealed] = useState(() => (streaming ? 0 : text.length));
  const textRef = useRef(text);
  const streamingRef = useRef(streaming);
  textRef.current = text;
  streamingRef.current = streaming;

  // Khi turn kết thúc, snap full text (bỏ phần animation còn lại để không trễ)
  useEffect(() => {
    if (!streaming) setRevealed(text.length);
  }, [streaming, text.length]);

  // Loop reveal liên tục — đọc text/streaming qua ref nên không cần restart interval
  useEffect(() => {
    const id = window.setInterval(() => {
      setRevealed((cur) => {
        const total = textRef.current.length;
        if (cur >= total) return cur;
        const backlog = total - cur;
        // Streaming: catch up trong ~24 ticks (~400ms), tối thiểu 2 char/tick (~125 c/s)
        // Done: flush trong ~6 ticks
        const step = streamingRef.current
          ? Math.max(2, Math.ceil(backlog / 24))
          : Math.max(20, Math.ceil(backlog / 6));
        return Math.min(total, cur + step);
      });
    }, 16);
    return () => clearInterval(id);
  }, []);

  return text.slice(0, revealed);
}
