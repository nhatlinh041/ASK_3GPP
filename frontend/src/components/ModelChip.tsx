import React, { useEffect, useRef, useState } from 'react';

interface ModelChipProps {
  models: readonly string[];
  value: string;
  onChange: (m: string) => void;
}

// Model selector chip — pill button + dropdown popover, opens upward (above the composer)
export function ModelChip({ models, value, onChange }: ModelChipProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs
                   bg-white border border-gray-200 text-gray-700 hover:border-gray-300
                   hover:bg-gray-50 transition-colors"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3" />
          <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12" />
        </svg>
        <span className="font-medium">{value}</span>
        <svg
          width="10"
          height="10"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          className={`transition-transform ${open ? 'rotate-180' : ''}`}
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div
          className="absolute bottom-full mb-2 left-0 min-w-[180px] py-1
                     bg-white border border-gray-200 rounded-lg shadow-lg z-50"
        >
          {models.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => {
                onChange(m);
                setOpen(false);
              }}
              className={`w-full text-left px-3 py-1.5 text-sm hover:bg-gray-50 flex items-center gap-2
                          ${m === value ? 'text-gray-900' : 'text-gray-700'}`}
            >
              <span className="w-3 inline-block">
                {m === value && (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <path d="M20 6L9 17l-5-5" />
                  </svg>
                )}
              </span>
              {m}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
