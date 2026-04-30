import React from 'react';
import { createRoot } from 'react-dom/client';
import { CypherTester } from './components/CypherTester';
import { ChatPopup } from './components/ChatPopup';
import './index.css';

// Separate Vite entry — this page is opened in a new tab from the Chat header
function CypherApp() {
  return (
    <div className="flex flex-col h-screen bg-white font-sans text-gray-900">
      <header className="flex items-center gap-3 px-4 py-2 border-b border-gray-200 bg-white">
        <span className="text-sm font-semibold text-gray-700">3GPP QA</span>
        <span className="text-gray-300">/</span>
        <span className="text-sm text-gray-600">Cypher Tester</span>
        <div className="flex-1" />
        <a
          href="/"
          className="text-xs text-gray-500 hover:text-gray-800 underline"
        >
          ← Back to Chat
        </a>
      </header>
      <div className="flex-1 overflow-hidden">
        <CypherTester />
      </div>
      {/* Floating chat widget — popup test chat without leaving the page */}
      <ChatPopup />
    </div>
  );
}

const root = document.getElementById('root')!;
createRoot(root).render(<CypherApp />);
