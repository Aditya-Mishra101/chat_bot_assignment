"use client";

import ModelToggle from "./ModelToggle";

export default function Header({
  health,
  healthError,
  llmBackend,
  onLlmBackendChange,
  onIngest,
  isIngesting,
  isChatting,
}) {
  return (
    <header className="flex flex-col gap-3 border-b border-neutral-200 bg-white px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
      <div>
        <h1 className="text-lg font-semibold text-neutral-900">RAG Chatbot</h1>
        <p className="mt-0.5 text-s text-neutral-500">
          {healthError
            ? "Backend unreachable"
            : health
            ? `${health.documents_indexed ?? 0} document${
                health.documents_indexed === 1 ? "" : "s"
              } indexed`
            : "Checking status…"}
        </p>
      </div>

      <div className="flex items-center gap-3">
        <ModelToggle
          value={llmBackend}
          onChange={onLlmBackendChange}
          disabled={isChatting}
        />
        <button
          type="button"
          onClick={onIngest}
          disabled={isIngesting}
          className="rounded-lg border border-neutral-300 bg-white px-3 py-1.5 text-sm font-medium text-neutral-700 transition-colors hover:bg-neutral-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isIngesting ? "Ingesting…" : "Re-ingest docs"}
        </button>
      </div>
    </header>
  );
}
