"use client";

import { useState } from "react";

export default function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";

  if (message.role === "error") {
    return (
      <div className="flex justify-center">
        <div className="max-w-[85%] rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] sm:max-w-[70%] ${isUser ? "order-2" : ""}`}>
        <div
          className={`whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
            isUser
              ? "rounded-br-sm bg-neutral-900 text-white"
              : "rounded-bl-sm bg-neutral-100 text-neutral-900"
          }`}
        >
          {message.content}
        </div>

        {!isUser && (message.sources?.length > 0 || message.meta) && (
          <div className="mt-1 flex items-center gap-2 px-1 text-xs text-neutral-400">
            {message.meta && (
              <span>
                {message.meta.llm_backend_used}
                {typeof message.meta.latency_ms === "number"
                  ? ` · ${Math.round(message.meta.latency_ms)}ms`
                  : ""}
              </span>
            )}
            {message.sources?.length > 0 && (
              <button
                type="button"
                onClick={() => setShowSources((v) => !v)}
                className="underline decoration-dotted underline-offset-2 hover:text-neutral-600"
              >
                {showSources ? "Hide sources" : `${message.sources.length} source${
                  message.sources.length === 1 ? "" : "s"
                }`}
              </button>
            )}
          </div>
        )}

        {!isUser && showSources && message.sources?.length > 0 && (
          <ul className="mt-2 space-y-2">
            {message.sources.map((source, i) => (
              <li
                key={i}
                className="rounded-lg border border-neutral-200 bg-white px-3 py-2 text-xs text-neutral-600"
              >
                <p className="line-clamp-3 whitespace-pre-wrap">{source.text}</p>
                {typeof source.similarity_score === "number" && (
                  <p className="mt-1 text-neutral-400">
                    similarity: {source.similarity_score.toFixed(3)}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
