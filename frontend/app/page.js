"use client";

import { useCallback, useEffect, useState } from "react";
import Header from "../components/Header";
import ChatWindow from "../components/ChatWindow";
import ChatInput from "../components/ChatInput";
import { getHealth, ingestDocuments, sendChatMessage, ApiError, streamChatMessage } from "../lib/api";

let idCounter = 0;
function nextId() {
  idCounter += 1;
  return idCounter;
}

export default function Page() {
  const [messages, setMessages] = useState([]);
  const [llmBackend, setLlmBackend] = useState("api");
  const [isChatting, setIsChatting] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(false);

  const refreshHealth = useCallback(async () => {
    try {
      const data = await getHealth();
      setHealth(data);
      setHealthError(false);
    } catch {
      setHealthError(true);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  async function handleSend(query) {
    setMessages((prev) => [...prev, { id: nextId(), role: "user", content: query }]);
    setIsChatting(true);
    try {
      const data = await sendChatMessage(query, llmBackend);
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: "assistant",
          content: data.answer,
          sources: data.sources,
          meta: {
            llm_backend_used: data.llm_backend_used,
            latency_ms: data.latency_ms,
          },
        },
      ]);
    } catch (err) {
      const detail = err instanceof ApiError ? err.message : "Something went wrong.";
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "error", content: `Couldn't get a response: ${detail}` },
      ]);
    } finally {
      setIsChatting(false);
    }
  }

  async function handleStreamSend(query) {
    setMessages((prev) => [...prev, { id: nextId(), role: "user", content: query }]);
    setIsChatting(true);
  
    const assistantId = nextId();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);
  
    try {
      await streamChatMessage(query, llmBackend, (fullText) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: fullText } : m))
        );
      });
    } catch (err) {
      const detail = err instanceof ApiError ? err.message : "Something went wrong.";
      setMessages((prev) =>
        prev
          .filter((m) => m.id !== assistantId)
          .concat({ id: nextId(), role: "error", content: `Couldn't get a response: ${detail}` })
      );
    } finally {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m))
      );
      setIsChatting(false);
    }
  }

  async function handleIngest() {
    setIsIngesting(true);
    try {
      await ingestDocuments();
      await refreshHealth();
    } catch (err) {
      const detail = err instanceof ApiError ? err.message : "Something went wrong.";
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "error", content: `Ingestion failed: ${detail}` },
      ]);
    } finally {
      setIsIngesting(false);
    }
  }

  return (
    <main className="flex h-dvh flex-col bg-neutral-50">
      <Header
        health={health}
        healthError={healthError}
        llmBackend={llmBackend}
        onLlmBackendChange={setLlmBackend}
        onIngest={handleIngest}
        isIngesting={isIngesting}
        isChatting={isChatting}
      />
      <ChatWindow messages={messages} isChatting={isChatting} />
      <ChatInput onSend={handleStreamSend} disabled={isChatting} />
    </main>
  );
}
