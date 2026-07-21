const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  return handleResponse(res);
}

export async function ingestDocuments() {
  const res = await fetch(`${API_BASE_URL}/ingest`, { method: "POST" });
  return handleResponse(res);
}


export async function sendChatMessage(query, llmBackend) {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, llm_backend: llmBackend }),
  });
  return handleResponse(res);
}

export async function streamChatMessage(query, llmBackend, onChunk) {
  const res = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      llm_backend: llmBackend,
    }),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {}

    throw new ApiError(detail, res.status);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    const chunk = decoder.decode(value, { stream: true });

    fullText += chunk;

    onChunk(fullText);
  }

  return fullText;
}

export { ApiError };
