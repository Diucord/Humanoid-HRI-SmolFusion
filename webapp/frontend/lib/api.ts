const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Persona {
  id: string;
  name: string;
  emoji: string;
  description: string;
  language: string;
  voice: string;
  tags: string[];
}

export interface VisionResult {
  has_person: boolean;
  age_group: string;
  gender: string;
  is_smiling: boolean;
  scene: string;
  is_new_person: boolean;
}

export interface ChatResponse {
  source: string;
  text: string;
  rag_used?: boolean;
}

export async function fetchPersonas(): Promise<Persona[]> {
  const res = await fetch(`${API}/personas`);
  const data = await res.json();
  return data.personas;
}

export async function analyzeFrame(
  sessionId: string,
  blob: Blob
): Promise<VisionResult> {
  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("image", blob, "frame.jpg");
  const res = await fetch(`${API}/vision/analyze`, { method: "POST", body: fd });
  return res.json();
}

export async function sendChat(params: {
  message: string;
  sessionId: string;
  personaId: string;
  customPrompt?: string;
  visionContext?: string;
  useRag?: boolean;
}): Promise<ChatResponse> {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: params.message,
      session_id: params.sessionId,
      persona_id: params.personaId,
      custom_prompt: params.customPrompt || "",
      vision_context: params.visionContext || "",
      use_rag: params.useRag ?? true,
    }),
  });
  return res.json();
}

export async function uploadDocs(
  personaId: string,
  files: FileList
): Promise<{ ok: boolean; chunks_added: number; total_chunks: number }> {
  const fd = new FormData();
  fd.append("persona_id", personaId);
  Array.from(files).forEach((f) => fd.append("files", f));
  const res = await fetch(`${API}/rag/upload`, { method: "POST", body: fd });
  return res.json();
}

export async function clearRag(personaId: string): Promise<void> {
  await fetch(`${API}/rag/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ persona_id: personaId }),
  });
}

export async function speak(text: string, lang = "ko", voice?: string): Promise<Blob> {
  const res = await fetch(`${API}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, lang, voice }),
  });
  return res.blob();
}

export async function resetSession(sessionId: string): Promise<void> {
  await fetch(`${API}/session/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}
