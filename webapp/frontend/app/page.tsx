"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import Camera from "@/components/Camera";
import { useSpeechRecognition } from "@/lib/useSpeech";
import {
  fetchPersonas,
  sendChat,
  uploadDocs,
  clearRag,
  speak,
  resetSession,
  Persona,
  VisionResult,
  ChatResponse,
} from "@/lib/api";

interface Msg {
  role: "user" | "bot";
  text: string;
  source?: string;
}

// 세션 ID (브라우저 새로고침마다 새로)
function genSession() {
  return "web_" + Math.random().toString(36).slice(2, 10);
}

export default function Home() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [personaId, setPersonaId] = useState("igris");
  const [customPrompt, setCustomPrompt] = useState("");
  const [sessionId] = useState(genSession);

  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  const [cameraOn, setCameraOn] = useState(false);
  const [useVision, setUseVision] = useState(true);
  const [useRag, setUseRag] = useState(true);
  const lastVisionRef = useRef<VisionResult | null>(null);

  const [ragStatus, setRagStatus] = useState("");
  const [ttsOn, setTtsOn] = useState(true);

  const persona = personas.find((p) => p.id === personaId);
  const lang = persona?.language === "en" ? "en-US" : "ko-KR";

  const speech = useSpeechRecognition(lang);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchPersonas().then(setPersonas).catch(() => {});
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [messages]);

  const playTTS = useCallback(
    async (text: string) => {
      if (!ttsOn) return;
      try {
        const blob = await speak(text, persona?.language || "ko", persona?.voice);
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.play().catch(() => {});
        audio.onended = () => URL.revokeObjectURL(url);
      } catch {}
    },
    [ttsOn, persona]
  );

  const handleSend = useCallback(
    async (text: string) => {
      const msg = text.trim();
      if (!msg || sending) return;
      setSending(true);
      setMessages((m) => [...m, { role: "user", text: msg }]);
      setInput("");

      // Vision 컨텍스트 구성
      let visionContext = "";
      if (useVision && lastVisionRef.current?.has_person) {
        const v = lastVisionRef.current;
        visionContext = v.scene || "";
      }

      try {
        const res: ChatResponse = await sendChat({
          message: msg,
          sessionId,
          personaId,
          customPrompt,
          visionContext,
          useRag,
        });
        setMessages((m) => [
          ...m,
          { role: "bot", text: res.text, source: res.source },
        ]);
        playTTS(res.text);
      } catch {
        setMessages((m) => [
          ...m,
          { role: "bot", text: "❌ 서버 연결 오류", source: "error" },
        ]);
      } finally {
        setSending(false);
      }
    },
    [sending, useVision, sessionId, personaId, customPrompt, useRag, playTTS]
  );

  const handleMic = useCallback(() => {
    if (speech.listening) {
      speech.stop();
    } else {
      speech.start((finalText) => handleSend(finalText));
    }
  }, [speech, handleSend]);

  const handleUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;
      setRagStatus("업로드 중...");
      try {
        const res = await uploadDocs(personaId, files);
        setRagStatus(`✅ ${res.chunks_added}개 청크 추가 (총 ${res.total_chunks})`);
      } catch {
        setRagStatus("❌ 업로드 실패");
      }
      e.target.value = "";
    },
    [personaId]
  );

  const handlePersonaChange = useCallback(
    (id: string) => {
      setPersonaId(id);
      setMessages([]);
      resetSession(sessionId).catch(() => {});
    },
    [sessionId]
  );

  return (
    <div className="app">
      <div className="header">
        <h1>🤖 Humanoid HRI · SmolFusion</h1>
        <span className="badge">Vision · RAG · Persona</span>
      </div>

      {/* 사이드바 */}
      <div className="sidebar">
        {/* 페르소나 */}
        <div className="panel">
          <div className="section-title">페르소나</div>
          <div className="persona-grid">
            {personas.map((p) => (
              <div
                key={p.id}
                className={`persona-card ${p.id === personaId ? "active" : ""}`}
                onClick={() => handlePersonaChange(p.id)}
              >
                <span className="emoji">{p.emoji}</span>
                <div className="info">
                  <div className="name">{p.name}</div>
                  <div className="desc">{p.description}</div>
                </div>
              </div>
            ))}
          </div>
          {personaId === "custom" && (
            <textarea
              style={{ marginTop: 10 }}
              rows={4}
              placeholder="당신은 ... 입니다."
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
            />
          )}
        </div>

        {/* 카메라 */}
        <div className="panel">
          <div className="section-title">실시간 Vision</div>
          <Camera
            sessionId={sessionId}
            enabled={cameraOn}
            onVision={(v) => (lastVisionRef.current = v)}
          />
          <div className="row" style={{ marginTop: 10 }}>
            <button
              className="ghost"
              style={{ flex: 1 }}
              onClick={() => setCameraOn((c) => !c)}
            >
              {cameraOn ? "카메라 끄기" : "카메라 켜기"}
            </button>
          </div>
          <label className="checkbox-row" style={{ marginTop: 10 }}>
            <input
              type="checkbox"
              checked={useVision}
              onChange={(e) => setUseVision(e.target.checked)}
            />
            대화에 시각 정보 활용
          </label>
        </div>

        {/* RAG */}
        <div className="panel">
          <div className="section-title">지식 베이스 (RAG)</div>
          <input
            className="file-input"
            type="file"
            multiple
            accept=".txt,.pdf,.md"
            onChange={handleUpload}
          />
          <label className="checkbox-row" style={{ marginTop: 10 }}>
            <input
              type="checkbox"
              checked={useRag}
              onChange={(e) => setUseRag(e.target.checked)}
            />
            검색 증강 사용
          </label>
          <div className="row" style={{ marginTop: 8 }}>
            <button
              className="ghost"
              style={{ flex: 1, fontSize: 12, padding: "8px" }}
              onClick={async () => {
                await clearRag(personaId);
                setRagStatus("🗑️ 초기화됨");
              }}
            >
              초기화
            </button>
          </div>
          {ragStatus && <div className="status">{ragStatus}</div>}
        </div>
      </div>

      {/* 채팅 */}
      <div className="panel chat-panel">
        <div className="section-title row" style={{ justifyContent: "space-between" }}>
          <span>{persona?.emoji} {persona?.name} 와의 대화</span>
          <label className="checkbox-row small">
            <input
              type="checkbox"
              checked={ttsOn}
              onChange={(e) => setTtsOn(e.target.checked)}
            />
            음성 출력
          </label>
        </div>

        <div className="chat-log" ref={logRef}>
          {messages.length === 0 && (
            <div className="small" style={{ margin: "auto", textAlign: "center" }}>
              🎤 마이크를 누르고 말하거나, 아래에 메시지를 입력하세요.
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              {m.text}
              {m.source && m.role === "bot" && (
                <div className="meta">{m.source}</div>
              )}
            </div>
          ))}
          {speech.interim && (
            <div className="bubble user" style={{ opacity: 0.5 }}>
              {speech.interim}
            </div>
          )}
        </div>

        <div className="chat-input-row">
          <button
            className={`mic ${speech.listening ? "listening" : "ghost"}`}
            onClick={handleMic}
            disabled={!speech.supported}
            title={speech.supported ? "음성 입력" : "브라우저가 음성 인식 미지원"}
          >
            {speech.listening ? "🔴" : "🎤"}
          </button>
          <input
            type="text"
            placeholder="메시지를 입력하세요..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend(input)}
          />
          <button onClick={() => handleSend(input)} disabled={sending}>
            전송
          </button>
        </div>
      </div>
    </div>
  );
}
