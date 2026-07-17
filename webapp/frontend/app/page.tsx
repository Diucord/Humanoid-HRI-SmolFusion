"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import Camera from "@/components/Camera";
import VisionPanel from "@/components/VisionPanel";
import RobotFace, { RobotState } from "@/components/RobotFace";
import CreatePersonaModal from "@/components/CreatePersonaModal";
import { ArrowUpIcon, MicIcon, StopIcon, ImageIcon, RobotIcon, CustomIcon } from "@/components/Icons";
import { useSpeechRecognition } from "@/lib/useSpeech";
import {
  fetchPersonas,
  sendChat,
  uploadDocs,
  clearRag,
  speak,
  resetSession,
  analyzeFrame,
  checkHealth,
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
  const [traits, setTraits] = useState({
    friendliness: 3,
    knowledge: 3,
    empathy: 3,
    formality: 3,
  });
  const [sessionId] = useState(genSession);

  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  const [cameraOn, setCameraOn] = useState(false);
  const [useVision, setUseVision] = useState(true);
  const [useRag, setUseRag] = useState(true);
  const [vision, setVision] = useState<VisionResult | null>(null);
  const [newPersonFlash, setNewPersonFlash] = useState(false);
  const [robotState, setRobotState] = useState<RobotState>("idle");
  const lastVisionRef = useRef<VisionResult | null>(null);
  const imgInputRef = useRef<HTMLInputElement>(null);

  const [visionLoading, setVisionLoading] = useState(false);
  const [visionError, setVisionError] = useState("");
  const [ragStatus, setRagStatus] = useState("");
  const [ttsOn, setTtsOn] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);

  const handlePersonaCreated = useCallback((p: Persona) => {
    setPersonas((prev) => [...prev, p]);
    setPersonaId(p.id);
    setMessages([]);
  }, []);

  const persona = personas.find((p) => p.id === personaId);
  const lang = persona?.language === "en" ? "en-US" : "ko-KR";

  const speech = useSpeechRecognition(lang);
  const logRef = useRef<HTMLDivElement>(null);
  const historyRef = useRef<HTMLDivElement>(null);

  const [serverOnline, setServerOnline] = useState<boolean | null>(null);

  useEffect(() => {
    fetchPersonas().then(setPersonas).catch(() => {});
    // 서버 상태 체크 (30초마다)
    const check = () => checkHealth().then(setServerOnline);
    check();
    const id = setInterval(check, 30000);
    return () => clearInterval(id);
  }, []);

  // 메시지 추가 시 양쪽 자동 스크롤 (맨 아래로)
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
    historyRef.current?.scrollTo({ top: historyRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, speech.interim]);

  // 음성 듣는 중 → 로봇 listening 상태 (말하는 중/생각 중이 아닐 때만)
  useEffect(() => {
    setRobotState((prev) => {
      if (prev === "speaking" || prev === "thinking" || prev === "greeting") return prev;
      return speech.listening ? "listening" : "idle";
    });
  }, [speech.listening]);

  const playTTS = useCallback(
    async (text: string) => {
      if (!ttsOn) return;
      try {
        const blob = await speak(text, persona?.language || "ko", persona?.voice);
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        setRobotState("speaking");
        audio.play().catch(() => setRobotState("idle"));
        audio.onended = () => {
          URL.revokeObjectURL(url);
          setRobotState("idle");
        };
      } catch {
        setRobotState("idle");
      }
    },
    [ttsOn, persona]
  );

  // 시각 분석 결과 처리 + 새 사람이면 자동 인사
  const handleVision = useCallback(
    (v: VisionResult) => {
      lastVisionRef.current = v;
      setVision(v);
      if (v.is_new_person && v.has_person) {
        setNewPersonFlash(true);
        setTimeout(() => setNewPersonFlash(false), 3000);
        // 자동 인사
        const greeting = v.greeting;
        if (greeting && personaId === "igris") {
          setRobotState("greeting");
          setMessages((m) => [
            ...m,
            { role: "bot", text: greeting, source: "greeting" },
          ]);
          playTTS(greeting);
        }
      }
    },
    [personaId, playTTS]
  );

  // 이미지 파일로 VLM 테스트 (카메라 대신)
  const handleTestImage = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setVision(null);
      setVisionError("");
      setVisionLoading(true);
      try {
        const result = await analyzeFrame(sessionId, file, true); // 수동 업로드 → 새 사람 인사
        handleVision(result);
      } catch (err: any) {
        setVisionError(`분석 실패: ${err?.message || err}`);
      } finally {
        setVisionLoading(false);
      }
      e.target.value = "";
    },
    [sessionId, handleVision]
  );

  const handleSend = useCallback(
    async (text: string) => {
      const msg = text.trim();
      if (!msg || sending) return;
      setSending(true);
      setRobotState("thinking");
      setMessages((m) => [...m, { role: "user", text: msg }]);
      setInput("");

      // Vision 컨텍스트 구성 (나이/성별/표정 → 사람 맞춤 응대)
      let visionContext = "";
      if (useVision && lastVisionRef.current?.has_person) {
        const v = lastVisionRef.current;
        const ageKo: Record<string, string> = {
          child: "어린이", teenager: "청소년", "young adult": "청년",
          "middle aged": "중년", elderly: "노년",
        };
        const parts: string[] = [];
        if (v.age_group && v.age_group !== "unknown") parts.push(`연령대: ${ageKo[v.age_group] || v.age_group}`);
        if (v.gender === "male") parts.push("성별: 남성");
        else if (v.gender === "female") parts.push("성별: 여성");
        if (v.is_smiling) parts.push("웃고 있음");
        if (v.scene) parts.push(v.scene);
        visionContext = parts.join(", ");
      }

      try {
        const res: ChatResponse = await sendChat({
          message: msg,
          sessionId,
          personaId,
          customPrompt,
          visionContext,
          useRag,
          traits: personaId === "custom" ? traits : undefined,
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
    [sending, useVision, sessionId, personaId, customPrompt, useRag, traits, playTTS]
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
      {/* 물결 배경 블롭 */}
      <div className="blob-bg" aria-hidden>
        <div className="blob b1" />
        <div className="blob b2" />
        <div className="blob b3" />
        <div className="blob b4" />
      </div>

      <div className="header">
        <h1>Hera</h1>
        <span className="badge">Human-robot Engagement Responsive AI</span>
        {serverOnline !== null && (
          <span className={`server-status ${serverOnline ? "online" : "offline"}`}>
            <span className="status-dot" />
            {serverOnline ? "Server Online" : "Server Offline"}
          </span>
        )}
      </div>

      {/* 오프라인 안내 배너 */}
      {serverOnline === false && (
        <div className="offline-banner">
          ⚠️ 데모 서버가 현재 오프라인입니다. 잠시만 기다려 주세요.
        </div>
      )}

      {/* 사이드바 */}
      <div className="sidebar">
        {/* 페르소나 */}
        <div className="panel">
          <div className="section-title">Persona</div>
          <div className="persona-grid">
            {personas.map((p) => (
              <div
                key={p.id}
                className={`persona-card ${p.id === personaId ? "active" : ""}`}
                onClick={() => handlePersonaChange(p.id)}
              >
                <span className="persona-icon">
                  {p.id === "igris" ? <RobotIcon /> : <CustomIcon />}
                </span>
                <div className="info">
                  <div className="name">{p.name}</div>
                  <div className="desc">{p.description}</div>
                </div>
              </div>
            ))}
          </div>

          <button
            className="ghost create-persona-btn"
            onClick={() => setShowCreateModal(true)}
          >
+ New Persona
          </button>
          {personaId === "custom" && (
            <div className="custom-config">
              <TraitSlider
                label="친절도"
                value={traits.friendliness}
                onChange={(v) => setTraits((t) => ({ ...t, friendliness: v }))}
                left="사무적"
                right="다정함"
              />
              <TraitSlider
                label="지식 수준"
                value={traits.knowledge}
                onChange={(v) => setTraits((t) => ({ ...t, knowledge: v }))}
                left="쉽게"
                right="전문적"
              />
              <TraitSlider
                label="공감 능력"
                value={traits.empathy}
                onChange={(v) => setTraits((t) => ({ ...t, empathy: v }))}
                left="객관적"
                right="공감적"
              />
              <TraitSlider
                label="말투"
                value={traits.formality}
                onChange={(v) => setTraits((t) => ({ ...t, formality: v }))}
                left="반말"
                right="격식"
              />
              <textarea
                style={{ marginTop: 10 }}
                rows={3}
                placeholder="추가 설정 (선택): 예) 당신은 천문학 전문가입니다."
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
              />
            </div>
          )}
        </div>

        {/* 카메라 */}
        <div className="panel">
          <div className="section-title">Realtime Vision Analysis (Qwen3-VL-4B)</div>
          <Camera sessionId={sessionId} enabled={cameraOn} onVision={handleVision} />
          <div className="row" style={{ marginTop: 10 }}>
            <button
              className="ghost"
              style={{ flex: 1 }}
              onClick={() => setCameraOn((c) => !c)}
            >
              {cameraOn ? "Camera Off" : "Camera On"}
            </button>
          </div>

          {/* 이미지 업로드 테스트 (카메라 없는 환경/원격용) */}
          <input
            ref={imgInputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleTestImage}
          />
          <button
            className="ghost"
            style={{ width: "100%", marginTop: 8, fontSize: 13, display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}
            onClick={() => imgInputRef.current?.click()}
            disabled={visionLoading}
          >
            <ImageIcon /> {visionLoading ? "Analyzing..." : "Test with Image"}
          </button>

          <label className="checkbox-row" style={{ marginTop: 10 }}>
            <input
              type="checkbox"
              checked={useVision}
              onChange={(e) => setUseVision(e.target.checked)}
            />
Use vision context
          </label>
        </div>

        {/* RAG */}
        <div className="panel">
          <div className="section-title">Knowledge Base (RAG)</div>
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
Enable retrieval
          </label>
          <div className="row" style={{ marginTop: 8 }}>
            <button
              className="ghost"
              style={{ flex: 1, fontSize: 12, padding: "8px" }}
              onClick={async () => {
                await clearRag(personaId);
                setRagStatus("🗑️ Cleared");
              }}
            >
Clear
            </button>
          </div>
          {ragStatus && <div className="status">{ragStatus}</div>}
        </div>
      </div>

      {/* 채팅 */}
      <div className="panel chat-panel">
        {/* 로봇 캐릭터 */}
        <div className="robot-stage">
          <RobotFace state={robotState} size={130} />
          <div className="robot-status-text">
            {robotState === "listening"
              ? "Listening..."
              : robotState === "thinking"
              ? "Thinking..."
              : robotState === "speaking"
              ? "Speaking..."
              : robotState === "greeting"
              ? "Hello!"
              : `${persona?.name || "Hera"} · Idle`}
          </div>
        </div>

        {/* 시각 분석 패널 */}
        <VisionPanel
          vision={vision}
          newPersonFlash={newPersonFlash}
          loading={visionLoading}
          error={visionError}
        />

        <div
          className="section-title row"
          style={{ justifyContent: "space-between", marginTop: 14 }}
        >
          <span>Dialogue</span>
          <label className="checkbox-row small">
            <input
              type="checkbox"
              checked={ttsOn}
              onChange={(e) => setTtsOn(e.target.checked)}
            />
Voice output
          </label>
        </div>

        <div className="chat-split">
          {/* 좌: 현재 대화 (크게) */}
          <div className="chat-main" ref={logRef}>
            {messages.length === 0 && (
              <div className="small" style={{ margin: "auto", textAlign: "center" }}>
                마이크를 누르고 말하거나, 아래에 메시지를 입력하세요.
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

          {/* 우: 전체 히스토리 (짧게) */}
          <div className="chat-history">
            <div className="history-title">History</div>
            <div className="history-list" ref={historyRef}>
              {messages.length === 0 && (
                <div className="small" style={{ textAlign: "center", marginTop: 20 }}>
No messages yet
                </div>
              )}
              {messages.map((m, i) => (
                <div key={i} className={`history-item ${m.role}`}>
                  <span className="history-role">
                    {m.role === "user" ? "You" : persona?.name || "Hera"}
                  </span>
                  <span className="history-text">{m.text}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="chat-input-row">
          <button
            className={`mic ${speech.listening ? "listening" : "ghost"}`}
            onClick={handleMic}
            disabled={!speech.supported}
            title={speech.supported ? "음성 입력" : "브라우저가 음성 인식 미지원"}
          >
            {speech.listening ? <StopIcon /> : <MicIcon />}
          </button>
          <input
            type="text"
            placeholder="메시지를 입력하세요..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend(input)}
          />
          <button
            className="send-btn"
            onClick={() => handleSend(input)}
            disabled={sending}
            aria-label="전송"
          >
            <ArrowUpIcon />
          </button>
        </div>
      </div>

      {/* 페르소나 생성 모달 */}
      {showCreateModal && (
        <CreatePersonaModal
          onClose={() => setShowCreateModal(false)}
          onCreated={handlePersonaCreated}
        />
      )}
    </div>
  );
}

function TraitSlider({
  label,
  value,
  onChange,
  left,
  right,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  left: string;
  right: string;
}) {
  return (
    <div className="trait-slider">
      <div className="trait-head">
        <span className="trait-label">{label}</span>
        <span className="trait-val">{value}/5</span>
      </div>
      <input
        type="range"
        min={1}
        max={5}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="trait-ends">
        <span>{left}</span>
        <span>{right}</span>
      </div>
    </div>
  );
}
