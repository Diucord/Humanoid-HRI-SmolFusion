"use client";
import { useState, useRef } from "react";
import { createPersona, uploadDocs, Persona } from "@/lib/api";

interface Props {
  onClose: () => void;
  onCreated: (p: Persona) => void;
}

const TRAITS = [
  { key: "friendliness", label: "친절도", left: "사무적", right: "다정함" },
  { key: "knowledge", label: "지식 수준", left: "쉽게", right: "전문적" },
  { key: "empathy", label: "공감 능력", left: "객관적", right: "공감적" },
  { key: "formality", label: "말투", left: "반말", right: "격식" },
] as const;

export default function CreatePersonaModal({ onClose, onCreated }: Props) {
  const [name, setName] = useState("");
  const [prompt, setPrompt] = useState("");
  const [traits, setTraits] = useState({
    friendliness: 3,
    knowledge: 3,
    empathy: 3,
    formality: 3,
  });
  const [files, setFiles] = useState<FileList | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleCreate = async () => {
    if (!name.trim()) {
      setStatus("이름을 입력해주세요.");
      return;
    }
    setBusy(true);
    setStatus("페르소나 생성 중...");
    try {
      const persona = await createPersona({
        name,
        systemPrompt: prompt,
        traits,
      });
      // RAG 문서가 있으면 해당 페르소나에 업로드
      if (files && files.length > 0) {
        setStatus("지식 문서 업로드 중...");
        await uploadDocs(persona.id, files);
      }
      onCreated(persona);
      onClose();
    } catch (e: any) {
      setStatus(`실패: ${e?.message || e}`);
      setBusy(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>새 페르소나 만들기</h2>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <label className="field">
            <span className="field-label">페르소나 이름</span>
            <input
              type="text"
              placeholder="예: 천문학 튜터, 마케팅 도우미"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>

          <label className="field">
            <span className="field-label">시스템 프롬프트 (역할 설정)</span>
            <textarea
              rows={3}
              placeholder="예: 당신은 천문학 전문가입니다. 우주와 천체에 대해 알려줍니다."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </label>

          <div className="field">
            <span className="field-label">성격 커스텀</span>
            <div className="modal-traits">
              {TRAITS.map((t) => (
                <div key={t.key} className="trait-slider">
                  <div className="trait-head">
                    <span className="trait-label">{t.label}</span>
                    <span className="trait-val">{(traits as any)[t.key]}/5</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={5}
                    value={(traits as any)[t.key]}
                    onChange={(e) =>
                      setTraits((prev) => ({ ...prev, [t.key]: Number(e.target.value) }))
                    }
                  />
                  <div className="trait-ends">
                    <span>{t.left}</span>
                    <span>{t.right}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <label className="field">
            <span className="field-label">지식 베이스 (RAG) — 선택</span>
            <input
              ref={fileRef}
              className="file-input"
              type="file"
              multiple
              accept=".txt,.pdf,.md"
              onChange={(e) => setFiles(e.target.files)}
            />
            {files && files.length > 0 && (
              <span className="small" style={{ marginTop: 4 }}>
                {files.length}개 파일 선택됨
              </span>
            )}
          </label>

          {status && <div className="status">{status}</div>}
        </div>

        <div className="modal-foot">
          <button className="ghost" onClick={onClose} disabled={busy}>
            취소
          </button>
          <button onClick={handleCreate} disabled={busy}>
            {busy ? "생성 중..." : "페르소나 생성하기"}
          </button>
        </div>
      </div>
    </div>
  );
}
