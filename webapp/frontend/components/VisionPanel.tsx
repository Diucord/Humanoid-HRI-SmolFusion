"use client";
import { VisionResult } from "@/lib/api";
import { SmileIcon, NeutralIcon, CheckIcon, PersonIcon } from "@/components/Icons";

interface Props {
  vision: VisionResult | null;
  newPersonFlash: boolean;
  loading?: boolean;
  error?: string;
}

export default function VisionPanel({ vision, newPersonFlash, loading, error }: Props) {
  const present = vision?.has_person ?? false;

  // 값은 영어 원본 그대로 (middle aged, male, smiling 등)
  const age = vision && vision.age_group !== "unknown" ? vision.age_group : "-";
  const gender = vision && vision.gender !== "unknown" ? vision.gender : "-";

  return (
    <div className="vision-panel">
      <div className="vp-head">
        <span className={`vp-dot ${present ? "on" : "off"}`} />
        <span className="vp-title">Vision Analysis</span>
        {newPersonFlash && <span className="vp-newperson">NEW PERSON</span>}
      </div>

      <div className="vp-grid">
        <Stat
          label="present"
          value={present ? "yes" : "no"}
          highlight={present}
        />
        <Stat
          label="people"
          icon={vision?.person_count ? <PersonIcon size={15} /> : undefined}
          value={vision ? `${vision.person_count}` : "-"}
        />
        <Stat label="age" value={age} />
        <Stat label="gender" value={gender} />
        <Stat
          label="smile"
          icon={
            vision
              ? vision.is_smiling
                ? <SmileIcon size={16} />
                : <NeutralIcon size={16} />
              : undefined
          }
          value={vision ? (vision.is_smiling ? "smiling" : "neutral") : "-"}
        />
        <Stat
          label="face"
          icon={vision?.face_detected ? <CheckIcon size={15} /> : undefined}
          value={vision?.face_detected ? "" : "-"}
        />
      </div>

      {loading && <div className="vp-scene">⏳ analyzing...</div>}
      {error && <div className="vp-scene" style={{ color: "#e06a7a" }}>⚠️ {error}</div>}
      {!loading && !error && vision?.scene && (
        <div className="vp-scene">{vision.scene}</div>
      )}
      {!loading && !error && !vision && (
        <div className="vp-scene vp-dim">카메라/이미지를 입력하면 분석돼요.</div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  icon,
  highlight,
}: {
  label: string;
  value: string;
  icon?: React.ReactNode;
  highlight?: boolean;
}) {
  return (
    <div className={`vp-stat ${highlight ? "hl" : ""}`}>
      <div className="vp-label">{label}</div>
      <div className="vp-value">
        {icon}
        {value && <span>{value}</span>}
      </div>
    </div>
  );
}
