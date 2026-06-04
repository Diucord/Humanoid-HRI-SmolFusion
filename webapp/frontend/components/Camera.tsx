"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { analyzeFrame, VisionResult } from "@/lib/api";

interface Props {
  sessionId: string;
  enabled: boolean;
  onVision: (v: VisionResult) => void;
  intervalMs?: number;
}

export default function Camera({
  sessionId,
  enabled,
  onVision,
  intervalMs = 3000,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [vision, setVision] = useState<VisionResult | null>(null);
  const [error, setError] = useState("");
  const busyRef = useRef(false);

  // 카메라 시작/정지
  useEffect(() => {
    if (!enabled) return;
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (e) {
        setError("카메라 접근 실패. 권한을 허용해 주세요.");
      }
    })();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [enabled]);

  const captureBlob = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const v = videoRef.current;
      const c = canvasRef.current;
      if (!v || !c || v.videoWidth === 0) return resolve(null);
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      const ctx = c.getContext("2d");
      if (!ctx) return resolve(null);
      ctx.drawImage(v, 0, 0, c.width, c.height);
      c.toBlob((b) => resolve(b), "image/jpeg", 0.7);
    });
  }, []);

  // 주기적 분석
  useEffect(() => {
    if (!enabled) return;
    const tick = async () => {
      if (busyRef.current) return;
      busyRef.current = true;
      try {
        const blob = await captureBlob();
        if (blob) {
          const result = await analyzeFrame(sessionId, blob);
          setVision(result);
          onVision(result);
        }
      } catch {
        // 무시 (다음 틱에 재시도)
      } finally {
        busyRef.current = false;
      }
    };
    const id = setInterval(tick, intervalMs);
    return () => clearInterval(id);
  }, [enabled, sessionId, intervalMs, captureBlob, onVision]);

  if (!enabled) {
    return (
      <div className="camera-wrap" style={{ display: "grid", placeItems: "center" }}>
        <span className="small">카메라 꺼짐</span>
      </div>
    );
  }

  return (
    <div className="camera-wrap">
      <video ref={videoRef} autoPlay playsInline muted />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <div className="camera-overlay">
        {error
          ? error
          : vision
          ? formatVision(vision)
          : "분석 중..."}
      </div>
    </div>
  );
}

function formatVision(v: VisionResult): string {
  if (!v.has_person) return "👤 사람 감지 안 됨";
  const age = v.age_group !== "unknown" ? v.age_group : "";
  const gender =
    v.gender === "male" ? "남성" : v.gender === "female" ? "여성" : "";
  const smile = v.is_smiling ? "😊" : "";
  const parts = [age, gender].filter(Boolean).join(" · ");
  return `👁 ${parts || "사람 감지"} ${smile}${v.scene ? " — " + v.scene : ""}`;
}
