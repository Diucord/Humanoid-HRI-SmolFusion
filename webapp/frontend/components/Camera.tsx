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
  intervalMs = 1500,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [vision, setVision] = useState<VisionResult | null>(null);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const busyRef = useRef(false);

  const startCamera = useCallback(async () => {
    setError("");
    setReady(false);

    // 1) 브라우저 지원 확인
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      // secure context (https/localhost)가 아니면 mediaDevices 자체가 없음
      if (typeof window !== "undefined" && !window.isSecureContext) {
        setError(
          "⚠️ 보안 컨텍스트가 아닙니다. 카메라는 https:// 또는 http://localhost 에서만 동작해요.\n" +
            `현재 주소: ${window.location.origin}\n→ http://localhost:${window.location.port} 로 접속하세요.`
        );
      } else {
        setError("이 브라우저는 카메라(getUserMedia)를 지원하지 않아요. Chrome/Edge를 권장해요.");
      }
      return;
    }

    // 2) 권한 요청 (여기서 브라우저 팝업이 뜸)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setReady(true);
    } catch (e: any) {
      const name = e?.name || "";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        setError(
          "🚫 카메라 권한이 거부되었어요.\n" +
            "주소창 왼쪽 자물쇠(또는 ⓘ) 아이콘 → 카메라 → '허용'으로 바꾸고 아래 '다시 시도'를 눌러주세요."
        );
      } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        setError("📷 연결된 카메라를 찾지 못했어요. 웹캠이 연결됐는지 확인해주세요.");
      } else if (name === "NotReadableError") {
        setError("⚠️ 카메라를 다른 앱(줌/팀즈 등)이 사용 중이에요. 해당 앱을 닫고 다시 시도해주세요.");
      } else {
        setError(`카메라 오류: ${name || e?.message || "알 수 없음"}`);
      }
    }
  }, []);

  // enabled 토글에 따라 시작/정지
  useEffect(() => {
    if (enabled) {
      startCamera();
    } else {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setReady(false);
      setVision(null);
    }
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [enabled, startCamera]);

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
    if (!enabled || !ready) return;
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
        // 무시 (다음 틱 재시도)
      } finally {
        busyRef.current = false;
      }
    };
    const id = setInterval(tick, intervalMs);
    return () => clearInterval(id);
  }, [enabled, ready, sessionId, intervalMs, captureBlob, onVision]);

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

      {error ? (
        <div className="camera-error">
          <div style={{ whiteSpace: "pre-wrap", marginBottom: 10 }}>{error}</div>
          <button onClick={startCamera}>🔄 다시 시도</button>
        </div>
      ) : !ready ? (
        <div className="camera-overlay">카메라 권한 요청 중... (팝업에서 '허용')</div>
      ) : (
        <div className="camera-overlay">
          {vision ? formatVision(vision) : "분석 중..."}
        </div>
      )}
    </div>
  );
}

function formatVision(v: VisionResult): string {
  if (!v.has_person) return "👤 사람 감지 안 됨";
  const age = v.age_group !== "unknown" ? v.age_group : "";
  const gender = v.gender === "male" ? "남성" : v.gender === "female" ? "여성" : "";
  const smile = v.is_smiling ? "😊" : "";
  const parts = [age, gender].filter(Boolean).join(" · ");
  return `👁 ${parts || "사람 감지"} ${smile}${v.scene ? " — " + v.scene : ""}`;
}
