"use client";
import { useEffect, useRef, useState, useCallback } from "react";

// Web Speech API 타입 (브라우저 내장, 기존 speech_recognition과 동일 엔진)
type SpeechRecognition = any;

export function useSpeechRecognition(lang = "ko-KR") {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interim, setInterim] = useState("");
  const recogRef = useRef<SpeechRecognition | null>(null);
  const onFinalRef = useRef<((text: string) => void) | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const SR =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SR) {
      setSupported(false);
      return;
    }
    setSupported(true);
    const recog = new SR();
    recog.lang = lang;
    recog.continuous = false;
    recog.interimResults = true;

    recog.onresult = (event: any) => {
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += t;
        else interimText += t;
      }
      setInterim(interimText);
      if (finalText) {
        setTranscript(finalText);
        setInterim("");
        onFinalRef.current?.(finalText);
      }
    };
    recog.onend = () => setListening(false);
    recog.onerror = () => setListening(false);

    recogRef.current = recog;
    return () => {
      try {
        recog.abort();
      } catch {}
    };
  }, [lang]);

  const start = useCallback((onFinal?: (text: string) => void) => {
    if (!recogRef.current) return;
    onFinalRef.current = onFinal || null;
    setInterim("");
    try {
      recogRef.current.start();
      setListening(true);
    } catch {}
  }, []);

  const stop = useCallback(() => {
    recogRef.current?.stop();
    setListening(false);
  }, []);

  return { supported, listening, transcript, interim, start, stop };
}
