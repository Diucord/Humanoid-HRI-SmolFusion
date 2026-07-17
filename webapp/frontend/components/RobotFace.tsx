"use client";

export type RobotState = "idle" | "listening" | "speaking" | "thinking" | "greeting";

interface Props {
  state: RobotState;
  size?: number;
}

/**
 * 글래스모피즘 둥근 로봇 캐릭터.
 * 상태에 따라 눈 모양이 바뀜 (레퍼런스 무드).
 */
export default function RobotFace({ state, size = 160 }: Props) {
  return (
    <div className={`robot-face state-${state}`} style={{ width: size, height: size }}>
      <div className="robot-glow" />
      <div className="robot-head">
        <div className="robot-screen">
          <Eyes state={state} />
        </div>
      </div>
      <div className="robot-shadow" />
    </div>
  );
}

function Eyes({ state }: { state: RobotState }) {
  // 상태별 눈 모양
  if (state === "speaking") {
    // 말하는 중: 위아래로 움직이는 눈 + 입
    return (
      <div className="eyes speaking">
        <span className="eye round" />
        <span className="eye round" />
      </div>
    );
  }
  if (state === "listening") {
    // 듣는 중: 반짝이는 별 눈
    return (
      <div className="eyes listening">
        <span className="eye sparkle">✦</span>
        <span className="eye sparkle">✦</span>
      </div>
    );
  }
  if (state === "thinking") {
    // 생각 중: 점 깜빡임 (감은 눈)
    return (
      <div className="eyes thinking">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    );
  }
  if (state === "greeting") {
    // 인사: 웃는 눈 (^^)
    return (
      <div className="eyes greeting">
        <span className="eye happy" />
        <span className="eye happy" />
      </div>
    );
  }
  // idle: 평범한 눈
  return (
    <div className="eyes idle">
      <span className="eye normal" />
      <span className="eye normal" />
    </div>
  );
}
