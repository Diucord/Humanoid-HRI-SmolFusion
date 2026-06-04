import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Humanoid HRI · SmolFusion",
  description: "실시간 멀티모달 HRI 데모 — Vision · RAG · 페르소나 대화",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
