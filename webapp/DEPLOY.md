# Hera 배포 가이드 (Vercel + Cloudflare Tunnel)

```
포폴 사이트 → Vercel(프론트, 항상 켜짐) → Cloudflare Tunnel → 내 PC(백엔드+GPU)
```

프론트는 Vercel에 영구 배포, 백엔드는 내 PC에서 돌리고 터널로 노출.
**내 PC가 켜져 있을 때만** 데모 작동 (꺼지면 "오프라인" 배너 표시).

---

## 1. 백엔드 + 터널 켜기 (데모 시연 전마다)

### 1-1. llama.cpp 서버 3개 + FastAPI 백엔드 기동
(README.md의 실행 방법 참고)

### 1-2. Cloudflare Tunnel 실행
```powershell
cd webapp\tunnel
.\cloudflared.exe tunnel --url http://localhost:8000
```
→ 출력에서 `https://xxxx.trycloudflare.com` URL 확인 (이게 백엔드 공개 주소)

> ⚠️ **Quick Tunnel은 재시작하면 URL이 바뀜.**
> 고정 URL이 필요하면 아래 "고정 URL(Named Tunnel)" 참고.

---

## 2. 프론트 Vercel 배포 (최초 1회)

```powershell
cd webapp\frontend
vercel login          # 브라우저로 로그인 (GitHub/이메일)
vercel                # 첫 배포 — 질문에 기본값(Enter)으로 진행
```

배포 후 환경변수에 **터널 URL** 등록:
```powershell
vercel env add NEXT_PUBLIC_API_URL production
# 값 입력: https://xxxx.trycloudflare.com  (터널 URL, 끝에 / 없이)

vercel --prod         # 프로덕션 재배포 (환경변수 반영)
```

→ `https://hera-xxx.vercel.app` 같은 URL이 나옴. 이걸 포폴에 링크.

---

## 3. 터널 URL이 바뀌었을 때 (Quick Tunnel 사용 시)

cloudflared 재시작하면 URL이 바뀌므로, Vercel 환경변수도 갱신:
```powershell
vercel env rm NEXT_PUBLIC_API_URL production
vercel env add NEXT_PUBLIC_API_URL production    # 새 터널 URL 입력
vercel --prod
```

---

## 4. (권장) 고정 URL — Named Tunnel

매번 URL이 바뀌는 게 번거로우면 Cloudflare 계정 + 도메인으로 고정:

```powershell
cd webapp\tunnel
.\cloudflared.exe tunnel login                      # 브라우저 인증
.\cloudflared.exe tunnel create hera                # 터널 생성
.\cloudflared.exe tunnel route dns hera api.내도메인.com
# config.yml 작성 후:
.\cloudflared.exe tunnel run hera
```
→ `https://api.내도메인.com` 고정 주소. Vercel 환경변수도 이걸로 한 번만 설정.

도메인이 없으면 Cloudflare에서 무료 도메인은 안 주지만,
가비아/내도메인 등에서 싼 도메인(.com 약 1.5만원/년) 사면 됨.

---

## 체크리스트

- [ ] llama.cpp 3개 (8080/8081/8082) 기동
- [ ] FastAPI 백엔드 (8000) 기동
- [ ] cloudflared 터널 실행 → URL 확인
- [ ] Vercel 환경변수 `NEXT_PUBLIC_API_URL` = 터널 URL
- [ ] `vercel --prod` 재배포
- [ ] Vercel URL 접속 → "데모 서버 온라인" 초록 표시 확인
