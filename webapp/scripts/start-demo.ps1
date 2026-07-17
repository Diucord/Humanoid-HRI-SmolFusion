<#
.SYNOPSIS
    SmolFusion 데모 전체 기동 (llama.cpp x3 + FastAPI + Cloudflare Tunnel + Vercel 갱신)

.DESCRIPTION
    로그온/부팅 시 자동 실행되도록 설계. 각 단계는 이전 단계의 헬스체크를 통과해야 진행한다.
    Quick Tunnel은 실행할 때마다 URL이 바뀌므로, 새 URL을 Vercel 환경변수에 반영하고
    프로덕션을 재배포한다.

.PARAMETER SkipVercel
    Vercel 환경변수 갱신/재배포를 건너뛴다. 로컬 확인만 할 때 사용.

.PARAMETER SkipTunnel
    터널도 띄우지 않는다 (로컬 전용). SkipVercel을 자동 포함.

.EXAMPLE
    .\start-demo.ps1                # 전체 기동 + Vercel 반영
    .\start-demo.ps1 -SkipTunnel    # 로컬만 (localhost:3000)
#>
param(
    [switch]$SkipVercel,
    [switch]$SkipTunnel
)

$ErrorActionPreference = "Stop"
if ($SkipTunnel) { $SkipVercel = $true }

# ===== 경로 =====
$Root     = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repo 루트
$WebApp   = Join-Path $Root "webapp"
$LlamaSrv = Join-Path $WebApp "llamacpp\llama-server.exe"
$Models   = Join-Path $WebApp "models"
$NxModels = Join-Path $Root "nx\models"
$Backend  = Join-Path $WebApp "backend"
$Frontend = Join-Path $WebApp "frontend"
$Tunnel   = Join-Path $WebApp "tunnel\cloudflared.exe"
$LogDir   = Join-Path $WebApp "logs"
$CondaPy  = "$env:USERPROFILE\anaconda3\envs\smolfusion\python.exe"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunLog = Join-Path $LogDir "start-demo-$Stamp.log"

function Log {
    param([string]$Msg, [string]$Level = "INFO")
    $line = "[{0}] [{1}] {2}" -f (Get-Date -Format "HH:mm:ss"), $Level, $Msg
    Write-Host $line
    Add-Content -Path $RunLog -Value $line -Encoding utf8
}

function Test-Port {
    param([int]$Port)
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $Port); $c.Close(); return $true
    } catch { return $false }
}

function Wait-Port {
    param([int]$Port, [string]$Name, [int]$TimeoutSec = 180)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        if (Test-Port $Port) { Log "$Name (:$Port) 준비 완료"; return $true }
        Start-Sleep -Seconds 2
    }
    Log "$Name (:$Port) ${TimeoutSec}s 내 응답 없음" "FAIL"
    return $false
}

function Start-Bg {
    # 주의: 파라미터명에 $Args를 쓰면 PowerShell 자동 변수와 충돌해 항상 비어 온다.
    param([string]$File, [string[]]$ArgList, [string]$LogName, [string]$Cwd = $Root)
    $out = Join-Path $LogDir "$LogName.log"
    Start-Process -FilePath $File -ArgumentList $ArgList -WorkingDirectory $Cwd `
        -WindowStyle Hidden -RedirectStandardOutput $out `
        -RedirectStandardError (Join-Path $LogDir "$LogName.err.log")
}

Log "===== SmolFusion 데모 기동 시작 ====="
Log "로그: $RunLog"

# ===== 0) 사전 점검 =====
foreach ($p in @($LlamaSrv, $CondaPy)) {
    if (-not (Test-Path $p)) { Log "필수 파일 없음: $p" "FAIL"; exit 1 }
}

# ===== 1) llama.cpp 서버 3개 =====
if (Test-Port 8081) { Log "VLM(:8081) 이미 실행 중 - 건너뜀" } else {
    Log "VLM 기동 (Qwen3-VL-4B, GPU)"
    Start-Bg $LlamaSrv @(
        "-m", "$Models\Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        "--mmproj", "$Models\mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
        "-ngl", "99", "-c", "4096", "--port", "8081"
    ) "vlm"
}

if (Test-Port 8080) { Log "LLM-igris(:8080) 이미 실행 중 - 건너뜀" } else {
    Log "LLM 기동 (igris 파인튜닝, GPU)"
    Start-Bg $LlamaSrv @(
        "-m", "$NxModels\qwen3-igris-1.7b.Q4_K_M.gguf",
        "-ngl", "99", "-c", "4096", "--port", "8080",
        "--alias", "qwen3-igris-1.7b"
    ) "llm-igris"
}

if (Test-Port 8082) { Log "LLM-general(:8082) 이미 실행 중 - 건너뜀" } else {
    Log "LLM 기동 (Qwen3-1.7B, CPU)"
    Start-Bg $LlamaSrv @(
        "-m", "$Models\Qwen3-1.7B-Q8_0.gguf",
        "-ngl", "0", "-c", "4096", "-t", "8", "--port", "8082",
        "--alias", "Qwen3-1.7B-Q8_0.gguf"
    ) "llm-general"
}

if (-not (Wait-Port 8081 "VLM"))         { exit 1 }
if (-not (Wait-Port 8080 "LLM-igris"))   { exit 1 }
if (-not (Wait-Port 8082 "LLM-general")) { exit 1 }

# ===== 2) FastAPI 백엔드 =====
if (Test-Port 8000) { Log "백엔드(:8000) 이미 실행 중 - 건너뜀" } else {
    Log "백엔드 기동 (FastAPI)"
    $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
    Start-Bg $CondaPy @("app.py") "backend" $Backend
}
if (-not (Wait-Port 8000 "백엔드" 120)) { exit 1 }

# ===== 3) 프론트엔드 =====
if (Test-Port 3000) { Log "프론트(:3000) 이미 실행 중 - 건너뜀" } else {
    Log "프론트엔드 기동 (Next.js)"
    $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if ($npm) { Start-Bg $npm @("run", "dev") "frontend" $Frontend }
    else { Log "npm 없음 - 프론트 건너뜀" "WARN" }
}
Wait-Port 3000 "프론트엔드" 90 | Out-Null

if ($SkipTunnel) {
    Log "===== 로컬 기동 완료 -> http://localhost:3000 ====="
    exit 0
}

# ===== 4) Cloudflare Tunnel =====
if (-not (Test-Path $Tunnel)) {
    Log "cloudflared.exe 없음: $Tunnel" "FAIL"
    Log "https://github.com/cloudflare/cloudflared/releases 에서 받아 배치" "HINT"
    exit 1
}

# Quick Tunnel은 실행할 때마다 새 URL을 발급한다. 기존 프로세스를 정리하지 않으면
# 터널이 중복 기동되어 어느 URL이 Vercel에 물려 있는지 추적할 수 없게 된다.
$existing = Get-Process cloudflared -ErrorAction SilentlyContinue
if ($existing) {
    Log "기존 cloudflared $($existing.Count)개 종료 (URL 중복 방지)"
    $existing | Stop-Process -Force
    Start-Sleep -Seconds 2
}

Log "Cloudflare Tunnel 기동"
$TunLog = Join-Path $LogDir "tunnel.log"
Remove-Item $TunLog, (Join-Path $LogDir "tunnel.err.log") -ErrorAction SilentlyContinue
Start-Bg $Tunnel @("tunnel", "--url", "http://localhost:8000") "tunnel"

# 로그에서 trycloudflare URL 추출 (최대 60초 대기)
$TunnelUrl = $null
$sw = [Diagnostics.Stopwatch]::StartNew()
while ($sw.Elapsed.TotalSeconds -lt 60) {
    Start-Sleep -Seconds 2
    foreach ($f in @($TunLog, (Join-Path $LogDir "tunnel.err.log"))) {
        if (Test-Path $f) {
            $m = Select-String -Path $f -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" `
                 -AllMatches -ErrorAction SilentlyContinue
            if ($m) { $TunnelUrl = $m.Matches[0].Value; break }
        }
    }
    if ($TunnelUrl) { break }
}

if (-not $TunnelUrl) {
    Log "터널 URL 추출 실패 - $TunLog 확인" "FAIL"
    exit 1
}

Log "터널 URL: $TunnelUrl"
Set-Content -Path (Join-Path $LogDir "current-tunnel-url.txt") -Value $TunnelUrl -Encoding utf8

# 터널이 실제로 백엔드를 뚫는지 확인.
# 터널 생성 직후에는 DNS가 아직 전파되지 않아 이름 해석이 실패한다. 재시도로 흡수.
$tunnelOk = $false
for ($i = 1; $i -le 10; $i++) {
    try {
        $r = Invoke-RestMethod -Uri "$TunnelUrl/health" -TimeoutSec 15
        Log "터널 헬스체크 통과 (시도 $i): $($r | ConvertTo-Json -Compress)"
        $tunnelOk = $true
        break
    } catch {
        if ($i -eq 10) { Log "터널 헬스체크 실패 (10회): $_" "WARN" }
        else { Start-Sleep -Seconds 3 }
    }
}
if (-not $tunnelOk) {
    Log "터널이 응답하지 않지만 Vercel 갱신은 계속합니다 (DNS 전파 지연일 수 있음)" "WARN"
}

if ($SkipVercel) {
    Log "===== 완료 (Vercel 갱신 생략) ====="
    Log "터널 URL: $TunnelUrl"
    exit 0
}

# ===== 5) Vercel 환경변수 갱신 + 재배포 =====
Push-Location $Frontend
try {
    # 인증 확인. 토큰이 만료되면 CLI가 대화형 로그인 플로우로 진입해 무한 대기하므로
    # 자동 실행 환경에서는 먼저 걸러내야 한다.
    $who = & npx vercel whoami 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Vercel 인증 없음. 터미널에서 'npx vercel login' 실행 후 재시도하세요."
    }
    Log "Vercel 계정: $($who | Select-Object -Last 1)"

    Log "환경변수 갱신 중..."
    # 기존 값 제거 (등록돼 있지 않으면 실패하지만 무시)
    & npx vercel env rm NEXT_PUBLIC_API_URL production --yes 2>&1 | Out-Null

    # 새 URL 주입 (stdin으로 값 전달)
    $TunnelUrl | & npx vercel env add NEXT_PUBLIC_API_URL production 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "env add 실패 (exit $LASTEXITCODE)" }
    Log "환경변수 반영: NEXT_PUBLIC_API_URL=$TunnelUrl"

    Log "프로덕션 재배포 중... (수 분 소요)"
    $deploy = & npx vercel --prod --yes 2>&1
    if ($LASTEXITCODE -ne 0) { throw "배포 실패 (exit $LASTEXITCODE)`n$deploy" }

    $url = ($deploy | Select-String -Pattern "https://[^\s]+\.vercel\.app" -AllMatches |
            Select-Object -Last 1).Matches.Value
    Log "배포 완료: $url"
    Log "===== 데모 준비 완료 ====="
    Log "  로컬  : http://localhost:3000"
    Log "  터널  : $TunnelUrl"
    Log "  공개  : $url"
} catch {
    Log "Vercel 단계 실패: $_" "FAIL"
    Log "로컬/터널은 살아 있음. 수동 재배포: cd webapp\frontend; npx vercel --prod" "HINT"
    exit 1
} finally {
    Pop-Location
}
