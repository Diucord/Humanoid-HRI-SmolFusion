<#
.SYNOPSIS
    start-demo.ps1을 Windows 작업 스케줄러에 등록/해제한다.

.DESCRIPTION
    기본은 '로그온 시 실행'. PC가 꺼져 있는 시간이 많다면 시각 고정 트리거는
    무의미하므로(꺼진 PC는 트리거되지 않음) 로그온 트리거가 현실적이다.

    -DailyAt 을 주면 매일 지정 시각 트리거로 등록한다. PC가 절전 상태면
    깨워서 실행하도록 설정하지만, 완전히 꺼져 있으면 실행되지 않는다.

.PARAMETER DailyAt
    "08:00" 형식. 지정 시 로그온 대신 매일 해당 시각에 실행.

.PARAMETER Unregister
    등록된 작업을 삭제한다.

.EXAMPLE
    .\register-task.ps1                    # 로그온 시 자동 실행 (권장)
    .\register-task.ps1 -DailyAt "08:00"   # 매일 8시 (PC가 켜져 있거나 절전일 때만)
    .\register-task.ps1 -Unregister        # 해제
#>
param(
    [string]$DailyAt,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$TaskName = "SmolFusion-Demo"
$Script   = Join-Path $PSScriptRoot "start-demo.ps1"

# 관리자 권한 확인
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "관리자 권한이 필요합니다. PowerShell을 '관리자 권한으로 실행' 후 재시도하세요." -ForegroundColor Yellow
    exit 1
}

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "'$TaskName' 작업을 삭제했습니다."
    } else {
        Write-Host "'$TaskName' 작업이 없습니다."
    }
    exit 0
}

if (-not (Test-Path $Script)) { Write-Host "스크립트 없음: $Script" -ForegroundColor Red; exit 1 }

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Script`""

if ($DailyAt) {
    $trigger = New-ScheduledTaskTrigger -Daily -At $DailyAt
    $desc = "매일 $DailyAt"
} else {
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $desc = "로그온 시"
}

# 절전 해제 후 실행 허용, 배터리에서도 중단하지 않음, 실패 시 재시도
$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)   # 무제한 (서버는 계속 떠 있어야 함)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "SmolFusion HRI 데모 자동 기동 (llama.cpp x3 + FastAPI + Tunnel + Vercel)" | Out-Null

Write-Host "'$TaskName' 등록 완료 — 트리거: $desc" -ForegroundColor Green
Write-Host ""
Write-Host "  즉시 테스트 : Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  상태 확인   : Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "  로그        : webapp\logs\start-demo-*.log"
Write-Host "  해제        : .\register-task.ps1 -Unregister"

if ($DailyAt) {
    Write-Host ""
    Write-Host "주의: PC가 완전히 꺼져 있으면 이 작업은 실행되지 않습니다." -ForegroundColor Yellow
    Write-Host "      절전 상태라면 WakeToRun 설정으로 깨어나 실행합니다 (BIOS 지원 필요)." -ForegroundColor Yellow
}
