<#
.SYNOPSIS
    SmolFusion 데모 관련 프로세스 전체 종료.

.DESCRIPTION
    start-demo.ps1이 띄운 llama-server, cloudflared, 백엔드 python, Next.js dev를 정리한다.
    포트로 소유 프로세스를 찾아 종료하므로, 다른 용도의 python/node는 건드리지 않는다.
#>

$Ports = @{
    8080 = "LLM-igris"; 8081 = "VLM"; 8082 = "LLM-general"
    8000 = "백엔드"; 3000 = "프론트엔드"
}

foreach ($p in $Ports.Keys | Sort-Object) {
    $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) { Write-Host "[skip] :$p ($($Ports[$p])) 실행 중 아님"; continue }
    foreach ($c in $conns) {
        try {
            $proc = Get-Process -Id $c.OwningProcess -ErrorAction Stop
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            Write-Host "[kill] :$p ($($Ports[$p])) -> $($proc.ProcessName) PID $($proc.Id)"
        } catch {
            Write-Host "[warn] :$p 종료 실패 - $_"
        }
    }
}

# cloudflared는 포트를 리슨하지 않으므로 이름으로 종료
Get-Process cloudflared -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.Id -Force
    Write-Host "[kill] cloudflared PID $($_.Id)"
}

Write-Host "`n종료 완료."
