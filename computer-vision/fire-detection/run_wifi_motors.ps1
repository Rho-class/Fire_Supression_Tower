# Launch detector over Wi-Fi. Start this first, then join Wi-Fi "StepperControl".
# Retries the ESP WebSocket until it answers (or times out).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$py = Join-Path $here ".venv\Scripts\python.exe"
$hostEsp = "192.168.4.1"
$deadline = (Get-Date).AddMinutes(5)

Write-Host "Waiting for ESP32 at http://$hostEsp/ (join Wi-Fi StepperControl now)..."
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "http://$hostEsp/" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            Write-Host "ESP32 web page is up. Starting detector..."
            & $py -u detect_candle_flame_c270.py --camera 0 --motors --wifi --wifi-host $hostEsp
            exit $LASTEXITCODE
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}
Write-Host "Timed out waiting for $hostEsp. Is the ESP powered and are you on StepperControl?"
exit 1
