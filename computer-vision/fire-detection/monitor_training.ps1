param(
    [string]$RunDirectory = ".\runs\candle_yolo_public_960",
    [int]$PlannedEpochs = 120,
    [int]$RefreshSeconds = 5,
    [int]$EarlyStoppingPatience = 25
)

$resolvedRun = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $RunDirectory))
$resultsFile = Join-Path $resolvedRun "results.csv"

while ($true) {
    Clear-Host
    Write-Host "Fire Tower - YOLO Training Monitor" -ForegroundColor Cyan
    Write-Host "Updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Run: $resolvedRun"
    Write-Host "Planned maximum: $PlannedEpochs epochs"
    Write-Host "Early stopping: $EarlyStoppingPatience epochs without validation improvement"
    Write-Host ""

    if (Test-Path $resultsFile) {
        $rows = @(Import-Csv $resultsFile)
        if ($rows.Count -gt 0) {
            $last = $rows[-1]
            $epochProperty = $last.PSObject.Properties.Name | Where-Object { $_.Trim() -eq "epoch" } | Select-Object -First 1
            $epoch = if ($epochProperty) { [int]$last.$epochProperty } else { $rows.Count }
            $percent = [math]::Round(100 * $epoch / $PlannedEpochs, 1)
            Write-Host "Completed epochs: $epoch / $PlannedEpochs ($percent%)" -ForegroundColor Green
            Write-Host ""
            $last | Format-List
        } else {
            Write-Host "Epoch 1 is running; metrics are written after validation."
        }
    } else {
        Write-Host "Epoch 1 is running; results.csv has not been written yet."
    }

    Write-Host "GPU" -ForegroundColor Yellow
    nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
    Write-Host ""
    Write-Host "Press Ctrl+C to close this monitor. Training will continue."
    Start-Sleep -Seconds $RefreshSeconds
}
