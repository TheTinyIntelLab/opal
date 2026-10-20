$ErrorActionPreference = 'Stop'
$taskRoot = 'E:/opal-research-next-20261021'
$repoRoot = 'C:/Users/USER/Desktop/code/thetinyintellab/opal'
$pythonPath = Join-Path $repoRoot '.venv/Scripts/python.exe'
$nativeRoot = Join-Path $taskRoot 'native-block-v6'
$familyRoot = Join-Path $taskRoot 'family-v1'
$receiptPath = Join-Path $taskRoot 'postprocess.json'
$startedPath = Join-Path $taskRoot 'postprocess_started.json'
if (Test-Path -LiteralPath $startedPath) { throw 'Preserve the previous continuation attempt' }
$steps = [System.Collections.Generic.List[object]]::new()
$receipt = [ordered]@{started_utc=[DateTime]::UtcNow.ToString('o');status='waiting';steps=$steps}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $startedPath -Encoding utf8
Set-Location -LiteralPath $repoRoot
$env:TEMP = Join-Path $taskRoot 'temp'
$env:TMP = $env:TEMP
try {
    $deadline = [DateTime]::UtcNow.AddSeconds(6000)
    while ($true) {
        $worker = Get-CimInstance Win32_Process -Filter 'ProcessId=28188'
        if ($null -eq $worker -or $worker.CommandLine -notlike '*resume_native_blocks.py*fit*native-block-v6*') { break }
        if ([DateTime]::UtcNow -ge $deadline) { throw 'Bounded native wait expired; no scoring performed' }
        Start-Sleep -Seconds 10
    }
    if (Test-Path -LiteralPath (Join-Path $nativeRoot 'failure.json')) { throw 'Native fit failed; preserve its receipt and do not score' }
    if (-not (Test-Path -LiteralPath (Join-Path $nativeRoot 'selection.json'))) { throw 'Native process ended without a complete selection manifest' }
    $commands = @(
        @{name='family-select';script='tools/select_block_family.py';action='select';output=$familyRoot},
        @{name='native-score';script='tools/resume_native_blocks.py';action='score';output=$nativeRoot},
        @{name='native-replay';script='tools/resume_native_blocks.py';action='replay';output=$nativeRoot},
        @{name='family-score';script='tools/select_block_family.py';action='score';output=$familyRoot},
        @{name='family-replay';script='tools/select_block_family.py';action='replay';output=$familyRoot}
    )
    foreach ($command in $commands) {
        $logPath = Join-Path $taskRoot ($command.name + '.log')
        if (Test-Path -LiteralPath $logPath) { throw 'Preserve prior command logs' }
        $stepStart = [DateTime]::UtcNow.ToString('o')
        & $pythonPath $command.script $command.action --output $command.output *> $logPath
        $exitCode = $LASTEXITCODE
        $steps.Add([ordered]@{name=$command.name;started_utc=$stepStart;completed_utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitCode;log=$logPath;log_sha256=(Get-FileHash -LiteralPath $logPath -Algorithm SHA256).Hash.ToLowerInvariant()})
        $receipt.status = 'running'
        $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
        if ($exitCode -ne 0) { throw ('Registered command failed: ' + $command.name) }
    }
    $receipt.status = 'complete'
} catch {
    $receipt.status = 'blocked'
    $receipt.error = $_.Exception.Message
} finally {
    $receipt.completed_utc = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
}
