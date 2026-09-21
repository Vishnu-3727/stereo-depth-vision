# Overnight helper: push E3 seed 2 as soon as a Kaggle GPU slot frees.
#
# Kaggle allows 2 concurrent batch GPU sessions and REFUSES a third rather than
# queuing it, so seed 2 cannot be submitted up front. This polls seeds 0 and 1
# every 5 minutes and pushes seed 2 when either finishes, then exits.
#
# Detached from Claude Code on purpose: a background shell inside the session
# can be reaped under local memory pressure, which already happened once.
#
# Self-terminating. Hard stop after 8 hours so it cannot linger.

$repo = 'C:\Users\vishn\stereo_depth_vision'
$log  = Join-Path $repo 'stage_e_recipe\kaggle\autopush_e3_seed2.log'
$deadline = (Get-Date).AddHours(8)
Set-Location $repo

function Write-Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
}

Write-Log "autopush started; waiting for a free GPU slot"

while ((Get-Date) -lt $deadline) {
    $free = $false
    foreach ($s in 0, 1) {
        $r = python -m kaggle kernels status "vishnu3727/stage-e-e3-seed$s" 2>&1 | Out-String
        if ($r -match 'KernelWorkerStatus\.(\w+)') {
            $state = $Matches[1]
            if ($state -in @('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED')) {
                $free = $true
                Write-Log "seed$s is $state - a slot is free"
            }
        }
    }

    if ($free) {
        Write-Log "pushing e3 seed 2"
        $out = python stage_e_recipe\kaggle\push_run.py push 2 e3 2>&1 | Out-String
        Write-Log $out.Trim()
        if ($out -match 'successfully pushed') {
            Write-Log "seed 2 pushed; autopush exiting"
            exit 0
        }
        Write-Log "push did not succeed; will retry in 5 minutes"
    }

    Start-Sleep -Seconds 300
}

Write-Log "deadline reached without pushing seed 2; exiting"
exit 1
