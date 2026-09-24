# Push E4 finetune seed 2 as soon as a Kaggle GPU slot frees (same pattern as
# autopush_e3_seed2.ps1). Kaggle refuses a third concurrent batch GPU session
# instead of queuing it. Polls seeds 0 and 1 every 5 minutes, pushes seed 2
# once either finishes, then exits. Hard stop after 4 hours (a 200-epoch
# run takes about 80 min).
#
# Uses the pyenv interpreter explicitly: bare `python` now resolves to
# Anaconda, which has no kaggle module.

$repo = 'C:\Users\vishn\stereo_depth_vision'
$py   = 'C:\Users\vishn\.pyenv\pyenv-win\versions\3.12.9\python.exe'
$sha  = '4c16fbe32724d0c1157aa91b0503f67e7e1576faf597325d62c87510aa6b829f'
$log  = Join-Path $repo 'stage_e_recipe\kaggle\autopush_e4ft_seed2.log'
$deadline = (Get-Date).AddHours(4)
$env:STAGE_E_KAGGLE_USER = 'vishnuvardhanksece'
Set-Location $repo

function Write-Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
}

Write-Log "autopush started; waiting for a free GPU slot"

while ((Get-Date) -lt $deadline) {
    $free = $false
    foreach ($s in 0, 1) {
        $r = & $py -m kaggle kernels status "vishnuvardhanksece/stage-e-e4-finetune-seed$s" 2>&1 | Out-String
        if ($r -match 'KernelWorkerStatus\.(\w+)') {
            $state = $Matches[1]
            if ($state -in @('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED')) {
                $free = $true
                Write-Log "seed$s is $state - a slot is free"
            }
        }
    }

    if ($free) {
        Write-Log "pushing e4ft seed 2"
        $out = & $py stage_e_recipe\kaggle\push_e4.py push-finetune 2 $sha 2>&1 | Out-String
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
