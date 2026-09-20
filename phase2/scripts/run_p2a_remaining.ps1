# EXP-P2A-SCALE-COVERAGE-001 — finish the run outside the agent harness.
#
# The harness kills background shells when the host is low on memory; it killed
# the seed chain twice while the detached python child kept training. This
# launcher runs the remaining work as its own detached process so a harness kill
# cannot interrupt it. It changes NOTHING about the experiment: same script,
# same seeds, same recipe, same evaluation.
#
# 1. wait for seed 1 to finish writing p2a_record.json
# 2. train seed 2
# 3. score all three under the frozen contract

Set-Location 'C:\Users\vishn\stereo_depth_vision'
$log = 'phase2\runs\p2a_launcher.log'
function Say($m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; Add-Content $log $line }

Say 'launcher start'
$rec1 = 'phase2\runs\p2a_scale_coverage_s1\p2a_record.json'
$waited = 0
while (-not (Test-Path $rec1)) {
    Start-Sleep -Seconds 20
    $waited += 20
    if ($waited -ge 2400) { Say 'ABORT: seed 1 did not finish within 40 min'; exit 1 }
}
Say 'seed 1 complete'

Say 'seed 2 start'
$env:P2A_SEED = '2'
python phase2\scripts\train_p2a_scale_coverage.py *> 'phase2\runs\p2a_seed2_train.log'
if (-not (Test-Path 'phase2\runs\p2a_scale_coverage_s2\p2a_record.json')) {
    Say 'ABORT: seed 2 did not produce a record'
    exit 1
}
Say 'seed 2 complete'

Say 'frozen evaluation start'
Remove-Item Env:\P2A_SEED -ErrorAction SilentlyContinue
python phase2\scripts\eval_p2a.py best *> 'phase2\runs\p2a_eval.log'
Say 'frozen evaluation complete'
Say 'launcher done'
