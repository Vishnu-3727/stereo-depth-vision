# Detached launcher for ARM P Stage 1 pretrain. Stdout tee'd to stage1_stdout.log.
$py = "C:\Users\vishn\.pyenv\pyenv-win\versions\3.12.9\python.exe"
$script = "C:\Users\vishn\stereo_depth_vision\stage_b_armp\20260918T062146Z_stage1_pretrain\scripts\train_armp_stage1.py"
$log = "C:\Users\vishn\stereo_depth_vision\stage_b_armp\20260918T062146Z_stage1_pretrain\stage1_stdout.log"
Write-Host "=== ARM-P STAGE1 pretrain starting (detached) ===" -ForegroundColor Cyan
& $py $script | Tee-Object -FilePath $log
Write-Host "=== ARM-P STAGE1 pretrain process exited ===" -ForegroundColor Cyan
