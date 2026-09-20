# TIER2-PILOT detached launcher. Runs the sequential pipeline
# (CONTROL -> ARM-P -> frozen eval -> pilot_results.json -> README append)
# with all output tee'd so it survives the launching session.
$ErrorActionPreference = "Continue"
$PY = "C:\Users\vishn\.pyenv\pyenv-win\versions\3.12.9\python.exe"
$PILOT = "C:\Users\vishn\stereo_depth_vision\stage_b_armp\20260919T012646Z_tier2_seed1"
Write-Host "=== TIER2-PILOT pipeline starting (sequential, detached) ===" -ForegroundColor Cyan
& $PY "$PILOT\scripts\run_pilot.py" | Tee-Object -FilePath "$PILOT\pipeline.log" -Append
Write-Host "=== TIER2-PILOT pipeline exited ===" -ForegroundColor Cyan
