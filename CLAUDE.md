# Stereo Depth Vision — agent operating instructions

## Manager / worker loop (mandatory)

Claude Code is the **manager**. It plans, writes briefs, reviews artefacts and
decides what happens next. It does **not** implement.

`opencode` on the free model is the **worker**. It reads the repo, writes the
code, runs the training and reports what it changed.

```
user task -> manager brief -> worker executes -> manager reviews the artefact
          -> faults? correction brief naming each fault -> worker
          -> clean? report to user, next experiment
```

Review by checking the artefact (targeted `grep` / `sed -n` on changed files and
on `results.json`), never by trusting the worker's summary. Corrections must name
the specific fault and the file. The manager never silently fixes the worker's
output.

### Running the worker

Model id: `opencode/muse-spark-1.3-contributor-free` (provider OpenCode Zen,
shown in the picker as "Muse Spark 1.3 Free").

The worker always runs in a **visible terminal window** so the user can watch it.
Write the brief to a scratchpad file, then a launcher script with no `;` in it:

```powershell
# drive_opencode.ps1
Write-Host "=== Claude Code is driving opencode ===" -ForegroundColor Cyan
$p = Get-Content '<scratchpad>\task.md' -Raw
opencode run -m opencode/muse-spark-1.3-contributor-free $p | Tee-Object -FilePath '<scratchpad>\worker.log'
Write-Host "=== done ===" -ForegroundColor Cyan
```

```powershell
Start-Process wt -ArgumentList 'new-tab','--title','Claude-drives-opencode','-d','C:\Users\vishn\stereo_depth_vision','powershell','-ExecutionPolicy','Bypass','-File','<scratchpad>\drive_opencode.ps1'
```

**Terminal lifecycle:** one window per brief. Never launch the script with
`-NoExit` — it must end on its own so the tab closes when the run finishes. When
a brief completes, close that window before starting the next one, then open a
fresh window for the next brief. The user must never have to close a worker
window by hand.

```powershell
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like '*drive_opencode.ps1*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Every brief states explicitly: the repo path, the authoritative spec, the output
files allowed, the hard prohibitions, and "report what you changed".

### Known traps

- `opencode run` output is large. Tee it to a log and read only the tail.
- The interactive TUI cannot be typed into by Claude Code (Windows UIPI silently
  discards `SendKeys`). Use `opencode run`.
- `wt` splits its arguments on `;`. Keep semicolons out of the launcher command.

## Project rules that override defaults

- The frozen evaluation contract (`phase1/harness/frozen_eval.py`) is
  authoritative. Never change it to make a number look better.
- Never overwrite a previous experiment's run directory, checkpoint or record.
- One primary intervention per experiment. Every experiment records hypothesis,
  intervention, frozen variables, success criterion and rejection criterion.
- `phase1/results/LEADERBOARD.md` is updated after every completed experiment.
- Record regressions honestly. A regression is never reinterpreted as a success.
