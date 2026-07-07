# GUIDE.md — How to Build This With Claude Code (for you)

This is your operating manual. It covers the full session roadmap, how to run each
session, and exactly what to type to instruct Claude Code. Pair it with two files
Claude Code reads: `CLAUDE.md` (its always-on memory) and `LAYER_SPECS.md` (the
detailed spec for each layer).

## The three documents and who reads them

| File | Who reads it | Purpose |
|------|--------------|---------|
| `CLAUDE.md` | Claude Code (auto, every session) | Objective, conventions, build order, current task |
| `LAYER_SPECS.md` | Claude Code (when told) | The full contract + acceptance tests for each layer |
| `GUIDE.md` | You | How to run sessions and give instructions |

The division of labor: **planning + architecture + reading results happens in the
Claude.ai chat**; **all code, tests, and training runs happen in Claude Code**.

## Session roadmap

One session = one layer. Don't start a layer until the one below it passes its
validation. Each session ends with a green validation script and a git commit.

| Session | Layer | You end up with | Depends on |
|---------|-------|-----------------|------------|
| 1 | Vehicle model | ✅ Done — validated car physics | — |
| 2 | Track environment | Silverstone from FastF1 + synthetic fallback | 1 |
| 3 | Conditions model | Tire compounds, degradation, fuel burn, weather | 1 |
| 4 | Gym environment | A Gymnasium env wrapping car+track+conditions | 1,2,3 |
| 5 | SAC driver | An SAC agent that learns to drive a lap | 4 |
| 6 | Training + curriculum | A generalist driver (domain-randomized setups/weather) | 5 |
| 7 | Telemetry calibration | Physics matched to real data within tolerance | 2,6 |
| 8 | Setup optimization | Best car setup per track+weather (the deliverable) | 6,7 |
| 9 | Showcase (optional) | Plots, reports, portfolio write-up | 8 |

Layers 5–6 are the RL centerpiece (the driver). Layer 7 is what lets you honestly
say "without physical testing." Layer 8 is the actual "fine-tune the car" outcome.

## The per-session loop

1. **In chat (here):** we finalize any open questions and confirm the layer is ready.
2. **Open Claude Code:** `cd f1-racing-rl && claude`, then `Shift+Tab` into Plan Mode.
3. **Paste the start prompt** (below). Review the plan + acceptance tests it proposes.
   Do NOT let it code yet.
4. **Approve** → it implements the layer and its test script. Review the diff.
5. **Run the validation** (it can run it itself). If red, paste the output, let it fix.
6. **When green:** commit, tick the status in `CLAUDE.md`, run `/clear`, and bring the
   validation output back to chat so we plan the next layer.

## Exactly what to type in Claude Code

**Start of every session (in Plan Mode):**
> Read CLAUDE.md and LAYER_SPECS.md. We're doing Layer N. Show me your implementation
> plan and the acceptance tests you'll write, following the spec. Do not write code yet.

**After you approve the plan:**
> Looks good. Implement Layer N and its validation script exactly as specced. Touch
> only the files for this layer.

**To run the check:**
> Run the Layer N validation script and show me the output.

**If it fails:**
> Here's the failure. Diagnose the root cause before changing code, then fix it and
> re-run. Don't loosen the acceptance thresholds to make it pass.

**To finish:**
> Update the Layer N status to done in CLAUDE.md, then give me a one-paragraph summary
> of what was built and the validation numbers.

Then, in your terminal: `git add -A && git commit -m "Layer N: <name>"`.

## Habits that keep it clean

- **Always review diffs** before accepting. You're the engineer; Claude Code is the hands.
- **One layer per session**, then `/clear`. CLAUDE.md re-grounds context every time, so
  a fresh context stays sharp.
- **Never let it weaken an acceptance threshold** to force a pass — that hides real bugs.
- **Commit at every green validation.** Small commits make it trivial to revert.
- **Keep training runs in Claude Code**, not chat: it can launch, monitor, and read logs.

## Troubleshooting

- *It edits unrelated files* → remind it: "only touch Layer N files."
- *FastF1 download fails* → it needs internet + the `data/fastf1_cache` dir; the sandbox
  can't reach the F1 API, so run those pulls on your own machine/Colab. The synthetic
  fallback lets tests pass offline.
- *Env is slow* → have it profile; the environment must do thousands of steps/sec on CPU.
- *Training diverges* → bring the tensorboard curves back to chat; we adjust the reward
  or hyperparameters here, not by guessing in Claude Code.

## When to come back to chat

After each layer, and any time a design choice appears (reward shaping, observation
design, curriculum stages, which setup params to optimize, calibration tolerances).
Those are planning decisions — we make them here, then hand Claude Code the update.
