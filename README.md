# TypeSafe Mario

An experimental controller that lets TypeSafe's Jev model directly choose NES
controller inputs for the original Super Mario Bros.

The model does **not** receive screenshots. The harness translates emulator telemetry
and RAM into compact, object-centric JSON containing Mario's motion, jump trajectory,
upcoming enemies, terrain, measured response delay, recent-control results, and episode
progress. Jev chooses one of the legal controller actions, the emulator advances several
frames, and the loop repeats. The raw local tile grid remains available in debug logs
and the UI, but is not duplicated in the model input.

## Architecture

```text
NES emulator -> telemetry/RAM parser -> structured JSON -> Jev Choice -> controller input
```

The initial action set is intentionally small:

- `noop`
- `right`
- `right_jump`
- `right_run`
- `right_run_jump`
- `jump`
- `left`

## Requirements

- Python 3.13 or newer
- A TypeSafe API key in `TYPESAFE_API_KEY`
- A legal local setup for Super Mario Bros.

This repository contains no Nintendo ROM or other copyrighted game data. You are
responsible for ensuring that your emulator and game files are obtained and used
lawfully.

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[mario,dev]"
$env:TYPESAFE_API_KEY = "your-key"
```

Inspect the exact JSON and text that will be sent to TypeSafe without launching the
game or calling the API:

```powershell
.venv\Scripts\typesafe-mario state-demo
```

Run World 1-1 with eight emulator frames per control cycle. The default
display is a single recordable window with the live game and model telemetry:

```powershell
.venv\Scripts\typesafe-mario play --env SuperMarioBros-1-1-v0 --frames-per-decision 8
```

All display modes use the same cycle loop. At each cycle's start, Jev receives the
current state while the emulator executes the previous decision for exactly eight
frames (or the configured `--frames-per-decision`). The first cycle uses `noop`.
The next action starts only at the cycle boundary: an early response waits for the
boundary, and a late response pauses the emulator there until it arrives. Death or
level completion can end a cycle early. The last requested action still gets its
cycle before `--max-decisions` ends the run.

State parsing runs once per emulator frame. Speeds use pixels per frame, and
airtime, action duration, and stalled time count actual emulator frames. Waiting
does not advance these counters or the game clock. Observation-to-action delay is
one control cycle; API latency is recorded separately in wall-clock milliseconds.
The dashboard continues handling quit and restart controls while paused.

Each request includes `committed_control`: the macro already running during the
next-choice delay, its first actual frame action, and the frames until the new choice
can apply. Jump macros hold A on ascent, release A during observed descent, and press
again on landing, including landings inside a cycle. Button releases consume the
same frame budget. This prepares a new jump; it does not guarantee takeoff if Mario
has already left the edge before the game processes the press.

The dashboard shows the selected action, full Choice probability distribution,
confidence, Jev latency, jump probability, danger score, reward, and parsed game
state. Press `R` or click **Restart** for a fresh episode; the dashboard remains open
after death or level completion. Press `Esc` or `Q` to quit. Use `--display game` for
only the emulator window or `--display none` for a headless benchmark.

Each decision is written to `artifacts/run-<timestamp>.jsonl`. These records include
latency, action probabilities, confidence, canonical model state, raw debug state, and
game outcome, providing the data for a live overlay or rendered social clip.

The companion `run-<timestamp>.trace.jsonl` records environment/seed, package versions,
source hashes, episode boundaries, inference requests, action application, and every
emulated frame's actual action, telemetry, and 2 KiB RAM dump (hex). It includes the
initial NOOP cycle, jump-button release frames, and requests discarded at termination
or restart. Trace `episode` and `decision` identify requests across restarts; decision
records also include `execution` frame references. A frame number counts completed
`env.step` calls: `apply_frame: 8` means the first action step is frame 9. Waiting for
inference does not increment this counter. Request events contain the model input
and the action committed for the intervening cycle; these diagnostics do not change
the policy input. Full traces use roughly 0.3 MB per simulated second at 60 fps.

## What the parser produces

TypeSafe accepts JSON directly, so there is no need to flatten telemetry into prose.
The model-facing object groups observations by meaning:

- `player`: position, velocity, grounded state, jump phase, and power-up
- `trajectory`: airtime, distance since takeoff, and committed gap crossing
- `hazard`: up to three enemies, projected positions, contact timing, and takeoff deadline
- `terrain`: obstacle/pit geometry, lower landings, observation reliability, and last grounded preview
- `reaction_timing`: action duration and observation-to-action delay in emulator frames
- `recent_control`: chosen action, duration, progress gained, and observed outcome
- `committed_control`: already committed macro, first frame action, and next-choice delay
- `episode`: lives, clock, progress, stalls, death, and level completion

Terrain checks the complete RAM block-buffer columns below a missing surface to
distinguish a lower landing (`drop_*`) from a pit. With incomplete RAM, a gap remains
`gap_kind: "unknown"`; it is not confirmed to be a pit. Coins, hidden blocks, and
climbable tiles do not count as lower landing support.
Here `pit` means no static block support at or below the current reference surface;
it does not rule out higher blocks or moving object platforms.

Hazard timing uses valid collision-box edges and RAM fractional speed for normally
walking Goombas/green Koopas, with labeled fallbacks. Contact timing assumes constant
velocity; acceleration and direction changes can invalidate it. Landing estimates
project RAM descent onto static tiles at constant horizontal speed; ascent, missing
support, or projected enemy contact returns unknown. Jump urgency includes landing
during the committed action delay, and remains true when the clearance budget is missed.

Grounded terrain history includes sample frame, sample X, age, displacement and
absolute gap edges. An unknown right edge means the observed width is incomplete.
The debug view additionally exposes enemy RAM state, collision boxes, vertical
overlap, player foot probes and complete static support columns. These diagnostic
additions do not replace the restored `hazard` model interface. Landing estimates
reject side/interior entry into staircase tiles; this independent physical-estimate
correction does not change the action-selection policy.

For humans, the fuller debug snapshot can still be rendered as compact text:

```text
Goal: Reach the flag in World 1-1 without dying.
Mario: x=172 y=79, moving right, airborne=False, status=small
Progress: 172 (best 172), time=387, lives=2
Nearby enemies: goomba 42px ahead
Local grid (# solid, . empty, E enemy, M Mario):
...........
...........
...........
..M..E.....
###########
```

The structured object is canonical; the text view is only for debugging and UI.

## TypeSafe judgments

Each request evaluates three independent judgments over the same state:

- a `Choice` selects the next controller macro;
- a `Noul` estimates whether a forward jump is useful now;
- a `Score` measures immediate danger for the live visualization.

The restored policy contract includes heuristic hazard timing and jump-urgency
fields. These are estimates and policy assumptions, not RAM observations. Jev
selects the controller macro; there is no scripted recovery-action override.
The existing height and future-support limitations remain unresolved.

## Development

```powershell
.venv\Scripts\ruff format --check src tests
.venv\Scripts\ruff check src tests
.venv\Scripts\python -m pytest -q
```
