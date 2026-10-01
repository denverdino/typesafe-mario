# TypeSafe Mario

An experimental controller that lets TypeSafe's Jev model directly choose NES
controller inputs for the original Super Mario Bros.

The model does **not** receive screenshots. The harness translates emulator telemetry
and RAM into compact, object-centric JSON containing Mario's motion, jump trajectory,
upcoming enemies, terrain, measured response delay, recent-control results, and episode
progress. Jev chooses one of the legal controller actions, the emulator advances several
frames, and the loop repeats. The raw local tile grid remains available in debug logs
and the UI, but is not duplicated in the model input.

All display modes use fixed emulator-frame cycles controlled only by
`--frames-per-decision` (default: 8). The first decision is requested while paused.
When action A starts, the harness immediately asks Jev to choose action B using the
current state, action A, its actual first-frame button input, and the number of frames
until B will apply. Action A runs for exactly that many frames while B is being planned.
An early response waits for the boundary; a late response pauses the emulator at the
boundary until it arrives. Network delay changes wall-clock pace, not action duration.
Death or stage completion stops the cycle; landing does not shorten it.

At 60 FPS, eight frames represent about 133 ms of game time. Larger cycles require
longer-range predictions; smaller cycles shorten the prediction horizon and increase
API request frequency. `--api-run-ms` has been removed. The dashboard continues to
refresh and accept restart/quit input while waiting. Headless and game-only modes use
the same cycle semantics.

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

On macOS, with Python 3.13 installed, run these commands from the repository root:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[mario,dev]"
export TYPESAFE_API_KEY="your-key"
```

In each new terminal session, activate the virtual environment and export your API
key again before running the commands below.

Inspect the exact JSON and text that will be sent to TypeSafe without launching the
game or calling the API:

```sh
typesafe-mario state-demo
```

Run World 1-1 with Jev making a decision every eight emulator steps. The default
display is a single recordable window with the live game and model telemetry:

```sh
typesafe-mario play --env SuperMarioBros-1-1-v0 --frames-per-decision 8
```

Use `--frames-per-decision 4` for a shorter prediction horizon, or `16` for a longer one.

The dashboard shows the selected action, full Choice probability distribution,
confidence, Jev latency, jump probability, danger score, reward, and parsed game
state. Press `R` or click **Restart** for a fresh episode; the dashboard remains open
after death or level completion. Press `Esc` or `Q` to quit. Use `--display game` for
only the emulator window or `--display none` for a headless benchmark.

Each decision is written to `artifacts/run-<timestamp>.jsonl`. These records include
latency, action probabilities, confidence, canonical model state, raw debug state, and
game outcome, providing the data for a live overlay or rendered social clip.

### Retry from a checkpoint

Use a prior fixed-frame run log to start directly before a selected decision:

```sh
typesafe-mario play --env SuperMarioBros-1-2-v0 --frames-per-decision 8 \
  --resume-log artifacts/exit-pipe-1-2/run-20261001T154823.274866Z.jsonl \
  --resume-decision 152
```

Decision numbers are zero-based and match the log's `decision` field. This example
restores Mario at x=2060, before the early jump that led to the x=2144 collision.
The first load replays recorded **actual per-frame buttons**, using the original seed,
without API calls. Environment and frame cadence must match the source log, and a
position mismatch stops replay. Logs containing multiple episodes require
`--resume-episode <episode_id>`. Select a point before death or stage completion.

The resulting in-memory snapshot includes the native emulator, reward bookkeeping,
Gym frame counter and parser history (velocities, enemies and held buttons).
Press **R / Restart** to restore this snapshot directly, without replay or returning
to the beginning of the level. The old pending decision is discarded and a fresh
immediate decision is requested; subsequent decisions retain normal fixed-frame
lookahead timing. Retry logs start their own decision/frame numbering at zero and
record their origin in `run_config.resume`.

The installed native snapshot API does not support serialization to a standalone
save file. Across process launches, retain the source logs: they reconstruct the
checkpoint, including ancestry when resuming a log that itself began at a checkpoint.
Use the same emulator/ROM version for reproducible replay. Without resume options,
Restart continues to begin a fresh level as before.

Log schema version 2 uses `record_type=decision` and `record_type=episode_end`.
Older control versions may include `decision_discarded` records for landing replans;
fixed-frame runs do not discard responses merely because Mario lands.
Each episode has a unique `episode_id`, including dashboard restarts. Decision records
keep the input `state`, the actual execution start state, per-frame controller actions,
executed frame count, measured response delay, reward, and `result_state`. Frames spent
holding an old action during the next request belong to that old decision. Terminal
records preserve death, stage clear, decision limit, restart, quit, or policy error,
even when no further response arrives. `run_config` records environment, seed, display,
action cadence, control mode, policy type, and prompt/state versions. Readers of the old
format should filter decision records before reading action probabilities.

## What the parser produces

TypeSafe accepts JSON directly, so there is no need to flatten telemetry into prose.
The model-facing object groups observations by meaning:

- `player`: position, center x, per-frame velocity, RAM-confirmed grounded state, jump phase,
  power-up, held jump button, and release/repress requirement
- `trajectory`: airtime, distance since takeoff, committed gap crossing, and an approximate
  descending landing projection
- `hazard`: up to three enemies, projected positions, contact timing, and takeoff deadline
- `terrain`: contiguous gaps, visible far edges, static landing surfaces and safe center
  intervals, observation reliability, and a grounded preview rebased to current position
  with its observation age
- `reaction_timing`: action duration and measured observation-to-action delay
- `recent_control`: chosen action, duration, progress gained, and observed outcome
- `episode`: lives, clock, progress, stalls, death, and level completion

For humans, the fuller debug snapshot can still be rendered as compact text:

```text
Goal: Complete the current stage without dying.
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

- `next_action` (`Choice`) selects a controller macro, prioritizing survival and a supported
  landing before forward progress. During ascent it explicitly maintains the forward jump;
  clear supported ground favors running. These rules apply to the predicted state at action
  application time. Future requests keep all supplied actions available because Mario may
  land or take off during the scheduled action. Only immediate falling-state requests filter
  redundant jump buttons when non-jump choices are available.
- `jump_intent` (`Choice`) distinguishes `start`, `hold`, `release`, and `none`. It is diagnostic
  and does not override the selected controller action. The dashboard's start/hold percentage
  is the sum of those two probabilities, also retained as `jump_needed_probability` in logs.
- `danger` (`Score`) assesses the risk if the scheduled action continues over the reaction
  horizon: 0 = supported travel, 1 = plausible threat, 2 = imminent damage or death. The UI
  normalizes this score to a bar; it is not a calibrated death probability.

Questions distinguish observation time, action application at D, and cycle completion at
D+H. Jump continuity applies when Mario is predicted to still be rising at application.
Landing safety takes priority over repeating the previous jump or continuing to run.

When a future landing leaves inadequate space before an enemy ahead, the questions add
an approximate `landing_reference`: signed enemy distances at
touchdown and at D+H, vertical separation at the landing surface, and space for about eight
frames of takeoff clearance. The early-airborne-braking rule applies only when the
estimated touchdown falls inside the requested cycle, after the committed action; a
touchdown or stomp before application can already have changed the trajectory. Ordinary
states retain the established questions. Safe touchdown alone does not imply safe travel
for the rest of the cycle. These estimates assume unchanged relative velocity and enemy height; they
neither simulate candidate controls nor override Jev's selected action. Enemies behind
Mario at touchdown must not trigger braking back toward them.

For falling enemies (including ones just behind Mario), `hazard.overhead_threats`
reports measured vertical motion and possible platform-edge drops into the requested
action cycle. Screen-coordinate vertical speed is positive downward; relative vertical
speed also includes Mario's movement. The first observation, slot reuse, and screen
wraps leave motion unknown. Goombas and green koopas near a visible platform edge can
warn before a fall starts; ordinary red koopas turn at edges instead. Horizontal
extrapolation checks the interval D through D+H. A falling enemy projected to be below
Mario before application does not trigger this escape rule. Both estimates remain approximate.

Only while this warning is present, the questions suspend normal jump continuity:
prefer releasing A and moving underneath on supported ground, rather than rising into
the enemy or retreating toward one behind. Required pit-crossing height still takes
precedence. The provider retains all applicable action choices; there is no hardcoded
enemy-avoidance action override.

`terrain.gap_takeoff_window` supplies a deadline when grounded forward travel would
pass a visible floor's safe takeoff edge by D+H. It projects Mario's center at D and
D+H using the observed speed, rather than waiting for the current `gap_ahead` tile
threshold. If the window is still open at D, the questions prioritize a forward jump
for a reachable crossing (or braking if the route is blocked). A scheduled jump,
airborne state, unknown support, or a floor that does not extend to the gap cannot
trigger this cue. A missed window is reported separately and does not promise that
an airborne jump is possible. These constant-speed estimates are conservative and
do not simulate acceleration, collisions, or jump reachability.

Piranha plants use SMB enemy ID `0x0d` (`0x12` is Spiny). Their RAM movement flag,
direction, endpoint positions and pause timer distinguish `rising`, `extended`,
`retracting`, and `hidden`; a zero per-frame velocity alone cannot identify the phase.
An emergence that has started is already `rising`, even before its first movement frame.
`hazard.pipe_plants` includes nearby plants, pipe height and approximate phase timing.
At an occupied pipe within 48 pixels, the questions switch to waiting guidance:
release A, land/wait on supported ground beside the pipe, then jump once the plant is
fully hidden. This replaces the ordinary repeat-jump and blocked-progress guidance
for that state. A plant still retracting is not yet hidden. A hidden, idle plant does
not start emerging while Mario remains within 33 horizontal pixels; moving away can
remove that protection. Timing does not guarantee a safe landing, and an airborne
Mario already safely above the plant should continue toward support. As with other
questions, the provider chooses the action; there is no automatic button override.

`terrain.side_exit_pipe` preserves the horizontal mouth metatiles (`0x1c`/`0x1f`)
that the binary collision grid otherwise treats as an ordinary wall. Near that mouth,
questions replace repeated-jump guidance with entry alignment: release A, land at the
entry floor height and walk RIGHT into the pipe. If Mario is already on top of the lip,
retreat LEFT onto the supported approach, descend, then turn RIGHT to enter. The cue
includes the mouth position, floor height, retreat target and nearby support; positions
come from observed tiles rather than hardcoded World 1-2 coordinates. Engine state 2
reports automatic side-pipe entry. A pipe transition is not completion: resume ordinary
navigation after emergence and continue to the flag. The model still chooses buttons.

`terrain.stair_approach` prevents a premature slow jump before distant ascending
stairs. It requires predicted ground support at action application, walking speed,
at least three contiguous 16-pixel rises, and a supported approach cycle. While the
first step is still more than 48 pixels away, the questions prefer walking closer
instead of immediately bouncing after landing. Once closer, ordinary jump guidance
resumes. Immediate enemies can require a running takeoff; adding B only after launch
is not assumed to repair the trajectory. The cue is approximate and does not promise
enemy clearance. In particular, stomping the first enemy in a group does not guarantee
clearance over the next, higher enemy.

When taking off from a raised step with lower ground available before a visible gap,
`terrain.jump_reach_reference` compares
the predicted takeoff position, current horizontal speed and landing height with an
approximate fully held jump arc. Horizontal speed comes from the fractional RAM value
when available, avoiding one-frame integer rounding (for example, 1.75 versus 2 px/frame).
The cue applies only when Mario is on support at application, including a predicted
landing during the committed cycle; it does not ask for another takeoff in midair.
If the jump appears too short and the next cycle has a continuous same/lower walkway
before the safe gap edge, the questions prefer moving right without A before jumping.
If the far bank is not yet visible, its distance and height remain unknown; a supported
approach can reveal it before committing to a jump. When there are multiple landing
heights at the far edge, an overhead block does not conceal a usable lower floor.
Descending approaches also check that Mario can land and reach the next action boundary
within the lower floor's safe interval.
Flat-ground approaches continue to use the existing gap takeoff deadline.
This estimate ignores acceleration, ceilings, walls and enemies and is not proof of
reachability. A lower step may require landing again before the next jump; running
forward blindly can exhaust that takeoff window.

Before a visible gap within 128 pixels, `precision_landing_target` selects the highest
forward platform up to 64 pixels above Mario and keeps its world coordinates throughout
the jump. At four pixels above that platform, `precision_target_cleared` signals enough
height to release A and land before attempting the gap. The signal stops once Mario passes
the platform's safe center interval, with `precision_target_missed` reporting that case.
The target is re-evaluated on ground. These are approximate planning cues; normal
enemy-clearing jumps still hold A through ascent.

Both headless and dashboard modes parse every executed emulator frame and use the same
jump release/repress logic. Even one pixel of upward motion is classified as rising so
slow ascent does not imply that A should be released. A grounded jump macro releases a previously held A for one
frame before pressing it again. With a one-frame macro, the following decision must choose
a jump again to press A. `reaction_timing.frames_until_action`, `scheduled_action` and
`scheduled_first_frame_action` describe the committed movement before the requested
action applies. Player and terrain fields remain observations of the current frame,
not invented future telemetry. The model must predict the future state from this context.
The reaction horizon uses the planned delay plus the next action cycle; historical
`last_inference_delay_frames` is retained only as telemetry. Logs retain both request
state and actual application state, so the prediction horizon can be checked directly.

Geometry uses the full vertical collision buffer separately from the actor overlay, so
jumping does not hide the floor. Coins, invisible blocks and climbable tiles are excluded
from landing support using the original game's [collision rules](https://gist.github.com/1wErt3r/4048722).
Separate holes are not combined, and a lower platform is distinguished from a pit.
Unknown support never means a clear path. Surface coordinates describe static terrain;
`safe_center_min_x`/`safe_center_max_x` should be compared to `player.center_x`.

Landing estimates are explicitly approximate and descending-only: constant horizontal
velocity, 0.5 px/frame² downward acceleration, and a 5 px/frame fall-speed cap. They do not
simulate the candidate action, ceilings, walls or moving platforms. A separate approximate
landing-threat feature projects nearby enemies to the landing time, including enemies that
are not vertically overlapping yet. If that predicts contact with an enemy ahead and visible ground supports
braking, the prompt asks for a brief airborne brake rather than waiting until landing.
A threat already behind Mario instead calls for forward movement, avoiding braking back
into that enemy. An unknown
projection is not evidence of safety. Enemy timing uses a conservative 16-pixel horizontal
margin, current vertical overlap, and an approximate eight-frame clearance budget; these
are urgency estimates rather than guarantees. Jev retains control of the selected action.

For comparisons, use identical environments, seeds, control versions and frame cycles.
Compare complete `episode_end` records, report stage-clear rate, death positions, stalls
and decisions per completed stage, and keep quit/restart/policy-error runs separate.
Old synchronous and millisecond-budget runs have different control semantics. Fixed-frame
logs identify `control_mode=fixed_frame_lookahead`; complete actions should have exactly
`frames_per_decision` execution frames, with that same application delay after bootstrap.
Unit tests and scripted emulator replays validate the harness; they do not establish an
improved Jev stage-clear rate.

## Development

```sh
ruff format --check src tests
ruff check src tests
python -m pytest -q
```
