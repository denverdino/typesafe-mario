# TypeSafe Mario

An experimental controller that lets TypeSafe's Jev model directly choose NES
controller inputs for the original Super Mario Bros.

The model does **not** receive screenshots. The harness translates emulator telemetry
and RAM into compact, object-centric JSON containing Mario's motion, jump trajectory,
upcoming enemies, terrain, measured response delay, recent-control results, and episode
progress, visible scoring opportunities and confirmed scoring events. Jev chooses one of the legal controller actions, the emulator advances several
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

The dashboard shows game score, the current coin counter, confirmed collected coins
and confirmed stomps separately from environment reward. Confirmed counts are lower
bounds; unknown values display an em dash. It also shows the selected action, full Choice probability distribution,
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
- `scoring`: score, coin counter, previous-frame score delta, confirmed collection/stomp
  totals, unattributed score gain and constraints on extra effort for scoring
- `opportunities`: up to eight currently observed coins, reward blocks, mushroom/
  fire-flower items or suitable walking Goomba/green Koopa targets, with positions, state, source and
  observation frame; blocks in Mario's current column remain observable

## Best-effort scoring

Jev prioritizes survival and reaching the flag, then actively tries to collect visible
coins, hit reachable reward blocks from below, obtain mushroom/fire-flower upgrades
and stomp suitable enemies along the way.
It may slow down or adjust a jump,
but must not backtrack for missed rewards, wait to farm enemies or jeopardize a gap
crossing. LEFT remains available for evasion and recovery. Hidden rewards, shell
kicking, chain-stomp optimization and side routes are outside this first version.
The observation layer distinguishes visible coin question blocks (`0xc0`) and
multi-coin bricks (`0x58` / `0x5d`) from loose coins. These are `coin_block` targets
with `block_kind`, `required_interaction: hit_from_below` and tile bounds. Mushroom/
flower question blocks (`0xc1`) are separate `powerup_block` targets. Their
`expected_powerup_if_hit_now` is a mushroom while small, a fire flower while big
or fiery, and null for an unknown status. Actual contents depend on status at the
hit. Empty/hidden blocks, stars, 1-ups and other special bricks are excluded.
This describes current RAM
contents, not a guaranteed reachable jump or a remaining coin count.

For a reward block, `head_bump_geometry` supplies the current background-collision
head probe, the block's horizontal interval relative to that probe, the upward
distance to its underside (positive when the head is below it), and whether a
block bounce is active. The interval is half-open: it contains zero when aligned.
Small/crouching and standing big Mario use different head offsets; unsupported
swimming geometry is null. These are observations, not predicted contact times.
Jev accounts for momentum and committed controls, approaches at controlled speed,
and chooses a forward jump or a jump without directional input to hit from below.
After a hit it reassesses the target; additional multi-coin hits must fit the same
safety, time and progress constraints. The executor does not force a jump or
select the target. No new action macro is introduced.

After a power-up block is hit, slot 5 supplies an actual `powerup` touch target,
separate from hostile enemies. `powerup_kind` identifies mushroom/fire_flower;
`emergence_phase` distinguishes emerging from active. During early emergence the
collision box is null because the game has not refreshed it yet. Mushroom motion
includes falling off a block; flowers stay still. Mario's screen collision box is
also provided for Jev to judge contact. Emerging does not mean collected.
Jev plans the hit and pickup together, controlling momentum before the hit and
using safe positioning jumps/brief slowdowns to avoid passing the item. It resumes
forward progress when the item is lost or the existing pursuit budget expires.
Already fiery Mario only collects further flowers incidentally. This adds no
auto-pickup or forced action, and does not permit backtracking for missed items.

When remaining game time is 100 or less (or unknown), or 48 emulator frames have
passed without improving best horizontal progress, `scoring.pursuit.allow_extra_effort`
becomes false. These are policy parameters, not physical safety guarantees. Incidental
scoring while advancing is still allowed; the executor never replaces Jev's choice
based on these fields, jump probability or danger score.

Opportunity relative Y is positive downward; player vertical speed remains positive
upward. Screen collision boxes are distinct from world X positions. A coin's tile
origin is not a collision box. Block `tile_bounds_screen_xyxy` are half-open screen
rectangles, distinct from object collision boxes. A block may have a slightly
negative relative X while Mario overlaps its column. A stompable enemy type does not predict successful
contact or a safe landing. Unavailable observations and truncated candidate lists
do not establish safety. Counters, disappearing objects and score changes are not
interchangeable evidence; see [scoring evidence](tests/fixtures/scoring-observations.md).

Trace schema v2 adds confirmed coin/stomp/injury events, score changes and an
`episode_end.summary`. Decision records add `scoring_result` over
`(start_frame, end_frame]`, without changing the original request snapshot.
The initial NOOP cycle contributes only to the episode total.
`score_before_clear` is the immediately preceding observation's score and
`score_at_clear` is the first clear frame's score. Neither is the score after
the ending animation, because execution stops at clear. Missing score pairs yield
unknown gains; coverage counts are included. Score-source attribution remains
unknown without unique evidence, even when a collection and score change coincide.
Budget termination is not a successful clear.

Validate strategy changes with complete autonomous episodes using the same environment,
seeds and budget for baseline and candidate. Compare completion/death first, then
pre-clear score and confirmed coins/stomps, reporting failures and observation gaps.
Unit tests and a single high-score run do not demonstrate strategy improvement.

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
