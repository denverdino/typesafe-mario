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
NES emulator -> visible observation -> prediction evidence
  -> tactical state + candidate arbitration -> captured request -> Jev Choice
  -> fixed-frame controller input -> observed feedback
```

`tactical.build_tactical_state()` owns provider-state projection and the single
candidate-risk assessment, including recovery context. The predictor returns
evidence without choosing eligible actions. `questions.build_questions()` builds
the questions independently of the SDK. Policies expose `prepare()` and `resolve()`;
`choose(snapshot, actions)` remains a convenience wrapper for direct callers.

`DecisionRequest` freezes the exact `state` and `questions` arguments as JSON before
dispatch, preserving candidate order. The SDK and logs receive independent copies
of that captured payload. Preparation and serialization are outside API latency.

Prediction version 49 and provider-state version 49 use `observed_at_frame` and
`apply_at_frame` as absolute frames on the parser's observation clock. The delay is
`action_delay_frames`, and execution duration is `action_cycle_frames`.
Trajectory `frame`, contact-frame and risk `through_frame` values remain offsets
from `observed_at_frame`, declared by `prediction.frame_reference`. In particular,
the old prediction `application_frame` was a **delay**, not an absolute frame.
Request preparation rejects mismatched forecast timing; execution checks the actual
application frame and candidate membership before applying buttons. Checkpoint
resume preserves the observation clock while log `execution.start_frame` restarts
at zero; `execution.apply_at_frame` records the observation-clock value.

State version 49 exposes one `action_assessments` entry per eligible action. Its
`execution` describes the cycle that will actually run, `held_input` describes
repeating the same input beyond that cycle, and `continuation` describes a proposed
future sequence. Conditional stomps/bounces and continuation utility cannot clear
an executable-cycle failure. `decision_context.comparison_status` distinguishes an
invalid application state, all cycles predicting failure, insufficient evidence,
and incomparable risks.
Full `action_forecasts` remain local; references to them below describe diagnostics,
not duplicate model-facing action evaluations.

Prompt version 58 asks only `next_action` by default, using mechanical button
descriptions and a short timing/risk contract. Add `--diagnostic-questions` to also
request independent `jump_intent` and `danger` judgments. Missing diagnostics display
as unavailable, not zero danger. Scene hints are included only when relevant.
`next_action.instructions.game_rules` explicitly explains top-down Goomba stomps
and bounces, Koopa shells, dangerous side contact, pits and head collisions. These
rules describe possible interactions; forecasts must still establish feasibility.

For an ongoing flight, an uncertain braking landing cannot be the sole lower-risk
choice merely because its short repeated-input window is clear. Comparable complete
forward routes with only landing uncertainty remain eligible, with all warnings
retained. `unsupported_descent` describes being over a gap during the executable
cycle separately from a predicted fall; the resolved flight still checks landing.
After a grounded input reversal, `prediction.motion_sensitivity` propagates a
heuristic ±1 px/frame initial-speed variation through the committed input and two
control cycles. Divergent head/wall contacts retain `uncertain_wall_contact` and
broaden position ranges. These bounds are sensitivity checks, not calibrated safety.
Horizontal speed fitting preserves continuous input across takeoff, while rejecting
landing, bounce and collision windows. The walking-flight cap is calibrated from
steady visible airborne motion; accelerating and reversing windows cannot calibrate
that cap. Pressing B in midair still cannot create a running takeoff.
Same-frame upward clearance is resolved before a side-wall stop, so clearing a
pipe or brick top does not erase horizontal momentum using the old frame's height.

### Observation-driven prediction

The predictor receives an explicit whitelist of current structured observations:
Mario position and measured motion, visible actors, visible solid tiles, current
movement mode and actual previous buttons. The camera origin is derived from world
position minus screen position. Geometry covers the visible viewport, including
space behind Mario for retreat; offscreen actors, exact RAM velocity, internal enemy states and plant
countdowns are excluded from both prediction and the TypeSafe request. This remains
telemetry-assisted perception, not a screenshot-only agent.

Enemy X combines the RAM page and low byte (`$6E + slot`, `$87 + slot`);
screen X subtracts the camera origin. Visible enemies must also have Y page 1
(`$B6 + slot`) before interpreting the Y low byte (`$CF + slot`). Fish and
Paratroopa variants have distinct type names. Visible actor identities restart
after disappearance, missing telemetry, position discontinuities or parser reset,
so a reused slot cannot inherit history across an observed gap. Slot reuse without
an observable discontinuity remains indistinguishable. Internal state (`$1E + slot`),
raw direction (`$46 + slot`) and activation flags remain diagnostic data only;
direction and behavior meanings depend on the enemy type.

The tactical state adds `actor_assessments`, joining visible actors to same-frame
velocity estimates by identity. It reports actor-minus-Mario position (x rightward,
y upward), observed movement directions and horizontal separation trends accounting
for Mario's motion. Missing history or unmatched evidence gives `unknown`, rather
than stationary. Zero measured displacement does not establish that an actor will
remain still. These summaries describe the observation frame, not the delayed action
boundary; approaching horizontally does not establish a collision, safe stomp,
facing direction or attack intent.

Prediction never receives the environment, reads a save state, calls `env.step`, or
tries alternate timelines. `Predictor.observe()` learns from actual consecutive
observations only. Approximate acceleration, braking, friction, gravity, jump and
water-stroke parameters start with conservative priors and adapt from real motion.
Short past-position windows reduce integer-pixel quantization. Collisions, contact,
coordinate discontinuities, mode changes and episode restarts do not become ordinary
acceleration samples. Fast and slow takeoffs have separate jump-impulse and held-ascent
gravity estimates. Air acceleration is fitted below the walking speed limit because
pressing B during an existing flight does not establish a higher limit. Actor
velocities come from contiguous visible history only.
Nine uninterrupted, contact-free position observations can also estimate current
horizontal speed, avoiding the lag of a past displacement average during acceleration
or braking. Stationary recent windows and apparent NOOP reversals retain the fallback
estimate rather than inventing motion after a stop.
The takeoff speed class likewise uses a continuous grounded position window when
available, and jump-impulse learning shares that class. A single rounded pixel delta
must not turn a walking flight into a running flight.

The predictor first estimates the result of the already committed buttons, then
compares seven candidate inputs. Each candidate conditionally repeats at cycle
boundaries over four cycles (at least 48 frames, capped at 96). Cycles or application
delays above 96 frames return unavailable estimates; actual control timing is not
shortened. Grounded jumps and water strokes use the same release/repress convention
as execution. Visible blocks supply approximate wall, ceiling and landing contacts.
Visible walking enemies use approximate gravity, support and wall response; other
actors use observed motion with growing uncertainty. Enemy responses, stomps,
springs, pipe transitions and unseen spawns remain uncertain. A lingering Goomba body
is excluded only after an actual observed bounce, stationary body and visible score
increase, with no overlapping alternative actor to explain the bounce. This evidence
expires on disappearance, movement or observation gaps; Koopas remain hazards.

Descending contact near a brick underside retains horizontal momentum with an
`uncertain_wall_contact` warning instead of relying on a full sprite-box collision
to stop Mario. Vertically connected tile rows are treated as one face for this
check. Full walls still block movement. Rising ceiling contact uses the head center
instead of the wider body/foot span: grazing a brick edge must not invent a head
strike and turn a continuing jump into a predicted fall. An existing head overlap
is still resolved before horizontal motion can escape the tile.
The already committed input also checks
conditional enemy contacts: a possible Goomba bounce cannot be treated as a known
grounded application state. `committed_interactions` remain hypothetical; a stopped
projection marks `committed_future.valid_for_application=false` and all candidate
paths retain the corresponding uncertainty.

Each candidate also has a `continuation`: a bounded search over subsequent input
cycles in this same approximate model (at most six cycles, three retained branches
per first action, plus waiting/braking refuges). A nearby high wall enables up to ten
cycles and an 80-frame horizon with generic retreat/run/jump sequences, including
tapered stair faces. It can compare walking then jumping
with jumping immediately. Only the first selected input is executed; later inputs
are suggestions that require fresh observations. Progress, visible landing support
and close contacts contribute to the reported utility. Search is incomplete:
`predicted_failure` means the retained estimates failed, not that every possible
continuation must fail. A possible single-Goomba stomp can continue through a
conditional bounce so later enemies and terrain are still checked. That branch
alone temporarily removes the Goomba and retains `possible_stomp` and
`uncertain_bounce` warnings; it cannot qualify as a warning-free refuge. Bounce
impulse uses a bounded prior updated only after an actual scored Goomba stomp,
and its approximate ascent uses falling gravity. This does not establish survival.
An airborne endpoint now retains `unresolved_flight` even with ground below. A
bounded NOOP tail can spend up to two more control cycles finishing the flight,
within the 96-frame prediction limit; it cannot start another jump. `first_landing`
and `continuation.landings` report enemy body gaps, approximate braking margins,
jump clearance time and remaining frames before the next action can apply. A
`landing_contact_window` warning prevents a tight landing from becoming a clear
refuge simply because the plan suggests jumping again later.
`committed_future.landing_timing` reports a heuristic timing margin when the
scheduled input predicts a landing. If that margin overlaps action application,
jump branches retain `uncertain_takeoff`: pressing A before actual touchdown and
holding it through touchdown may never start a jump. Nominal grounded coordinates
alone cannot certify that takeoff. Observed grounded states and sufficiently
settled predicted landings do not receive this warning. Walking enemies that have
visibly reversed use their latest nonzero direction and average absolute movement,
so opposing steps no longer cancel their estimated speed toward zero.

Only an actually selected action can retain its proposed follow-up sequence.
`plan_followup` rechecks that remainder against fresh terrain, actors and dynamics;
discontinuous observations or differing actual inputs invalidate it. It is an
advisory alternative in the bounded search, not an execution commitment or a
learning sample. TypeSafe receives its current warnings and still chooses the
executed action without replacement.

Koopa stomps and spring contacts still stop the continuation estimate;
simultaneous contacts cannot erase another hazard. An actual observed Koopa bounce
followed by a stationary body can establish an inferred `stationary_shell`. Walking
into that shell from supported ground is a possible kick, not a proven fatal contact
or a safe route. Actual subsequent fast motion changes its role to `moving_shell`
and uses the newly observed velocity; both roles retain the actor as a hazard.
Slow movement, missing observations or ambiguous identity invalidate the kickable
shell inference. Merely seeing a motionless Koopa does not establish a shell.

TypeSafe receives these inferred roles and conditional left/right `kick_forecasts`.
The advisory paths use a stated speed prior, visible wall/pipe rebounds, and possible
return times measured from a hypothetical kick. They do not establish when a kick
will happen, whether it kills another enemy, or whether Mario survives. The policy
checks return paths and jump headroom, then reobserves after contact. Continuations
stop at a possible kick; its uncertainty remains in the executable-cycle risk filter
even when later nominal routes look clear. No emulator trials or speculative
updates to observed state are used.

When a visible wall connects a platform to an overhead ceiling, a clear visible
drop and lower corridor can propose a local descent goal. Continuations then value
reaching that lower floor, including temporarily moving left, instead of repeatedly
trying to jump through the sealed wall. The goal is recomputed from current geometry;
it does not establish safety or reachability. Enemy, fall and visibility checks still
apply, and TypeSafe receives the goal alongside the candidate forecasts.

Observed approaching walkers on continuous visible ground can also enable the
80-frame run-up search when the approach and group have sufficient overhead room.
`prediction.enemy_crossing` reports the group and available open-ground interval,
including nearby low clearance behind Mario. TypeSafe is asked to prepare a
directional crossing while that space remains, rather than repeatedly postponing
the jump or retreating underneath bricks. The hint is advisory: every actor,
ceiling contact and landing is still checked, and existing risk filters remain.

`prediction.committed_future` and `action_forecasts` contain nominal positions,
heuristic position ranges, possible enemy contacts, unknowns and qualitative
confidence. `no_contact_predicted` is not proof of safety; there are no native-verified
routes, exact future death frames or guaranteed landings. Before the API request,
an observation-based risk filter checks two control cycles after application and
follows an already committed flight through its first estimated landing. When
lower-risk alternatives exist, candidates with contact, unresolved fall, unreliable
motion or unknown-terrain warnings are excluded. If all ranges suggest contact,
nominal body collisions are distinguished from uncertainty-envelope overlap alone.
Unknown terrain or a possible fall cannot qualify as a lower-risk refuge. If no
lower-risk alternative can be identified, all allowed actions remain available.
When every input predicts nominal enemy contact, the filter can compare contact
duration, summed body-overlap area and final overlap within the first executable
cycle. It excludes a dominated option only when another has no worse values in
all three metrics and improves at least one. Fall, edge, visibility and reliability
warnings in either risk window prevent an option from dominating others. This is
an attempt to reduce exposure, not evidence that Mario survives a collision.
Bounded continuations can identify an alternative when repeated-input forecasts
cannot. A complete warning-free continuation can discharge a later repeated-input
fall, marginal-landing or landing-reaction-margin warning only when the first
executed cycle has no risk flags. It cannot discharge an enemy body-contact
warning or uncertain takeoff. Complete warning-free
continuations take priority over higher-progress plans with remaining warnings.
The continuation fallback also retains executable-cycle fall, unseen-terrain and
unreliable-motion warnings; a later nominal landing cannot erase those warnings.
For a flight already in progress at action application, its first continuation
landing must also retain support across the forecast's horizontal position range.
A narrow nominal landing beside a pit keeps `uncertain_landing`; later movement
on nominal ground cannot erase that first-landing risk. Visible lower support can
catch a platform-edge miss. Later optional jumps still require fresh observations
instead of inheriting an ever-growing position range through the entire search.
The controller uses Jev's selected action unchanged; there is no post-response
action replacement. Pipe geometry remains advisory. Observed stationary wall
pushes can exclude inputs that predict continued immobility when a complete clear
moving alternative exists. This requires grounded actual and queued states,
repeated ineffective rightward input, and a moving alternative with no first-cycle
risk flags; intentional waiting alone does not trigger it. Distant visible actors
do not disable recovery when the alternative's forecast remains clear.
Prolonged stationary NOOP input has a separate recovery check: after at least48
stationary frames including40 NOOP frames, it can exclude another NOOP when an
already eligible moving input has a clear first cycle and complete warning-free
continuation. This prevents continually postponing the movement that makes a waiting
plan look attractive. Brief waits and waits without such an alternative remain valid.
Recovery also measures a contiguous local motion envelope. After at least 64 frames
within eight horizontal and four vertical pixels near a wall, repeated forward
input can establish a local loop even when left/right shuffling resets stillness.
If an eligible action moves forward or up immediately and has a complete warning-free
continuation ending on support beyond the stalled position, choices that keep
postponing that escape are excluded. A real run-up, jump, observation gap or warp
breaks the local evidence; uncertain jumps do not qualify as escapes.
`recovery.horizontal_motion` separately records a contiguous eight-pixel horizontal
envelope without resetting on vertical motion. After 64 frames near a wall, a valid
rising application state with A held can reveal a repeated interrupted jump. A-release
choices that predict returning to the same horizontal envelope on lower support can
be excluded only when an already eligible A-held action gains height this cycle and
has a complete warning-free continuation landing higher and ending on support beyond
the loop. Descending, swimming and uncertain routes do not trigger this check.
`risk_control.recovery_filter` logs the evidence, retained hold alternatives and
excluded inputs as `interrupted_ascent_with_clear_landing`; Jev still chooses the input.

Provider-state `risk_control` and log `selection` expose candidate eligibility and
exclusion reasons. Remaining candidates are not certified safe. Partial body overlap
with a visible ceiling or wall is resolved conservatively instead of predicting
motion through the solid tile. Unsupported descent warns before falling below the
screen, but a later predicted landing on visible support can clear that warning;
ordinary short gaps must not make every forward action ineligible. Marginal edge
landings remain uncertain even if they occur after the ordinary control window.
Grounded estimates also check whether the horizontal position range retains visible
support. A nominal position just before a gap is insufficient when that range crosses
the edge. Short drops can remain eligible when both range endpoints are estimated
to reach the same visible lower floor with a margin.
After landing, later optional departures are replannable. Airborne reverse input
uses air acceleration rather than ground braking, and reliable past pixel motion
is averaged to reduce integer-velocity bias.
Gravity fitting retains both signs of one-pixel velocity fluctuation; discarding
only upward fluctuations would systematically underestimate jump height. Parameters
are fitted once per forecast and reused across hypothetical steps.
Air acceleration uses a quadratic fit to nine consecutive observed positions,
excluding input changes, contacts and motion near the speed cap. This avoids
mixing raw and smoothed velocity differences or selecting only quantized low-speed
samples. At a held-jump apex, a zero pixel height change can still be ascent:
continuous rising-height evidence can retain a fitted subpixel upward velocity
instead of prematurely applying falling gravity.
Held-ascent gravity also uses nine-position fits, restricted to a known flight
class and velocities above the apex quantization band. If contacts or short input
segments leave no suitable window, the original bounded prior remains in use.
During continuous held ascent, five observed positions estimate the latest velocity
using the fitted deceleration, avoiding a whole-pixel speed error that can falsely
reject a gap jump. Ground friction uses nine observed positions during uninterrupted
NOOP motion; it excludes stopped intervals instead of selecting nonzero pixel speeds.
Observed downward velocity changes of two pixels are also retained as quantized
measurements; discarding them biases release gravity downward and predicts landing
too late. Fitted parameters still obey their physical bounds, and observations near
contacts or discontinuities remain excluded from learning.

Repeated actual LEFT input without horizontal movement at the visible left edge
can establish a local movement limit. Forecasts then check approaching enemies
without inventing additional retreat beyond that limit. View changes, observation
gaps, unreliable data and contradictory movement invalidate the inference; it supplies
no offscreen terrain. A fresh grounded jump press has a one-step initiation delay,
based on observed input/height timing; stepping off an edge during that step cannot
invent a takeoff. A fresh press on a landing frame is likewise retained for the next
step. Cancellation on immediate release remains a model assumption.

Only selected, actually executed first cycles are compared with their saved forecasts.
`prediction.validation` reports rolling x/y absolute errors and observed range coverage;
changed inputs and discontinuities invalidate comparisons. Alternative actions are not
executed or used as training labels. Predictor history starts fresh on restart/resume.

The dashboard `Latency` and log `latency_ms` measure API-call duration only.
`prediction.compute_ms` separately measures local forecast computation. Model-facing
state is prepared by `tactical.build_tactical_state()` from the
`observation.model_state` whitelist; fuller parser/RAM-derived diagnostics remain in
local debug logs and are not sent to the provider. There is no native predictor fallback.
Provider requests include compact first-cycle estimates, risk warnings and
continuations, plus recent actual motion segments, per-actor velocity/trajectory
estimates, earliest contact offsets and observed prediction errors. Full
repeated-input trajectories remain in the local run log.
Manual user-requested checkpoint resume is a separate feature, described below.

The action set is `noop`, `right`, `right_jump`, `right_run`, `right_run_jump`,
`jump` and `left`. Water uses repeated A strokes for lift and release for descent;
movement modes are calibrated separately. A visible pipe mouth is a navigation hint,
not proof of successful entry. Stall recovery resets its progress anchor after
coordinate wraps/warps, while gradual retreat still counts as lack of forward progress.

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

Inspect a sample parser/debug JSON and text view without launching the game or
calling the API (actual provider payloads are captured in run logs):

```sh
typesafe-mario state-demo
```

Run World 1-1 with Jev making a decision every eight emulator steps. The default
display is a single recordable window with the live game and model telemetry:

```sh
typesafe-mario play --env SuperMarioBros-1-1-v0 --frames-per-decision 8
```

Use `--frames-per-decision 4` for a shorter prediction horizon, or `16` for a longer one.

Measure opening physics without Jev, then reuse the measured parameters:

```sh
typesafe-mario calibrate --env SuperMarioBros-1-2-v0 --output artifacts/physics-1-2.json
typesafe-mario play --env SuperMarioBros-1-2-v0 --frames-per-decision 8 \
  --physics-profile artifacts/physics-1-2.json
```

Calibration executes predefined walk, coast, brake and jump probes in fresh real
episodes. It stops a probe when visible space is insufficient or an actor approaches.
It never saves/restores emulator state or tests alternative routes. The adjacent
`.observations.json` contains the actual measurements. Fitting is a pure function;
six additional probes compare default and calibrated eight-frame prediction errors,
including input onsets, running takeoffs and air steering. Entry-animation frames
are excluded from these validation scores.
The profile lists parameters without enough evidence and retains their defaults.
Loaded parameters stay fixed; gameplay continues learning the remaining parameters.
Rerun calibration explicitly to replace a profile. Environment mismatches and invalid
parameter values are rejected, and each play log records the profile hash and values.

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

Log schema version 2 uses `record_type=policy_request`, `record_type=request_outcome`,
`record_type=decision` and `record_type=episode_end`.
Provider requests are flushed before worker dispatch with `status=prepared`, an
episode-scoped `request_id`, timing, candidate actions, payload, versions and SHA-256.
This records the SDK-bound arguments, not HTTP headers or credentials; a prepared
record does not prove the server received it. SDK model defaults and transport
metadata are outside this capture. Requests remain in the log if resolution fails,
the episode ends, or Restart discards the response. A matching decision record links
the request that was actually applied and stores its exact captured state. A prepared
record alone does not establish execution, particularly after an abrupt process exit.
Offline policies produce no provider request records. Existing schema-2 checkpoint logs remain replayable; replay ignores
the additional request records.

`request_outcome` records `received` when the runner consumes a response, `failed`
with the exception type, `applied` after the first executed frame, or `discarded`
with its reason and observed completion state. Discarding a still-running call does
not prove cancellation or HTTP completion; its later result is not used. Records
include UTC timestamps and episode-relative monotonic elapsed time. Provider model
and request identifiers are captured only when the SDK response actually supplies
them; otherwise the model is explicitly unresolved.

Each decision includes `prediction_comparison` for the application position and
the completed executable cycle, including x/y error and range coverage. A partial
cycle is not compared against a full-cycle endpoint; an invalid committed state
is not a valid application estimate. Landing forecasts and actual landing frames
are recorded separately with their time references. Landing/recovery events retain
a bounded 64-frame observation/button trace through the end of their cycle;
`episode_end.recent_frames` retains the final 64 frames. These are actual observations,
not counterfactual labels for unselected actions.

`debug_state.prediction_details` stores the full forecast once; its duplicate in
`debug_state.model_state` is omitted. Exact request and decision state are retained
for schema-2 compatibility. Older snapshot hazard/trajectory estimates are explicitly
labelled as diagnostics, not model input. `run_config.implementation` fingerprints
package Python sources (including uncommitted edits) and records dependency versions.
`predictor_initialization=cold_start` also applies to checkpoint retries: restoring
the emulator/parser does not restore learned prediction history.
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

### Offline run comparison

```bash
typesafe-mario analyze-log artifacts/run-20261003T115315.828217Z.jsonl
typesafe-mario analyze-log artifacts/before.jsonl artifacts/after.jsonl --json
```

No emulator or Jev request is needed. The command reports each episode's result,
furthest/final x, risk-filter counts, stalled decisions, unapplied requests and
prediction comparisons. It can derive position comparisons from older logs too;
an unfinished log is reported as incomplete, never as a death or a clear. Multiple
runs remain separate, with their configuration attached, so different code versions,
seeds or cold-start conditions are not silently pooled into one success rate.
For interrupted logs, `frames` counts finalized decision records only; explicit
application events without a finalized cycle appear in `unfinalized_applied_request_ids`.

`tests/fixtures/world1-2-v46-failures.json` preserves the two recorded failures as
compact actual-button sequences plus their final decision evidence. Regression tests
reproduce all 250 endpoint positions and guard against treating a conditional stomp
as clearance of first-cycle collision. They reproduce failures; they do not claim
an improved clear rate. Evaluate prompting changes with repeated closed-loop runs
at fixed environment, seed, cadence and implementation fingerprint.

## Model input and judgments

`observation.model_state(snapshot)` supplies the observation whitelist.
`tactical.build_tactical_state(snapshot, actions)` adds the authoritative risk report
and projects compact forecasts; `TypeSafePolicy.prepare()` captures that final state
and its questions:

- `player`: observed coordinates, pixel-difference velocity, current grounded/water
  state and motion reliability. World x increases rightward; y increases upward.
- `visible_actors`: visible identities, kinds and positions, without engine timers.
- `terrain`: visible block rectangles, known horizontal extent and visible pipe mouth.
- `reaction_timing`: committed input, delay and next action duration.
- `recovery`: observed stall duration and recent actual input counts.
- `prediction`: committed motion, actor forecasts, route cues and validation errors.
- `action_assessments`: one interpretation of executable risk and conditional routes.
- `decision_context`: allowed actions, comparison status and evidence priority.

The provider's `next_action` chooses the actual controller macro from the eligible
criteria. Optional `jump_intent` and `danger` are diagnostics and never override
buttons. A single remaining candidate does not establish safety or well-calibrated
model confidence. Unknown geometry does not imply an empty or safe path.

The original detailed parser remains useful for the local dashboard and diagnostics.
Its internal terrain heuristics and RAM-derived fields are not the model input.
Native exact-outcome tests have been replaced with tests for observation isolation,
causal learning, approximate motion, uncertainty and unmodified provider selection.
Native execution fixtures still validate real button timing and manual checkpoint replay.

## Development

For landing-reaction and planning-horizon changes, start with the pure observation
regressions. These run without a game environment or provider calls:

```sh
python -m pytest -q tests/test_landing_paths.py tests/test_planning_horizon.py
```

Actor contact scanning and reaction-window padding live in `landing.py`;
`continuation.py` owns the bounded physical planning horizon. Controller cadence
does not shorten that horizon. New movement regressions should use visible
geometry and actual observation history, including translated-coordinate cases.
Candidate paths must be computed locally, never by branching emulator snapshots
or trying alternative inputs against emulator memory. Full game runs remain a
separate end-to-end acceptance check.

```sh
ruff format --check src tests
ruff check src tests
python -m pytest -q
```
