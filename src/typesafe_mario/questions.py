"""Short execution contract; tactical assessments own the evidence interpretation."""

from collections.abc import Sequence

from .actions import ACTION_DESCRIPTIONS, Action

PROMPT_VERSION = 59


def build_questions(
    state: dict, candidates: Sequence[Action], *, diagnostics: bool = False
) -> dict:
    instructions = {
        "task": "Choose exactly one allowed controller action to complete the stage alive. "
        "Only next_action controls buttons; the selected eligible action is executed unchanged.",
        "game_rules": "Descending onto a Goomba from above can defeat it and bounce Mario. "
        "Stomping a walking green/red Koopa turns it into a shell; it does not remove the hazard. "
        "A stationary shell can be kicked and a moving shell can return after hitting a wall. "
        "Side or underside enemy contact is dangerous. Do not assume every enemy is stompable: "
        "plants and spiked enemies are not ordinary stomp targets; unknown kinds remain unknown. "
        "Stomping is a valid tactic when the descent, top contact and post-bounce path are feasible. "
        "A predicted stomp is conditional until confirmed by fresh observation. Falling into a "
        "pit kills Mario. Hitting a block from below can interrupt ascent and shorten a jump.",
        "execution": "reaction_timing identifies the observation frame, absolute application "
        "frame and cycle length. Input before application is already committed. Choose for the "
        "estimated application state, not the current position. Actions specify held buttons, "
        "not guaranteed movement: LEFT takes time to brake; NOOP can coast; A cannot start a "
        "grounded jump while airborne. At a grounded/swimming cycle boundary, an already-held "
        "A is released for the first frame before repressing. There is no automatic mid-cycle "
        "release on landing. Releasing A during ascent can shorten the jump.",
        "comparison": "action_assessments contains only eligible actions. Compare execution "
        "contacts by actor identity and earliest frame offset; inspect both the first enemy "
        "and where the action leaves Mario relative to the next enemy. Compare execution "
        "risk first: avoid predicted body contact or falling when an alternative has comparable "
        "evidence without those failures. Unknown is not safe. held_input describes repeating "
        "the input beyond this cycle; continuation is a conditional future sequence, not a "
        "commitment. Its score or hypothetical stomp/bounce cannot erase executable-cycle "
        "failure. Compare continuation score only within comparable execution risk. If all "
        "actions are risky, compare contact exposure and available terrain; do not call any "
        "safe. If the committed application state is invalid, downstream positions and plans "
        "are unreliable. Choose a mitigation without assuming the predicted interaction succeeded.",
        "progress": "Within comparable risk, favor useful progress toward navigation_goal or "
        "the stage exit. A revalidated plan_followup is advisory. During recovery, avoid "
        "repeating an ineffective input or endlessly postponing the same planned jump.",
        "coordinates": "Observation positions use world x rightward and y upward. Mario spans "
        "x..x+16 and y..y+height_pixels. Actor relative_position is actor minus Mario. "
        "Forecast frame values are offsets from observed_at_frame. Actor motion is an estimate "
        "from visible history, not facing or attack intent. Unknown motion is not stationary. "
        "No-warning estimates, model confidence and heuristic ranges are not safety guarantees.",
    }
    prediction = state.get("prediction", {})
    hints = []
    if state["player"]["swimming"]:
        hints.append(
            "In water A strokes upward; releasing A lets Mario sink. Landing is unnecessary."
        )
    if state.get("terrain", {}).get("side_exit_pipe"):
        hints.append(
            "At the visible side exit, align with entry_standing_y and walk right into the mouth."
        )
    if prediction.get("navigation_goal"):
        hints.append(
            "navigation_goal is a local geometric target; its utility can favor retreat/descent."
        )
    if prediction.get("enemy_crossing"):
        hints.append(
            "enemy_crossing identifies space to prepare a jump across a group; check headroom and landing beyond the group."
        )
    kinds = {a["kind"] for a in state.get("visible_actors", ())}
    if kinds & {"green_koopa", "red_koopa"}:
        hints.append(
            "Only observed_interactions can establish a shell: a motionless Koopa alone cannot. A hypothetical stomp never proves a kickable shell; moving shells can return from walls."
        )
    if "piranha_plant" in kinds:
        hints.append(
            "Plant timing is unknown. A previously occupied pipe requires fresh observation."
        )
    if "springboard" in kinds:
        hints.append(
            "Springboard bounce is not modelled; a route depending on it remains uncertain."
        )
    if hints:
        instructions["scene_hints"] = hints
    questions = {
        "next_action": {
            "type": "choice",
            "instructions": instructions,
            "criteria": {a.value: ACTION_DESCRIPTIONS[a] for a in candidates},
        }
    }
    if diagnostics:
        questions.update(
            {
                "jump_intent": {
                    "type": "choice",
                    "instructions": "Optional desired-input diagnosis at application time, not an "
                    "execution command or proof of takeoff. Airborne held A cannot start a new ground jump.",
                    "criteria": {
                        "start": "Request a new grounded jump or swim stroke if possible.",
                        "hold": "Keep A held for ascent or swimming.",
                        "release": "Release A.",
                        "none": "No jump input intended.",
                    },
                },
                "danger": {
                    "type": "score",
                    "instructions": "Qualitative danger during the executable cycle, including "
                    "unknowns. Diagnostic only; not a collision or survival probability.",
                    "criteria": [
                        "No immediate threat observed",
                        "Possible threat",
                        "Imminent threat",
                    ],
                },
            }
        )
    return questions
