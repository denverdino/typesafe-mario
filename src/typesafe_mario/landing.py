"""Reaction margins at a modelled landing; no claim of collision-free control."""

from math import ceil, copysign

from .actions import JUMP_ACTIONS, Action
from .dynamics import direction


def reaction_tail_frames(cycle, brake):
    """Pad actor paths past the planning endpoint for a full landing reaction.

    The player predictor caps speed at 3.5 px/frame. Jump clearance takes at
    most 16 ascent steps plus input arming/rearming in landing_window.
    """
    return max(2 * cycle, cycle + 18, cycle + ceil(3.5 / brake) + 3)


def first_path_contact(path, frame, *, x, y, vx, height, frames, brake_after=None, brake=0):
    """Compare an actor trajectory with coasting/braking on the landing plane.

    Offsets are relative to this landing. No environment, history mutation or
    assumption of a successful jump is needed to test this reaction window.
    """
    for offset in range(min(frames + 1, len(path) - frame)):
        if offset:
            if brake_after is not None and offset > brake_after:
                vx = copysign(max(0, abs(vx) - brake), vx)
            x += vx
        ex, ey, _ = path[frame + offset]
        if x < ex + 16 and x + 16 > ex and y < ey + 16 and y + height > ey:
            return offset
    return None


def landing_window(m, o, paths, frame, elapsed, cycle, dynamics, removed=()):
    remaining = (-elapsed) % cycle
    # Fresh A first arms takeoff; held A also needs a release. Estimate time
    # to clear an enemy's body using the fitted jump impulse and held gravity.
    clearance, speed, ascent = 0, dynamics.value("jump_speed"), 0
    while clearance < 8 and ascent < 16 and speed > 0:
        clearance += speed
        speed -= dynamics.value("gravity_hold")
        ascent += 1
    jump_frames = 1 + int(m.previous_action in JUMP_ACTIONS) + ascent
    headroom = not any(
        b.left < m.x + 14 and b.right > m.x + 2 and m.y < b.bottom < m.y + o.height + 8
        for b in o.blocks
    )
    if clearance < 8 or not headroom:
        jump_frames = None
    enemies = []
    for actor in o.actors:
        if actor.kind in {"flagpole", "springboard"} or actor.identity in removed:
            continue
        path = paths[actor.identity]
        ex, ey, known = path[frame]
        evx = path[min(frame + 1, len(path) - 1)][0] - ex
        side = 1 if ex >= m.x else -1
        gap = abs(ex - m.x) - 16
        toward = max(0, side * m.vx)
        enemy_toward = max(0, -side * evx)
        brake = dynamics.value("brake_accel")
        brake_time = toward / brake
        # An input already braking needs no new decision to start decelerating.
        locked = 0 if direction(Action(m.previous_action)) == -side else remaining
        braking_distance = toward * locked + toward * toward / (2 * brake)
        brake_margin = gap - braking_distance - enemy_toward * (locked + brake_time + 3)
        lookahead = max(2 * cycle, remaining + (jump_frames or 0), locked + ceil(brake_time) + 3)
        position = {"x": m.x, "y": m.y, "vx": m.vx, "height": o.height, "frames": lookahead}
        contact_time = first_path_contact(path, frame, **position)
        braking_contact = first_path_contact(
            path, frame, **position, brake_after=locked, brake=brake
        )
        complete = frame + lookahead < len(path)
        same_height = abs(ey + 8 - m.y) <= 16
        if not same_height and contact_time is None and braking_contact is None and complete:
            continue
        # Eight pixels of ascent can clear a ground actor, but cannot certify
        # escape from an actor falling from above into the jump itself.
        actor_jump_frames = jump_frames if abs(ey + 8 - m.y) <= 1 else None
        jump_late = actor_jump_frames is None or (
            contact_time is not None and contact_time <= remaining + actor_jump_frames
        )
        # Braking may already be active when landing occurs mid-cycle. In
        # that case a coasting collision alone does not invalidate the escape.
        unsafe = contact_time is not None and jump_late and braking_contact is not None
        enemies.append(
            {
                "actor_id": actor.identity,
                "body_gap_pixels": round(gap, 2),
                "contact_in_frames": round(contact_time, 2) if contact_time is not None else None,
                "braking_contact_in_frames": braking_contact,
                "braking_distance_pixels": round(braking_distance, 2),
                "braking_margin_pixels": round(brake_margin, 2),
                "jump_clearance_frames": actor_jump_frames,
                "contact_before_escape": unsafe,
                "reaction_window_complete": complete,
                "motion_uncertain": not known or not complete,
            }
        )
    return {
        "frame": frame,
        "x": round(m.x, 2),
        "y": round(m.y, 2),
        "vx": round(m.vx, 2),
        "frames_until_next_action": remaining,
        "enemies": enemies,
        "assumption": "Nominal landing and reaction margins, not safety. Actor paths include "
        "observed-geometry turns and falls; Mario coasts or brakes on the landing plane. "
        "Jump clearance, terrain support, unknown motion and contact responses remain approximate.",
    }
