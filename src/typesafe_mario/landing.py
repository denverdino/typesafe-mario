"""Reaction margins at a modelled landing; no claim of collision-free control."""

from .actions import JUMP_ACTIONS, Action
from .dynamics import direction


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
        if abs(ey + 8 - m.y) > 16:
            continue
        evx = path[min(frame + 1, len(path) - 1)][0] - ex
        side = 1 if ex >= m.x else -1
        gap = abs(ex - m.x) - 16
        closing = side * (m.vx - evx)
        toward = max(0, side * m.vx)
        enemy_toward = max(0, -side * evx)
        brake = dynamics.value("brake_accel")
        brake_time = toward / brake
        # An input already braking needs no new decision to start decelerating.
        locked = 0 if direction(Action(m.previous_action)) == -side else remaining
        braking_distance = toward * locked + toward * toward / (2 * brake)
        brake_margin = gap - braking_distance - enemy_toward * (locked + brake_time + 3)
        contact_time = max(0, gap) / closing if closing > 0 else None
        jump_late = jump_frames is None or (
            contact_time is not None and contact_time <= remaining + jump_frames
        )
        unsafe = gap <= 0 or (closing > 0 and jump_late and brake_margin <= 0)
        enemies.append(
            {
                "actor_id": actor.identity,
                "body_gap_pixels": round(gap, 2),
                "contact_in_frames": round(contact_time, 2) if contact_time is not None else None,
                "braking_distance_pixels": round(braking_distance, 2),
                "braking_margin_pixels": round(brake_margin, 2),
                "jump_clearance_frames": jump_frames,
                "contact_before_escape": unsafe,
                "motion_uncertain": not known,
            }
        )
    return {
        "frame": frame,
        "x": round(m.x, 2),
        "y": round(m.y, 2),
        "vx": round(m.vx, 2),
        "frames_until_next_action": remaining,
        "enemies": enemies,
        "assumption": "Nominal landing and reaction margins, not safety. Approximate braking "
        "and jump clearance; actor reversals, terrain and contact responses can change them.",
    }
