"""Approximate visible actor motion, including falls from observed platforms."""

from .observation import Actor, Observation

WALKERS = {"goomba", "green_koopa", "red_koopa"}


def actor_path(
    actor: Actor, velocity: tuple, o: Observation, frames: int, *, gravity: float = 0.25
) -> list[tuple]:
    vx, vy, known = velocity
    x, y = actor.x, actor.y
    path = [(x, y, known)]
    for _ in range(frames):
        nx = x + vx
        if actor.kind in WALKERS or actor.kind in {"stationary_shell", "moving_shell"}:
            if o.known_x is None or not o.known_x[0] <= nx + 8 < o.known_x[1]:
                # A visible actor can lie beyond the local tile observation.
                # Missing support there is unknown terrain, not an observed gap.
                # Extrapolate the observed motion until geometry is available.
                x, y, known = nx, y + vy, False
                path.append((x, y, known))
                continue
            # Actor telemetry is about 8px below the player's support coordinate.
            feet = y + 8
            for b in o.blocks:
                if feet < b.top and feet + 16 > b.bottom:
                    if vx > 0 and x < b.left and nx + 16 > b.left:
                        nx, vx = b.left - 16, -vx
                    elif vx < 0 and x + 16 > b.right and nx < b.right:
                        nx, vx = b.right, -vx
            support = any(b.left <= nx + 8 < b.right and abs(b.top - feet) <= 1 for b in o.blocks)
            if support and vy <= 0:
                vy = 0
            else:
                # No engine phase or spawn state: walking enemies accelerate
                # downward when they leave visible support, instead of hovering.
                # A measured level velocity does not establish the phase or
                # acceleration of this fall. Preserve that uncertainty even
                # after a nominal landing until a new observation confirms it.
                known = False
                vy = max(-5.0, vy - gravity)
            ny = y + vy
            for b in o.blocks:
                if b.left <= nx + 8 < b.right and feet >= b.top and ny + 8 <= b.top:
                    ny, vy = b.top - 8, 0
        else:
            ny = y + vy
        x, y = nx, ny
        path.append((x, y, known))
    return path
