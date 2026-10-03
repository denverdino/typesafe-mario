"""Shell evidence from actual visible contacts, never enemy engine state."""

from dataclasses import dataclass, replace

from .actor_motion import actor_path
from .observation import Actor

KOOPAS = {"green_koopa", "red_koopa"}
SHELLS = {"stationary_shell", "moving_shell"}


@dataclass
class ShellEvidence:
    actor: Actor
    since: int
    state: str = "pending_stomp"
    vx: float = 0
    vy: float = 0
    velocity_known: bool = True


class ObservedShells:
    def __init__(self):
        self.tracks = {}

    def observe(self, a, b):
        if (
            b.frame != a.frame + 1
            or a.level != b.level
            or not (a.motion_reliable and b.motion_reliable)
            or a.terminal
            or b.terminal
            or a.swimming
            or b.swimming
        ):
            self.tracks.clear()
            return
        actors = {actor.identity: actor for actor in b.actors}
        for identity, track in list(self.tracks.items()):
            actor = actors.get(identity)
            if actor is None or actor.kind != track.actor.kind:
                del self.tracks[identity]
                continue
            dx, dy = actor.x - track.actor.x, actor.y - track.actor.y
            if abs(dx) > 12 or abs(dy) > 16:
                del self.tracks[identity]
                continue
            if track.state == "pending_stomp":
                if dx != 0 or dy != 0:
                    del self.tracks[identity]
                    continue
                if b.frame - track.since >= 2:
                    track.state = "stationary_shell"
            elif track.state == "stationary_shell":
                if abs(dx) >= 2:
                    track.state = "moving_shell"
                    track.vx, track.vy = dx, dy
                elif dx != 0 or dy != 0:
                    # Slow walking can indicate revival; do not keep calling
                    # it a safely kickable shell without a new observed stomp.
                    del self.tracks[identity]
                    continue
            else:
                track.velocity_known = abs(dx) >= 2
                track.vx, track.vy = dx, dy
            track.actor = actor
        if a.vy < 0 < b.vy and not a.grounded and not b.grounded:
            contacts = [
                actor
                for actor in a.actors
                if a.x + 14 > actor.x + 2
                and a.x + 2 < actor.x + 14
                and actor.y + 7 <= a.y <= actor.y + 24
            ]
            if (
                len(contacts) == 1
                and contacts[0].kind in KOOPAS
                and not any(
                    actor.kind == "springboard" and abs(actor.x - a.x) < 24 for actor in a.actors
                )
            ):
                actor = actors.get(contacts[0].identity)
                if actor is not None and actor.kind == contacts[0].kind:
                    self.tracks[actor.identity] = ShellEvidence(actor, b.frame)

    def confirmed(self, observation):
        return {
            a.identity: self.tracks[a.identity]
            for a in observation.actors
            if a.identity in self.tracks
            and self.tracks[a.identity].actor == a
            and self.tracks[a.identity].state in SHELLS
        }


def kick_forecasts(actor, observation, horizon):
    """Advisory trajectories starting at a hypothetical kick, not game trials."""
    result = {}
    for name, direction in (("left", -1), ("right", 1)):
        path = actor_path(
            replace(actor, kind="moving_shell"), (direction * 4.0, 0, True), observation, horizon
        )
        rebound, return_frame = None, None
        for frame in range(1, len(path)):
            dx = path[frame][0] - path[frame - 1][0]
            if rebound is None and dx * direction < 0:
                rebound = frame
            if rebound is not None and abs(path[frame][0] - actor.x) <= 16:
                return_frame = frame
                break
        result[name] = {
            "assumption": "Conditional kick from the current shell position; speed prior "
            "4 px/frame, not a measured launch speed. Timing starts at the hypothetical kick. "
            "No enemy removal or safe contact is established. Reobserve after contact.",
            "first_rebound_frame": rebound,
            "return_near_kick_frame": return_frame,
            "trajectory": [
                {
                    "frame_after_kick": f,
                    "x": round(path[f][0], 2),
                    "y": round(path[f][1], 2),
                    "motion_uncertain": not path[f][2],
                }
                for f in sorted({0, min(8, horizon), min(16, horizon), min(24, horizon)})
            ],
        }
    return result


def possible_ground_kick(motion, actor, x, y):
    return (
        actor.kind == "stationary_shell"
        and motion.grounded
        and abs(motion.y - (y + 8)) <= 4
        and (x - motion.x) * motion.vx > 0
    )
