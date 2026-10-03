from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .actions import ACTION_DESCRIPTIONS, Action
from .policy import Decision
from .state import MarioSnapshot

Color = tuple[int, int, int]


class DashboardCommand(StrEnum):
    CONTINUE = "continue"
    RESTART = "restart"
    QUIT = "quit"


@dataclass(frozen=True)
class DashboardTheme:
    canvas: Color = (15, 17, 21)
    surface: Color = (22, 25, 31)
    raised: Color = (29, 33, 40)
    line: Color = (48, 54, 64)
    text: Color = (236, 239, 243)
    muted: Color = (141, 150, 163)
    accent: Color = (137, 166, 193)
    accent_soft: Color = (72, 89, 105)
    warning: Color = (210, 167, 100)
    danger: Color = (199, 112, 101)


@dataclass(frozen=True)
class DashboardConfig:
    width: int = 1440
    height: int = 810
    panel_width: int = 472
    fps: int = 60


def clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


class LiveDashboard:
    """One-window game feed and TypeSafe telemetry display."""

    def __init__(
        self,
        config: DashboardConfig | None = None,
        theme: DashboardTheme | None = None,
    ) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "The dashboard needs pygame. Install the mario extras first."
            ) from exc

        self.pg = pygame
        self.config = config or DashboardConfig()
        self.theme = theme or DashboardTheme()
        pygame.init()
        pygame.display.set_caption("TypeSafe plays Super Mario Bros.")
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("Segoe UI Semibold", 25)
        self.font_action = pygame.font.SysFont("Segoe UI Semibold", 31)
        self.font_body = pygame.font.SysFont("Segoe UI", 17)
        self.font_small = pygame.font.SysFont("Segoe UI", 14)
        self.font_label = pygame.font.SysFont("Segoe UI Semibold", 14)
        self.font_mono = pygame.font.SysFont("Cascadia Mono", 14)

    def close(self) -> None:
        self.pg.quit()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pg.image.save(self.screen, str(path))

    def _text(
        self,
        value: str,
        font: Any,
        color: Color,
        x: int,
        y: int,
    ) -> None:
        self.screen.blit(font.render(value, True, color), (x, y))

    def _rule(self, x: int, y: int, width: int) -> None:
        self.pg.draw.line(self.screen, self.theme.line, (x, y), (x + width, y), 1)

    def _bar(
        self,
        *,
        label: str,
        value: float,
        x: int,
        y: int,
        width: int,
        selected: bool = False,
        color: Color | None = None,
    ) -> None:
        value = clamp01(value)
        bar_color = color or (self.theme.accent if selected else self.theme.accent_soft)
        self._text(label, self.font_small, self.theme.text if selected else self.theme.muted, x, y)
        value_text = f"{value * 100:4.1f}%"
        text_width = self.font_small.size(value_text)[0]
        self._text(value_text, self.font_small, self.theme.muted, x + width - text_width, y)
        bar_y = y + 21
        self.pg.draw.rect(self.screen, self.theme.raised, (x, bar_y, width, 7), border_radius=3)
        fill = max(2, round(width * value)) if value > 0 else 0
        if fill:
            self.pg.draw.rect(self.screen, bar_color, (x, bar_y, fill, 7), border_radius=3)

    def _draw_game(self, frame: Any, x: int, y: int, width: int, height: int) -> None:
        self.pg.draw.rect(self.screen, self.theme.surface, (x, y, width, height), border_radius=4)
        if frame is None:
            self._text(
                "Waiting for emulator frame", self.font_body, self.theme.muted, x + 28, y + 28
            )
            return
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("The dashboard needs numpy.") from exc

        pixels = np.asarray(frame)
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            raise ValueError(f"Expected an RGB frame, received shape {pixels.shape!r}")
        pixels = pixels[:, :, :3]
        source = self.pg.surfarray.make_surface(pixels.swapaxes(0, 1))
        source_ratio = source.get_width() / source.get_height()
        target_ratio = width / height
        if source_ratio > target_ratio:
            target_width = width
            target_height = round(width / source_ratio)
        else:
            target_height = height
            target_width = round(height * source_ratio)
        scaled = self.pg.transform.scale(source, (target_width, target_height))
        target_x = x + (width - target_width) // 2
        target_y = y + (height - target_height) // 2
        self.screen.blit(scaled, (target_x, target_y))
        self.pg.draw.rect(self.screen, self.theme.line, (x, y, width, height), 1, border_radius=4)

    def _restart_rect(self) -> Any:
        return self.pg.Rect(self.config.width - 144, 13, 120, 36)

    def draw(
        self,
        frame: Any,
        snapshot: MarioSnapshot,
        decision: Decision | None,
        *,
        decision_index: int,
        episode_reward: float,
        waiting: bool = False,
        run_ended: bool = False,
    ) -> DashboardCommand:
        restart_rect = self._restart_rect()
        command = DashboardCommand.CONTINUE
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                return DashboardCommand.QUIT
            if event.type == self.pg.KEYDOWN and event.key in (self.pg.K_ESCAPE, self.pg.K_q):
                return DashboardCommand.QUIT
            if event.type == self.pg.KEYDOWN and event.key == self.pg.K_r:
                command = DashboardCommand.RESTART
            if (
                event.type == self.pg.MOUSEBUTTONDOWN
                and event.button == 1
                and restart_rect.collidepoint(event.pos)
            ):
                command = DashboardCommand.RESTART

        c = self.config
        t = self.theme
        self.screen.fill(t.canvas)
        margin = 24
        header_h = 58
        panel_x = c.width - c.panel_width
        game_x = margin
        game_y = header_h + 14
        game_w = panel_x - margin * 2
        game_h = c.height - game_y - margin

        self._text("TypeSafe plays Mario", self.font_title, t.text, margin, 18)
        if run_ended:
            status = "Run ended"
        else:
            status = "Waiting for Jev" if waiting else "Live decision loop"
        status_width = self.font_small.size(status)[0]
        dot_x = panel_x - status_width - 24
        status_color = t.danger if run_ended else (t.warning if waiting else t.accent)
        self.pg.draw.circle(self.screen, status_color, (dot_x, 31), 4)
        self._text(status, self.font_small, t.muted, dot_x + 12, 21)
        hovered = restart_rect.collidepoint(self.pg.mouse.get_pos())
        button_color = t.accent_soft if hovered else t.raised
        self.pg.draw.rect(self.screen, button_color, restart_rect, border_radius=5)
        self.pg.draw.rect(self.screen, t.accent, restart_rect, 1, border_radius=5)
        label = "Restart"
        label_width, label_height = self.font_label.size(label)
        self._text(
            label,
            self.font_label,
            t.text,
            restart_rect.centerx - label_width // 2,
            restart_rect.centery - label_height // 2,
        )
        self._rule(margin, header_h, c.width - margin * 2)
        self._draw_game(frame, game_x, game_y, game_w, game_h)

        x = panel_x + 18
        width = c.panel_width - 42
        y = game_y
        self._text(f"World {snapshot.world}–{snapshot.stage}", self.font_label, t.muted, x, y)
        self._text(f"Decision {decision_index:04d}", self.font_small, t.muted, x + width - 100, y)
        y += 27

        action_name = (
            "Reading state…" if decision is None else decision.action.value.replace("_", " ")
        )
        self._text(action_name, self.font_action, t.text, x, y)
        y += 42
        if decision is None:
            description = "Jev is evaluating the next bounded controller action."
        else:
            description = ACTION_DESCRIPTIONS[decision.action]
        self._text(description, self.font_small, t.muted, x, y)
        y += 32
        self._rule(x, y, width)
        y += 18

        confidence = decision.confidence if decision else 0.0
        latency = decision.latency_ms if decision else 0.0
        metric_width = width // 3
        metrics = (
            ("Confidence", f"{confidence * 100:.0f}%"),
            ("Latency", f"{latency:.0f} ms"),
            ("Reward", f"{episode_reward:+.1f}"),
        )
        for index, (label, value) in enumerate(metrics):
            metric_x = x + index * metric_width
            self._text(label, self.font_small, t.muted, metric_x, y)
            self._text(value, self.font_body, t.text, metric_x, y + 19)
        y += 58
        self._text("Controller probabilities", self.font_label, t.text, x, y)
        y += 24

        probabilities = decision.probabilities if decision else {}
        selected_action = decision.action.value if decision else None
        for action in Action:
            probability = float(probabilities.get(action.value, 0.0))
            self._bar(
                label=action.value.replace("_", " "),
                value=probability,
                x=x,
                y=y,
                width=width,
                selected=action.value == selected_action,
            )
            y += 36

        self._rule(x, y + 2, width)
        y += 17
        self._text("Situation", self.font_label, t.text, x, y)
        y += 24
        jump = decision.jump_needed_probability if decision else None
        danger = decision.danger_score if decision else None
        intent = decision.jump_intent if decision and decision.jump_intent else "unknown"
        if jump is None:
            self._text("Jump diagnostic: unavailable", self.font_small, t.muted, x, y)
        else:
            self._bar(label=f"Start/hold A ({intent})", value=clamp01(jump), x=x, y=y, width=width)
        y += 40
        if danger is None:
            self._text("Danger diagnostic: unavailable", self.font_small, t.muted, x, y)
        else:
            danger = clamp01(danger / 2.0)
            danger_color = t.danger if danger >= 0.66 else t.warning
            self._bar(
                label="Immediate danger", value=danger, x=x, y=y, width=width, color=danger_color
            )
        y += 43

        enemies = ", ".join(enemy.kind for enemy in snapshot.enemies) or "none"
        state_lines = (
            f"Position  {snapshot.x}, {snapshot.y}",
            f"Motion    {snapshot.direction.replace('_', ' ')}",
            f"Nearby    {enemies}",
            f"Time      {snapshot.time_left}",
        )
        for line in state_lines:
            self._text(line, self.font_mono, t.muted, x, y)
            y += 20

        footer = (
            "Click Restart or press R to play again" if run_ended else "R restarts · Esc or Q quits"
        )
        self._text(footer, self.font_small, t.muted, x, c.height - 28)
        self.pg.display.flip()
        self.clock.tick(c.fps)
        return command
