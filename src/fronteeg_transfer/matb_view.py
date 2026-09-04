"""PsychoPy rendering layer for the lightweight MATB task."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .matb import COMMUNICATION_KEYS, PUMP_KEYS, PUMP_ROUTES, MatbEngine


class MatbView:
    """Render four MATB-style panels without owning task state or timing."""

    PANEL_CENTERS = {
        "system": (-0.5, 0.5),
        "tracking": (0.5, 0.5),
        "communication": (-0.5, -0.5),
        "resource": (0.5, -0.5),
    }

    def __init__(self, window: Any, visual: Any, *, font: str) -> None:
        self.window = window
        self.visual = visual
        self.font = font
        self.panels = {
            name: visual.Rect(
                window,
                pos=center,
                width=0.92,
                height=0.86,
                lineColor="#9aa4b2",
                fillColor="#111827",
                lineWidth=1.5,
                units="norm",
            )
            for name, center in self.PANEL_CENTERS.items()
        }
        titles = {
            "system": "SYSTEM MONITORING",
            "tracking": "TRACKING",
            "communication": "COMMUNICATIONS",
            "resource": "RESOURCE MANAGEMENT",
        }
        self.titles = {
            name: self._text(title, (center[0], center[1] + 0.36), 0.040)
            for (name, center), title in zip(
                self.PANEL_CENTERS.items(), titles.values(), strict=True
            )
        }
        self.timer = self._text("", (0.0, 0.965), 0.035)

        system_positions = (
            (-0.77, 0.61),
            (-0.50, 0.61),
            (-0.23, 0.61),
            (-0.77, 0.34),
            (-0.50, 0.34),
            (-0.23, 0.34),
        )
        self.system_boxes = [
            visual.Rect(
                window,
                pos=position,
                width=0.20,
                height=0.16,
                lineColor="#d1d5db",
                fillColor="#1f9d55",
                units="norm",
            )
            for position in system_positions
        ]
        system_names = ("LIGHT", "LIGHT", "GAUGE", "GAUGE", "GAUGE", "GAUGE")
        self.system_labels = [
            self._text(f"{index + 1}\n{label}", position, 0.035)
            for index, (position, label) in enumerate(
                zip(system_positions, system_names, strict=True)
            )
        ]
        self.system_help = self._text(
            "Press 1-6 when the matching indicator turns red.", (-0.5, 0.13), 0.028
        )

        self.tracking_bounds = visual.Rect(
            window,
            pos=self.PANEL_CENTERS["tracking"],
            width=0.72,
            height=0.58,
            lineColor="#d1d5db",
            fillColor="#0b1220",
            units="norm",
        )
        self.tracking_center = visual.Rect(
            window,
            pos=self.PANEL_CENTERS["tracking"],
            width=0.28,
            height=0.28,
            lineColor="#34d399",
            fillColor=None,
            lineWidth=2.0,
            units="norm",
        )
        self.tracking_target = visual.Circle(
            window,
            pos=self.PANEL_CENTERS["tracking"],
            radius=0.035,
            lineColor="#f9fafb",
            fillColor="#f59e0b",
            units="norm",
        )
        self.tracking_help = self._text(
            "Move the mouse to push the target back into the center box.",
            (0.5, 0.13),
            0.027,
        )

        tank_x = (0.18, 0.31, 0.44, 0.57, 0.70, 0.83)
        self.tank_outlines = [
            visual.Rect(
                window,
                pos=(x, -0.51),
                width=0.085,
                height=0.48,
                lineColor="#d1d5db",
                fillColor="#0b1220",
                units="norm",
            )
            for x in tank_x
        ]
        self.tank_fills = [
            visual.Rect(
                window,
                pos=(x, -0.72),
                width=0.071,
                height=0.01,
                lineColor=None,
                fillColor="#3b82f6",
                units="norm",
            )
            for x in tank_x
        ]
        self.tank_labels = [
            self._text(name, (x, -0.22), 0.032)
            for name, x in zip("ABCDEF", tank_x, strict=True)
        ]
        self.pump_status = self._text("", (0.5, -0.82), 0.025)
        self.resource_help = self._text(
            "Keep tanks A and B near 2500. Toggle pumps with Q W E R T Y U I.",
            (0.5, -0.91),
            0.024,
        )
        self.resource_inactive = self._text("INACTIVE IN THIS RUN", (0.5, -0.52), 0.045)

        self.communication_own = self._text("YOUR CALLSIGN: EAGLE", (-0.5, -0.28), 0.040)
        self.communication_mapping = self._text("", (-0.5, -0.43), 0.029)
        self.communication_message = self._text("LISTENING...", (-0.5, -0.66), 0.041)
        self.communication_help = self._text(
            "EAGLE only: select Z/X/C/V, tune with LEFT/RIGHT, press ENTER.",
            (-0.5, -0.87),
            0.025,
        )
        self.communication_inactive = self._text(
            "INACTIVE IN THIS RUN", (-0.5, -0.52), 0.045
        )

    def _text(self, text: str, pos: tuple[float, float], height: float) -> Any:
        return self.visual.TextStim(
            self.window,
            text=text,
            pos=pos,
            font=self.font,
            color="#f3f4f6",
            height=height,
            wrapWidth=0.84,
            units="norm",
        )

    def _draw_system_monitoring(self, engine: MatbEngine) -> None:
        for index, (box, label) in enumerate(
            zip(self.system_boxes, self.system_labels, strict=True)
        ):
            box.fillColor = "#dc2626" if engine.sysmon_alarm_index == index else "#1f9d55"
            box.draw()
            label.draw()
        self.system_help.draw()

    def _draw_tracking(self, engine: MatbEngine) -> None:
        radius = engine.profile.tracking_center_radius
        self.tracking_center.width = radius * 2
        self.tracking_center.height = radius * 2
        self.tracking_target.pos = (
            0.5 + float(engine.tracking_position[0]),
            0.5 + float(engine.tracking_position[1]),
        )
        self.tracking_bounds.draw()
        self.tracking_center.draw()
        self.tracking_target.draw()
        self.tracking_help.draw()

    def _draw_resource_management(self, engine: MatbEngine) -> None:
        if not engine.profile.resource_management:
            self.resource_inactive.draw()
            return
        for index, (outline, fill, label) in enumerate(
            zip(self.tank_outlines, self.tank_fills, self.tank_labels, strict=True)
        ):
            ratio = float(
                np.clip(engine.tank_levels[index] / engine.config.tank_capacity, 0.0, 1.0)
            )
            height = max(0.006, 0.44 * ratio)
            fill.height = height
            fill.pos = (outline.pos[0], -0.73 + height / 2)
            if index < 2:
                deviation = abs(engine.tank_levels[index] - engine.config.tank_target)
                fill.fillColor = (
                    "#3b82f6" if deviation <= engine.config.tank_tolerance else "#dc2626"
                )
            else:
                fill.fillColor = "#3b82f6"
            outline.draw()
            fill.draw()
            label.draw()
        statuses: list[str] = []
        tank_names = "ABCDEF"
        for index, (key, route) in enumerate(zip(PUMP_KEYS, PUMP_ROUTES, strict=True)):
            if index in engine.failed_pumps:
                state = "FAIL"
            else:
                state = "ON" if engine.pump_active[index] else "OFF"
            source, destination = route
            statuses.append(
                f"{key.upper()}:{tank_names[source]}>{tank_names[destination]} {state}"
            )
        self.pump_status.text = "   ".join(statuses[:4]) + "\n" + "   ".join(statuses[4:])
        self.pump_status.draw()
        self.resource_help.draw()

    def _draw_communications(self, engine: MatbEngine) -> None:
        if not engine.profile.communications:
            self.communication_inactive.draw()
            return
        self.communication_own.draw()
        mapping_lines = []
        for channel, key in COMMUNICATION_KEYS.items():
            selector = ">" if channel == engine.selected_radio_channel else " "
            mapping_lines.append(
                f"{selector} {channel} [{key.upper()}] {engine.radio_frequencies[channel]:.1f}"
            )
        self.communication_mapping.text = "\n".join(mapping_lines)
        self.communication_mapping.draw()
        message = engine.communication_message
        if message is None:
            self.communication_message.text = "LISTENING..."
            self.communication_message.color = "#9ca3af"
        else:
            self.communication_message.text = (
                f"{message['callsign']}: SET {message['channel']} TO "
                f"{message['frequency']:.1f}"
            )
            self.communication_message.color = "#fbbf24"
        self.communication_message.draw()
        self.communication_help.draw()

    def draw(self, engine: MatbEngine, remaining_seconds: float) -> None:
        for panel in self.panels.values():
            panel.draw()
        for title in self.titles.values():
            title.draw()
        self.timer.text = f"TIME REMAINING  {max(0, int(math.ceil(remaining_seconds))):03d} s"
        self.timer.draw()
        self._draw_system_monitoring(engine)
        self._draw_tracking(engine)
        self._draw_resource_management(engine)
        self._draw_communications(engine)
