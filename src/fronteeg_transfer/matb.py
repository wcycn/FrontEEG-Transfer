"""Deterministic state engine for a lightweight MATB-II-style workload task."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CONDITIONS = ("Easy", "Medium", "Difficult")
SYSMON_KEYS = ("1", "2", "3", "4", "5", "6")
PUMP_KEYS = ("q", "w", "e", "r", "t", "y", "u", "i")
COMMUNICATION_KEYS = {"NAV1": "z", "NAV2": "x", "COM1": "c", "COM2": "v"}
RADIO_FREQUENCY_RANGES = {
    "NAV1": (108.0, 117.9),
    "NAV2": (108.0, 117.9),
    "COM1": (118.0, 136.9),
    "COM2": (118.0, 136.9),
}
PUMP_ROUTES = (
    (2, 0),
    (3, 0),
    (4, 1),
    (5, 1),
    (2, 3),
    (4, 5),
    (0, 1),
    (1, 0),
)


@dataclass(frozen=True)
class MatbProfile:
    resource_management: bool
    communications: bool
    pump_failures: bool
    tracking_noise: float
    tracking_center_radius: float


PROFILES = {
    "Easy": MatbProfile(False, False, False, 0.16, 0.14),
    "Medium": MatbProfile(True, False, False, 0.16, 0.14),
    "Difficult": MatbProfile(True, True, True, 0.29, 0.10),
}


@dataclass(frozen=True)
class MatbConfig:
    sysmon_interval_seconds: tuple[float, float] = (7.0, 12.0)
    sysmon_timeout_seconds: float = 3.0
    communication_interval_seconds: tuple[float, float] = (8.0, 14.0)
    communication_timeout_seconds: float = 5.0
    communication_target_probability: float = 0.5
    pump_failure_interval_seconds: tuple[float, float] = (20.0, 35.0)
    pump_failure_duration_seconds: float = 8.0
    tracking_damping: float = 1.4
    tracking_mouse_gain: float = 0.75
    tank_target: float = 2500.0
    tank_tolerance: float = 450.0
    tank_capacity: float = 5000.0
    main_tank_consumption_per_second: float = 20.0
    pump_flow_per_second: float = 18.0
    auxiliary_refill_per_second: float = 8.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> MatbConfig:
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown MATB fields: {sorted(unknown)}")
        converted = dict(values)
        for name in (
            "sysmon_interval_seconds",
            "communication_interval_seconds",
            "pump_failure_interval_seconds",
        ):
            if name in converted:
                converted[name] = tuple(float(value) for value in converted[name])
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        intervals = (
            self.sysmon_interval_seconds,
            self.communication_interval_seconds,
            self.pump_failure_interval_seconds,
        )
        if any(
            len(interval) != 2 or interval[0] <= 0 or interval[1] < interval[0]
            for interval in intervals
        ):
            raise ValueError("MATB event intervals must be positive [minimum, maximum] pairs")
        positive = (
            self.sysmon_timeout_seconds,
            self.communication_timeout_seconds,
            self.pump_failure_duration_seconds,
            self.tracking_damping,
            self.tracking_mouse_gain,
            self.tank_target,
            self.tank_tolerance,
            self.tank_capacity,
            self.main_tank_consumption_per_second,
            self.pump_flow_per_second,
            self.auxiliary_refill_per_second,
        )
        if min(positive) <= 0:
            raise ValueError("MATB timing and dynamics values must be positive")
        if not 0 < self.communication_target_probability < 1:
            raise ValueError("communication_target_probability must be in (0, 1)")
        if self.tank_target + self.tank_tolerance >= self.tank_capacity:
            raise ValueError("tank target range must remain below tank capacity")


@dataclass(frozen=True)
class MatbEvent:
    event_type: str
    detail: dict[str, Any]


class MatbEngine:
    """Frame-rate-independent task state and behavioral scoring."""

    def __init__(
        self,
        condition: str,
        *,
        seed: int,
        config: MatbConfig | None = None,
    ) -> None:
        if condition not in PROFILES:
            raise ValueError(f"unknown MATB condition: {condition}")
        self.condition = condition
        self.profile = PROFILES[condition]
        self.config = config or MatbConfig()
        self.config.validate()
        self.rng = np.random.default_rng(seed)

        self.tracking_position = np.zeros(2, dtype=np.float64)
        self.tracking_velocity = np.zeros(2, dtype=np.float64)
        self._tracking_seconds = 0.0
        self._tracking_squared_error = 0.0
        self._tracking_inside_seconds = 0.0

        self.sysmon_alarm_index: int | None = None
        self._sysmon_alarm_started = 0.0
        self._sysmon_alarm_deadline = math.inf
        self._next_sysmon = self._interval(self.config.sysmon_interval_seconds)
        self._sysmon_hits = 0
        self._sysmon_misses = 0
        self._sysmon_false_alarms = 0
        self._sysmon_reaction_times: list[float] = []

        self.tank_levels = np.asarray(
            [self.config.tank_target, self.config.tank_target, 4000, 4000, 4000, 4000],
            dtype=np.float64,
        )
        self.pump_active = np.zeros(8, dtype=bool)
        self.pump_active[[0, 2]] = True
        self.failed_pumps: dict[int, float] = {}
        self._next_pump_failure = (
            self._interval(self.config.pump_failure_interval_seconds)
            if self.profile.pump_failures
            else math.inf
        )
        self._resource_seconds = 0.0
        self._resource_absolute_error = 0.0
        self._resource_inside_seconds = 0.0
        self._pump_toggles = 0
        self._pump_failures = 0

        self.communication_message: dict[str, Any] | None = None
        self.radio_frequencies = {
            "NAV1": 112.0,
            "NAV2": 114.0,
            "COM1": 124.0,
            "COM2": 128.0,
        }
        self.selected_radio_channel = "NAV1"
        self._communication_deadline = math.inf
        self._next_communication = (
            self._interval(self.config.communication_interval_seconds)
            if self.profile.communications
            else math.inf
        )
        self._communication_targets = 0
        self._communication_distractors = 0
        self._communication_hits = 0
        self._communication_misses = 0
        self._communication_false_alarms = 0
        self._communication_correct_rejections = 0
        self._communication_reaction_times: list[float] = []

    def _interval(self, bounds: tuple[float, float]) -> float:
        return float(self.rng.uniform(bounds[0], bounds[1]))

    def _schedule_sysmon(self, now: float) -> None:
        self._next_sysmon = now + self._interval(self.config.sysmon_interval_seconds)

    def _schedule_communication(self, now: float) -> None:
        self._next_communication = now + self._interval(
            self.config.communication_interval_seconds
        )

    def _expire_events(self, now: float) -> list[MatbEvent]:
        events: list[MatbEvent] = []
        if self.sysmon_alarm_index is not None and now >= self._sysmon_alarm_deadline:
            events.append(
                MatbEvent("sysmon_miss", {"indicator": self.sysmon_alarm_index + 1})
            )
            self._sysmon_misses += 1
            self.sysmon_alarm_index = None
            self._sysmon_alarm_deadline = math.inf
            self._schedule_sysmon(now)

        if self.communication_message is not None and now >= self._communication_deadline:
            message = self.communication_message
            if message["target"]:
                self._communication_misses += 1
                event_type = "communication_miss"
            else:
                self._communication_correct_rejections += 1
                event_type = "communication_correct_rejection"
            events.append(MatbEvent(event_type, dict(message)))
            self.communication_message = None
            self._communication_deadline = math.inf
            self._schedule_communication(now)

        for pump, deadline in list(self.failed_pumps.items()):
            if now >= deadline:
                del self.failed_pumps[pump]
                events.append(MatbEvent("pump_recovered", {"pump": pump + 1}))
        return events

    def _start_due_events(self, now: float) -> list[MatbEvent]:
        events: list[MatbEvent] = []
        if self.sysmon_alarm_index is None and now >= self._next_sysmon:
            self.sysmon_alarm_index = int(self.rng.integers(0, len(SYSMON_KEYS)))
            self._sysmon_alarm_started = now
            self._sysmon_alarm_deadline = now + self.config.sysmon_timeout_seconds
            self._next_sysmon = math.inf
            events.append(
                MatbEvent("sysmon_alarm", {"indicator": self.sysmon_alarm_index + 1})
            )

        if (
            self.profile.communications
            and self.communication_message is None
            and now >= self._next_communication
        ):
            target = bool(self.rng.random() < self.config.communication_target_probability)
            callsign = "EAGLE" if target else str(
                self.rng.choice(("FALCON", "RAVEN", "VIPER"))
            )
            channel = str(self.rng.choice(tuple(COMMUNICATION_KEYS)))
            current_frequency = self.radio_frequencies[channel]
            offset = float(self.rng.choice((-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)))
            lower, upper = RADIO_FREQUENCY_RANGES[channel]
            frequency = round(float(np.clip(current_frequency + offset, lower, upper)), 1)
            self.communication_message = {
                "callsign": callsign,
                "channel": channel,
                "frequency": frequency,
                "target": target,
                "started": now,
            }
            self._communication_targets += int(target)
            self._communication_distractors += int(not target)
            self._communication_deadline = now + self.config.communication_timeout_seconds
            self._next_communication = math.inf
            events.append(MatbEvent("communication_message", dict(self.communication_message)))

        if self.profile.pump_failures and now >= self._next_pump_failure:
            candidates = [index for index in range(8) if index not in self.failed_pumps]
            if candidates:
                pump = int(self.rng.choice(candidates))
                self.pump_active[pump] = False
                self.failed_pumps[pump] = now + self.config.pump_failure_duration_seconds
                self._pump_failures += 1
                events.append(
                    MatbEvent(
                        "pump_failure",
                        {
                            "pump": pump + 1,
                            "duration_seconds": self.config.pump_failure_duration_seconds,
                        },
                    )
                )
            self._next_pump_failure = now + self._interval(
                self.config.pump_failure_interval_seconds
            )
        return events

    def _update_tracking(self, dt: float, mouse_delta: tuple[float, float]) -> None:
        if dt <= 0:
            return
        noise = self.rng.normal(0.0, self.profile.tracking_noise, size=2) * math.sqrt(dt)
        self.tracking_velocity += noise
        self.tracking_velocity *= math.exp(-self.config.tracking_damping * dt)
        self.tracking_position += self.tracking_velocity * dt
        self.tracking_position += (
            np.asarray(mouse_delta, dtype=np.float64) * self.config.tracking_mouse_gain
        )
        self.tracking_position = np.clip(self.tracking_position, -0.36, 0.36)
        distance = float(np.linalg.norm(self.tracking_position))
        self._tracking_seconds += dt
        self._tracking_squared_error += distance * distance * dt
        if distance <= self.profile.tracking_center_radius:
            self._tracking_inside_seconds += dt

    def _update_resource_management(self, dt: float) -> None:
        if not self.profile.resource_management or dt <= 0:
            return
        for pump, (source, destination) in enumerate(PUMP_ROUTES):
            if not self.pump_active[pump] or pump in self.failed_pumps:
                continue
            transfer = min(self.config.pump_flow_per_second * dt, self.tank_levels[source])
            self.tank_levels[source] -= transfer
            self.tank_levels[destination] += transfer
        self.tank_levels[:2] -= self.config.main_tank_consumption_per_second * dt
        self.tank_levels[2:] += self.config.auxiliary_refill_per_second * dt
        self.tank_levels = np.clip(self.tank_levels, 0.0, self.config.tank_capacity)
        deviation = np.abs(self.tank_levels[:2] - self.config.tank_target)
        self._resource_seconds += dt
        self._resource_absolute_error += float(deviation.mean()) * dt
        if bool(np.all(deviation <= self.config.tank_tolerance)):
            self._resource_inside_seconds += dt

    def step(
        self,
        now: float,
        dt: float,
        mouse_delta: tuple[float, float] = (0.0, 0.0),
    ) -> list[MatbEvent]:
        if now < 0 or dt < 0:
            raise ValueError("MATB time values cannot be negative")
        events = self._expire_events(now)
        events.extend(self._start_due_events(now))
        self._update_tracking(dt, mouse_delta)
        self._update_resource_management(dt)
        return events

    def handle_key(self, key: str, now: float) -> list[MatbEvent]:
        key = key.casefold()
        events: list[MatbEvent] = []
        if key in SYSMON_KEYS:
            indicator = SYSMON_KEYS.index(key)
            if self.sysmon_alarm_index == indicator:
                reaction_time = now - self._sysmon_alarm_started
                self._sysmon_hits += 1
                self._sysmon_reaction_times.append(reaction_time)
                events.append(
                    MatbEvent(
                        "sysmon_hit",
                        {"indicator": indicator + 1, "reaction_time_seconds": reaction_time},
                    )
                )
                self.sysmon_alarm_index = None
                self._sysmon_alarm_deadline = math.inf
                self._schedule_sysmon(now)
            else:
                self._sysmon_false_alarms += 1
                events.append(MatbEvent("sysmon_false_alarm", {"indicator": indicator + 1}))

        if key in PUMP_KEYS and self.profile.resource_management:
            pump = PUMP_KEYS.index(key)
            if pump in self.failed_pumps:
                events.append(MatbEvent("pump_toggle_rejected", {"pump": pump + 1}))
            else:
                self.pump_active[pump] = not self.pump_active[pump]
                self._pump_toggles += 1
                events.append(
                    MatbEvent(
                        "pump_toggle",
                        {"pump": pump + 1, "active": bool(self.pump_active[pump])},
                    )
                )

        if self.profile.communications:
            events.extend(self._handle_communication_key(key, now))
        return events

    def _close_communication(self, now: float) -> None:
        self.communication_message = None
        self._communication_deadline = math.inf
        self._schedule_communication(now)

    def _handle_communication_key(self, key: str, now: float) -> list[MatbEvent]:
        radio_keys = (*COMMUNICATION_KEYS.values(), "left", "right", "return")
        if key not in radio_keys:
            return []
        message = self.communication_message
        if message is None:
            return []
        if not message["target"]:
            self._communication_false_alarms += 1
            self._close_communication(now)
            return [MatbEvent("communication_false_alarm", {**message, "key": key})]
        if key in COMMUNICATION_KEYS.values():
            self.selected_radio_channel = next(
                channel
                for channel, selection_key in COMMUNICATION_KEYS.items()
                if selection_key == key
            )
            return [
                MatbEvent(
                    "communication_channel_selected",
                    {**message, "selected_channel": self.selected_radio_channel},
                )
            ]
        if key in ("left", "right"):
            channel = self.selected_radio_channel
            direction = -1.0 if key == "left" else 1.0
            lower, upper = RADIO_FREQUENCY_RANGES[channel]
            self.radio_frequencies[channel] = round(
                float(np.clip(self.radio_frequencies[channel] + 0.1 * direction, lower, upper)),
                1,
            )
            return [
                MatbEvent(
                    "communication_frequency_adjusted",
                    {
                        **message,
                        "selected_channel": channel,
                        "selected_frequency": self.radio_frequencies[channel],
                    },
                )
            ]
        selected_frequency = self.radio_frequencies[self.selected_radio_channel]
        correct = (
            self.selected_radio_channel == message["channel"]
            and math.isclose(selected_frequency, float(message["frequency"]), abs_tol=0.01)
        )
        detail = {
            **message,
            "selected_channel": self.selected_radio_channel,
            "selected_frequency": selected_frequency,
        }
        if correct:
            reaction_time = now - float(message["started"])
            self._communication_hits += 1
            self._communication_reaction_times.append(reaction_time)
            detail["reaction_time_seconds"] = reaction_time
            event_type = "communication_hit"
        else:
            self._communication_false_alarms += 1
            event_type = "communication_incorrect"
        self._close_communication(now)
        return [MatbEvent(event_type, detail)]

    def finalize(self, now: float) -> list[MatbEvent]:
        events = self._expire_events(now)
        if self.sysmon_alarm_index is not None:
            self._sysmon_misses += 1
            events.append(
                MatbEvent("sysmon_miss", {"indicator": self.sysmon_alarm_index + 1})
            )
            self.sysmon_alarm_index = None
        if self.communication_message is not None:
            if self.communication_message["target"]:
                self._communication_misses += 1
                event_type = "communication_miss"
            else:
                self._communication_correct_rejections += 1
                event_type = "communication_correct_rejection"
            events.append(MatbEvent(event_type, dict(self.communication_message)))
            self.communication_message = None
        return events

    def report(self) -> dict[str, Any]:
        tracking_seconds = max(self._tracking_seconds, np.finfo(float).eps)
        resource_seconds = max(self._resource_seconds, np.finfo(float).eps)
        return {
            "condition": self.condition,
            "active_tasks": {
                "tracking": True,
                "system_monitoring": True,
                "resource_management": self.profile.resource_management,
                "communications": self.profile.communications,
            },
            "profile": asdict(self.profile),
            "tracking": {
                "rmse_from_center": math.sqrt(
                    self._tracking_squared_error / tracking_seconds
                ),
                "fraction_inside_target": self._tracking_inside_seconds / tracking_seconds,
            },
            "system_monitoring": {
                "hits": self._sysmon_hits,
                "misses": self._sysmon_misses,
                "false_alarms": self._sysmon_false_alarms,
                "mean_reaction_time_seconds": (
                    float(np.mean(self._sysmon_reaction_times))
                    if self._sysmon_reaction_times
                    else None
                ),
            },
            "resource_management": {
                "enabled": self.profile.resource_management,
                "mean_absolute_main_tank_error": (
                    self._resource_absolute_error / resource_seconds
                    if self.profile.resource_management
                    else None
                ),
                "fraction_both_main_tanks_in_range": (
                    self._resource_inside_seconds / resource_seconds
                    if self.profile.resource_management
                    else None
                ),
                "pump_toggles": self._pump_toggles,
                "pump_failures": self._pump_failures,
                "final_tank_levels": self.tank_levels.tolist(),
            },
            "communications": {
                "enabled": self.profile.communications,
                "targets": self._communication_targets,
                "distractors": self._communication_distractors,
                "hits": self._communication_hits,
                "misses": self._communication_misses,
                "false_alarms": self._communication_false_alarms,
                "correct_rejections": self._communication_correct_rejections,
                "mean_reaction_time_seconds": (
                    float(np.mean(self._communication_reaction_times))
                    if self._communication_reaction_times
                    else None
                ),
            },
        }
