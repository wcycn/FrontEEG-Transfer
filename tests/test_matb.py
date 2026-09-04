from __future__ import annotations

import math

import numpy as np

from fronteeg_transfer.matb import (
    COMMUNICATION_KEYS,
    PROFILES,
    MatbConfig,
    MatbEngine,
)


def rapid_config() -> MatbConfig:
    return MatbConfig(
        sysmon_interval_seconds=(0.1, 0.1),
        sysmon_timeout_seconds=0.2,
        communication_interval_seconds=(0.1, 0.1),
        communication_timeout_seconds=0.2,
        pump_failure_interval_seconds=(0.1, 0.1),
        pump_failure_duration_seconds=0.2,
    )


def test_condition_profiles_match_matb_task_composition() -> None:
    assert not PROFILES["Easy"].resource_management
    assert not PROFILES["Easy"].communications
    assert PROFILES["Medium"].resource_management
    assert not PROFILES["Medium"].communications
    assert PROFILES["Difficult"].resource_management
    assert PROFILES["Difficult"].communications
    assert PROFILES["Difficult"].pump_failures
    assert PROFILES["Difficult"].tracking_noise > PROFILES["Medium"].tracking_noise


def test_engine_is_reproducible_for_a_fixed_seed() -> None:
    first = MatbEngine("Difficult", seed=17, config=rapid_config())
    second = MatbEngine("Difficult", seed=17, config=rapid_config())
    first_events = first.step(0.1, 0.1, (0.01, -0.02))
    second_events = second.step(0.1, 0.1, (0.01, -0.02))

    assert first_events == second_events
    np.testing.assert_allclose(first.tracking_position, second.tracking_position)
    np.testing.assert_allclose(first.tank_levels, second.tank_levels)


def test_system_monitoring_records_correct_response() -> None:
    engine = MatbEngine("Easy", seed=4, config=rapid_config())
    events = engine.step(0.1, 0.1)
    alarm = next(event for event in events if event.event_type == "sysmon_alarm")
    key = str(alarm.detail["indicator"])
    response = engine.handle_key(key, 0.15)

    assert [event.event_type for event in response] == ["sysmon_hit"]
    assert engine.report()["system_monitoring"]["hits"] == 1


def test_medium_enables_resource_dynamics_and_pump_controls() -> None:
    engine = MatbEngine("Medium", seed=9, config=rapid_config())
    initial = engine.tank_levels.copy()
    engine.step(1.0, 1.0)
    events = engine.handle_key("q", 1.0)

    assert not np.array_equal(engine.tank_levels, initial)
    assert [event.event_type for event in events] == ["pump_toggle"]
    report = engine.report()["resource_management"]
    assert report["enabled"] is True
    assert report["pump_toggles"] == 1


def test_difficult_generates_communications_and_pump_failures() -> None:
    engine = MatbEngine("Difficult", seed=12, config=rapid_config())
    events = engine.step(0.1, 0.1)
    event_types = {event.event_type for event in events}

    assert "communication_message" in event_types
    assert "pump_failure" in event_types
    message = engine.communication_message
    assert message is not None
    key = COMMUNICATION_KEYS[str(message["channel"])]
    response = engine.handle_key(key, 0.15)
    assert response


def test_target_communication_requires_channel_frequency_and_confirmation() -> None:
    engine = MatbEngine("Difficult", seed=1, config=rapid_config())
    engine.step(0.1, 0.1)
    message = engine.communication_message

    assert message is not None and message["target"]
    assert message["channel"] == "COM2"
    assert message["frequency"] == 128.3
    assert engine.handle_key("v", 0.12)[0].event_type == (
        "communication_channel_selected"
    )
    for now in (0.13, 0.14, 0.15):
        assert engine.handle_key("right", now)[0].event_type == (
            "communication_frequency_adjusted"
        )
    response = engine.handle_key("return", 0.16)

    assert [event.event_type for event in response] == ["communication_hit"]
    assert engine.report()["communications"]["hits"] == 1


def test_report_contains_finite_behavioral_scores() -> None:
    engine = MatbEngine("Difficult", seed=21, config=rapid_config())
    for index in range(1, 6):
        engine.step(index * 0.1, 0.1)
    engine.finalize(0.6)
    report = engine.report()

    assert math.isfinite(report["tracking"]["rmse_from_center"])
    assert 0.0 <= report["tracking"]["fraction_inside_target"] <= 1.0
    resource = report["resource_management"]
    assert math.isfinite(resource["mean_absolute_main_tank_error"])
    assert 0.0 <= resource["fraction_both_main_tanks_in_range"] <= 1.0
