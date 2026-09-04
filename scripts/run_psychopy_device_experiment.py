#!/usr/bin/env python3
"""Run the real-device MATB-style workload experiment and record Fp1/Fp2 over LSL."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

from fronteeg_transfer.config import load_yaml, resolve_path
from fronteeg_transfer.device_experiment import (
    BlockSpec,
    EventLog,
    ProtocolConfig,
    build_schedule,
    schedule_as_dicts,
    signal_quality_summary,
    validate_participant_id,
    write_json,
)
from fronteeg_transfer.lsl_device import LslEegRecorder, LslMarkerOutlet, SyntheticEegRecorder
from fronteeg_transfer.matb import (
    COMMUNICATION_KEYS,
    PUMP_KEYS,
    SYSMON_KEYS,
    MatbConfig,
    MatbEngine,
    MatbEvent,
)
from fronteeg_transfer.matb_view import MatbView


class ExperimentAborted(RuntimeError):
    pass


def select_display_font(configured: Any = None) -> str:
    """Choose an ASCII-capable font available on common experiment computers."""
    return str(configured) if configured else "Arial"


def validate_device_settings(
    device_config: dict[str, Any], channel_indices: tuple[int, int]
) -> str:
    """Reject ambiguous settings before recording an irreplaceable session."""
    if len(channel_indices) != 2 or len(set(channel_indices)) != 2:
        raise ValueError("channel_indices must contain two distinct Fp1/Fp2 indices")
    if min(channel_indices) < 0:
        raise ValueError("channel_indices cannot be negative")
    input_unit = str(device_config.get("input_unit", "microvolts"))
    if input_unit not in {"microvolts", "volts"}:
        raise ValueError("input_unit must be microvolts or volts")
    expected_labels = device_config.get("expected_channel_labels")
    if expected_labels is not None and len(expected_labels) != 2:
        raise ValueError("expected_channel_labels must contain exactly two labels or be null")
    return input_unit


def validate_warmup_signal(
    samples: np.ndarray, timestamps: np.ndarray, sampling_rate: float
) -> dict[str, Any]:
    """Fail fast on a stalled, malformed or duplicated selected EEG stream."""
    values = np.asarray(samples, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 10:
        raise RuntimeError(f"invalid warm-up EEG shape: {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(times).all():
        raise RuntimeError("warm-up EEG contains non-finite samples or timestamps")
    if len(times) != len(values) or not np.all(np.diff(times) > 0):
        raise RuntimeError("warm-up EEG timestamps are not strictly increasing")
    observed_rate = 1.0 / float(np.median(np.diff(times)))
    relative_error = abs(observed_rate - sampling_rate) / sampling_rate
    if relative_error > 0.20:
        raise RuntimeError(
            f"observed EEG rate {observed_rate:.2f} Hz differs from declared "
            f"{sampling_rate:.2f} Hz by more than 20%"
        )
    if np.any(np.ptp(values, axis=0) == 0):
        raise RuntimeError("one or more selected EEG channels are constant")
    if np.array_equal(values[:, 0], values[:, 1]):
        raise RuntimeError("the two selected EEG channels are exactly identical")
    return {
        "samples": int(len(values)),
        "observed_sampling_rate_hz": observed_rate,
        "channel_standard_deviation_raw_unit": values.std(axis=0).tolist(),
        "channel_peak_to_peak_raw_unit": np.ptp(values, axis=0).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/device_experiment.yaml")
    parser.add_argument("--participant", required=True, help="Pseudonym only, e.g. self01")
    parser.add_argument("--stream-name", default=None)
    parser.add_argument("--simulate-eeg", action="store_true")
    parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="Automation only; requires --simulate-eeg and must never be used for real data.",
    )
    parser.add_argument(
        "--windowed", action="store_true", help="Use a window instead of full screen"
    )
    parser.add_argument("--skip-artifact", action="store_true")
    return parser.parse_args()


class PsychoPyExperiment:
    def __init__(
        self,
        *,
        window: Any,
        visual: Any,
        core: Any,
        mouse: Any,
        keyboard: Any,
        recorder: Any,
        marker: Any,
        events: EventLog,
        protocol: ProtocolConfig,
        matb_config: MatbConfig,
        display: dict[str, Any],
        auto_advance: bool = False,
    ) -> None:
        self.window = window
        self.core = core
        self.mouse = mouse
        self.keyboard = keyboard
        self.recorder = recorder
        self.marker = marker
        self.events = events
        self.protocol = protocol
        self.matb_config = matb_config
        self.auto_advance = auto_advance
        self.block_reports: list[dict[str, Any]] = []
        font = select_display_font(display.get("font"))
        self.text = visual.TextStim(
            window,
            text="",
            font=font,
            color=display.get("text_color", "white"),
            height=float(display.get("text_height", 0.055)),
            wrapWidth=1.65,
            units="norm",
        )
        self.symbol = visual.TextStim(
            window,
            text="",
            font=font,
            color=display.get("text_color", "white"),
            height=float(display.get("symbol_height", 0.22)),
            units="norm",
        )
        self.matb_view = MatbView(window, visual, font=font)

    def emit(
        self,
        *,
        phase: str,
        block_id: str,
        condition: str,
        event_type: str,
        timestamp: float | None = None,
        **values: Any,
    ) -> float:
        timestamp = self.recorder.clock() if timestamp is None else timestamp
        self.events.append(
            lsl_timestamp=f"{timestamp:.9f}",
            phase=phase,
            block_id=block_id,
            condition=condition,
            event_type=event_type,
            **values,
        )
        if self.marker is not None:
            payload = f"{phase}|{block_id}|{condition}|{event_type}"
            self.marker.push(payload, timestamp)
        return timestamp

    def _keys(self, names: list[str]) -> list[Any]:
        keys = self.keyboard.getKeys(keyList=names, waitRelease=False, clear=True)
        if any(key.name == "escape" for key in keys):
            raise ExperimentAborted("escape pressed")
        return keys

    def show_and_wait(self, message: str, key: str = "space") -> None:
        self.mouse.setVisible(True)
        self.keyboard.clearEvents()
        while True:
            self.text.text = message
            self.text.draw()
            self.window.flip()
            if self.auto_advance:
                return
            if any(press.name == key for press in self._keys([key, "escape"])):
                return

    def timed_display(self, message: str, duration: float, *, large: bool = False) -> None:
        clock = self.core.Clock()
        while clock.getTime() < duration:
            stimulus = self.symbol if large else self.text
            stimulus.text = message
            stimulus.draw()
            self.window.flip()
            self._keys(["escape"])

    def countdown(self) -> None:
        for value in range(self.protocol.countdown_seconds, 0, -1):
            self.timed_display(str(value), 1.0, large=True)

    def _emit_matb_events(self, block: BlockSpec, events: list[MatbEvent]) -> None:
        for event in events:
            self.emit(
                phase=block.phase,
                block_id=block.block_id,
                condition=block.condition,
                event_type=event.event_type,
                detail=json.dumps(event.detail, ensure_ascii=True, separators=(",", ":")),
            )

    def _block_instruction(self, engine: MatbEngine, phase: str) -> str:
        active = ["TRACKING", "SYSTEM MONITORING"]
        if engine.profile.resource_management:
            active.append("RESOURCE MANAGEMENT")
        if engine.profile.communications:
            active.append("COMMUNICATIONS")
        return (
            ("PRACTICE RUN\n\n" if phase == "practice" else "NEXT RUN\n\n")
            + f"Active panels: {', '.join(active)}\n\n"
            "TRACKING: move the mouse to push the orange target into the center box.\n"
            "SYSTEM MONITORING: press 1-6 when the matching indicator turns red.\n"
            "RESOURCE MANAGEMENT: keep tanks A and B near 2500 using Q W E R T Y U I.\n"
            "COMMUNICATIONS: for EAGLE only, select Z/X/C/V, tune with arrow keys, "
            "then press ENTER.\n\n"
            "Keep working on all active panels at the same time.\n"
            "Press SPACE when you are ready."
        )

    def ask_mental_effort(self, block: BlockSpec) -> int:
        if self.auto_advance:
            rating = 5
            self.emit(
                phase=block.phase,
                block_id=block.block_id,
                condition=block.condition,
                event_type="mental_effort_rating",
                detail=json.dumps({"rating_0_to_9": rating, "automated": True}),
            )
            return rating
        self.mouse.setVisible(True)
        self.keyboard.clearEvents()
        keys = [str(value) for value in range(10)]
        while True:
            self.text.text = (
                "How much mental effort did this run require?\n\n"
                "0 = no effort                         9 = extreme effort\n\n"
                "Press one number from 0 to 9."
            )
            self.text.draw()
            self.window.flip()
            presses = self._keys([*keys, "escape"])
            if presses:
                rating = int(presses[0].name)
                self.emit(
                    phase=block.phase,
                    block_id=block.block_id,
                    condition=block.condition,
                    event_type="mental_effort_rating",
                    detail=json.dumps({"rating_0_to_9": rating}),
                )
                return rating

    def run_matb_block(self, block: BlockSpec) -> None:
        engine = MatbEngine(block.condition, seed=block.seed, config=self.matb_config)
        self.show_and_wait(self._block_instruction(engine, block.phase))
        self.countdown()
        self.keyboard.clearEvents()
        self.mouse.setVisible(False)
        self.mouse.setPos((0, 0))
        self.mouse.getPos()
        task_clock = self.core.Clock()
        previous = 0.0
        self.emit(
            phase=block.phase,
            block_id=block.block_id,
            condition=block.condition,
            event_type="block_start",
            detail=json.dumps(
                {
                    "paradigm": "MATB-lite",
                    "duration_seconds": block.duration_seconds,
                    "active_tasks": engine.report()["active_tasks"],
                },
                separators=(",", ":"),
            ),
        )
        allowed_keys = [
            "escape",
            *SYSMON_KEYS,
            *PUMP_KEYS,
            *COMMUNICATION_KEYS.values(),
            "left",
            "right",
            "return",
        ]
        while True:
            elapsed = task_clock.getTime()
            if elapsed >= block.duration_seconds:
                break
            dt = min(0.1, max(0.0, elapsed - previous))
            previous = elapsed
            delta = tuple(float(value) for value in self.mouse.getRel())
            self.mouse.setPos((0, 0))
            self.mouse.getPos()
            self._emit_matb_events(block, engine.step(elapsed, dt, delta))
            for press in self._keys(allowed_keys):
                self._emit_matb_events(block, engine.handle_key(press.name, elapsed))
            self.matb_view.draw(engine, block.duration_seconds - elapsed)
            self.window.flip()
        self._emit_matb_events(block, engine.finalize(block.duration_seconds))
        report = {
            "phase": block.phase,
            "block_id": block.block_id,
            "duration_seconds": block.duration_seconds,
            **engine.report(),
        }
        self.emit(
            phase=block.phase,
            block_id=block.block_id,
            condition=block.condition,
            event_type="block_end",
            detail=json.dumps(report, ensure_ascii=True, separators=(",", ":")),
        )
        report["mental_effort_rating_0_to_9"] = (
            None if block.phase == "practice" else self.ask_mental_effort(block)
        )
        self.block_reports.append(report)

    def run_workload(self, schedule: list[BlockSpec]) -> None:
        introduction = (
            "MATB-LITE MENTAL WORKLOAD EXPERIMENT\n\n"
            "You will operate up to four tasks at the same time:\n"
            "continuous tracking, system monitoring, resource management, and radio "
            "communications.\n\n"
            "Each run activates a different task combination. Accuracy and reaction time "
            "are recorded. Keep your head and body still and blink naturally.\n\n"
            "Press SPACE to read the first run instructions."
        )
        self.show_and_wait(introduction)
        previous_phase = ""
        for index, block in enumerate(schedule):
            if block.phase != previous_phase:
                message = {
                    "practice": (
                        "Practice stage: learn each task combination. Practice EEG will "
                        "not be used for calibration or testing."
                    ),
                    "calibration": "Stage 1: short calibration runs.",
                    "test": (
                        "Stage 2: independent test runs. Calibration data will not be "
                        "mixed with this stage."
                    ),
                }[block.phase]
                self.show_and_wait(message + "\n\nPress SPACE to continue.")
                previous_phase = block.phase
            self.run_matb_block(block)
            if index < len(schedule) - 1:
                break_seconds = (
                    min(10.0, self.protocol.break_seconds)
                    if block.phase == "practice"
                    else self.protocol.break_seconds
                )
                self.timed_display("Break. Please relax.", break_seconds)

    def run_artifact_stress(self, artifact: dict[str, Any]) -> None:
        self.show_and_wait(
            "The following artifact stress test is separate from the workload task. "
            "It will not be used as labeled workload data.\n\n"
            "Follow the prompts for eyes-open rest, blinking, and head movement.\n\n"
            "Press SPACE to begin."
        )
        segments = (
            ("eyes_open", float(artifact.get("eyes_open_seconds", 20.0))),
            ("paced_blink", float(artifact.get("paced_blink_seconds", 20.0))),
            ("head_motion", float(artifact.get("head_motion_seconds", 20.0))),
        )
        cue_interval = float(artifact.get("cue_interval_seconds", 2.0))
        cue_display = float(artifact.get("cue_display_seconds", 0.6))
        if any(duration <= 0 for _, duration in segments):
            raise ValueError("artifact segment durations must be positive")
        if cue_interval <= 0 or not 0 < cue_display < cue_interval:
            raise ValueError("artifact cue display must be positive and shorter than interval")
        for segment, duration in segments:
            instructions = {
                "eyes_open": "Keep still, keep your eyes open, and fixate on the cross.",
                "paced_blink": "Blink once when BLINK appears; otherwise fixate on the cross.",
                "head_motion": "Slowly turn as prompted, then return your head to the center.",
            }[segment]
            self.show_and_wait(instructions + "\n\nPress SPACE to begin.")
            self.countdown()
            self.emit(
                phase="artifact",
                block_id=segment,
                condition="",
                event_type="block_start",
            )
            clock = self.core.Clock()
            next_cue = 0.0
            cue_visible_until = 0.0
            cue_index = 0
            message = "+"
            while clock.getTime() < duration:
                elapsed = clock.getTime()
                if segment == "eyes_open":
                    message = "+"
                elif elapsed >= next_cue:
                    if segment == "paced_blink":
                        message = "BLINK"
                    else:
                        message = "TURN LEFT" if cue_index % 2 == 0 else "TURN RIGHT"
                    self.emit(
                        phase="artifact",
                        block_id=segment,
                        condition="",
                        event_type="artifact_cue",
                        detail=message,
                    )
                    cue_index += 1
                    cue_visible_until = elapsed + cue_display
                    next_cue += cue_interval
                elif elapsed >= cue_visible_until:
                    message = "+"
                self.text.text = message
                self.text.draw()
                self.window.flip()
                self._keys(["escape"])
            self.emit(
                phase="artifact",
                block_id=segment,
                condition="",
                event_type="block_end",
            )


def main() -> None:
    args = parse_args()
    participant = validate_participant_id(args.participant)
    if args.auto_advance and not args.simulate_eeg:
        raise ValueError("--auto-advance is restricted to --simulate-eeg")
    config = load_yaml(args.config)
    protocol = ProtocolConfig.from_mapping(dict(config["protocol"]))
    matb_config = MatbConfig.from_mapping(dict(config.get("matb", {})))
    schedule = build_schedule(protocol)
    device_config = dict(config.get("device", {}))
    display_config = dict(config.get("display", {}))
    artifact_config = dict(config.get("artifact", {}))
    output_root = resolve_path(config.get("output_root", "data/private_device_sessions"))
    session_id = f"{participant}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    session_directory = output_root / session_id
    session_directory.mkdir(parents=True, exist_ok=False)

    channel_indices = tuple(int(value) for value in device_config.get("channel_indices", [0, 1]))
    input_unit = validate_device_settings(device_config, channel_indices)
    recovery_path = session_directory / "raw_eeg_recovery.csv"
    stream_name = args.stream_name or device_config.get("stream_name")
    if args.simulate_eeg:
        recorder: Any = SyntheticEegRecorder(
            float(device_config.get("fallback_sampling_rate", 250.0)),
            protocol.seed,
            recovery_path=recovery_path,
        )
        marker = None
    else:
        recorder = LslEegRecorder(
            stream_name=stream_name,
            channel_indices=channel_indices,
            resolve_timeout_seconds=float(device_config.get("resolve_timeout_seconds", 8.0)),
            fallback_sampling_rate=device_config.get("fallback_sampling_rate"),
            recovery_path=recovery_path,
        )
        marker = (
            LslMarkerOutlet(session_id)
            if bool(device_config.get("publish_markers", True))
            else None
        )
    if max(channel_indices) >= recorder.metadata.channel_count:
        raise ValueError(
            f"channel indices {channel_indices} exceed stream count "
            f"{recorder.metadata.channel_count}"
        )

    try:
        from psychopy import core, event, visual
        from psychopy.hardware import keyboard
    except ImportError as error:
        raise RuntimeError(
            "PsychoPy is required. Install requirements-device.txt in Python 3.11."
        ) from error

    stream_metadata = dataclasses.asdict(recorder.metadata)
    stream_xml = stream_metadata.pop("stream_xml")
    if stream_xml:
        (session_directory / "lsl_stream_info.xml").write_text(stream_xml, encoding="utf-8")
    write_json(
        session_directory / "session_plan.json",
        {
            "schema_version": 3,
            "session_id": session_id,
            "participant_pseudonym": participant,
            "paradigm": "MATB-lite",
            "status": "planned",
            "input_unit": input_unit,
            "selected_channel_indices": list(channel_indices),
            "stream": stream_metadata,
            "protocol": dataclasses.asdict(protocol),
            "matb_config": dataclasses.asdict(matb_config),
            "artifact_config": artifact_config,
            "schedule": schedule_as_dicts(schedule),
        },
    )
    events = EventLog(session_directory / "events.tsv")
    window = None
    task: PsychoPyExperiment | None = None
    status = "initializing"
    failure = ""
    workload_completed = False
    warmup_quality: dict[str, Any] = {}
    window = visual.Window(
        size=tuple(display_config.get("window_size", [1280, 800])),
        fullscr=bool(display_config.get("full_screen", True)) and not args.windowed,
        screen=int(display_config.get("screen", 0)),
        color=display_config.get("background_color", [-0.15, -0.15, -0.15]),
        units="norm",
        waitBlanking=True,
        allowGUI=args.windowed,
    )
    task = PsychoPyExperiment(
        window=window,
        visual=visual,
        core=core,
        mouse=event.Mouse(win=window),
        keyboard=keyboard.Keyboard(),
        recorder=recorder,
        marker=marker,
        events=events,
        protocol=protocol,
        matb_config=matb_config,
        display=display_config,
        auto_advance=args.auto_advance,
    )
    recorder.start()
    try:
        warmup_samples = max(
            10,
            int(
                recorder.metadata.nominal_sampling_rate
                * float(device_config.get("warmup_seconds", 1.0))
            ),
        )
        warmup_deadline = time.monotonic() + float(
            device_config.get("warmup_timeout_seconds", 8.0)
        )
        while len(recorder.snapshot()[0]) < warmup_samples:
            if recorder.error is not None:
                raise RuntimeError(f"EEG recorder failed during warm-up: {recorder.error!r}")
            if time.monotonic() >= warmup_deadline:
                received = len(recorder.snapshot()[0])
                raise RuntimeError(
                    f"EEG warm-up timed out: received {received}/{warmup_samples} samples"
                )
            time.sleep(0.05)
        warmup_values, warmup_times, _ = recorder.snapshot()
        warmup_quality = validate_warmup_signal(
            warmup_values, warmup_times, recorder.metadata.nominal_sampling_rate
        )
        expected_labels = device_config.get("expected_channel_labels")
        if expected_labels is not None:
            selected_labels = [recorder.metadata.channel_labels[index] for index in channel_indices]
            if [str(value).casefold() for value in expected_labels] != [
                value.casefold() for value in selected_labels
            ]:
                raise RuntimeError(
                    f"selected LSL channel labels {selected_labels} do not match "
                    f"expected_channel_labels {expected_labels}"
                )
        selected_labels = [recorder.metadata.channel_labels[index] for index in channel_indices]
        task.show_and_wait(
            "EEG PREFLIGHT PASSED\n\n"
            f"Stream: {recorder.metadata.name}\n"
            f"Selected indices: {list(channel_indices)}\n"
            f"LSL labels: {selected_labels}\n"
            f"Configured unit: {input_unit}\n"
            f"Observed rate: {warmup_quality['observed_sampling_rate_hz']:.2f} Hz\n\n"
            f"Timestamp repairs: {recorder.timestamp_diagnostics['repaired_samples']} "
            "samples\n\n"
            "Confirm these are Fp1 then Fp2. Press SPACE to continue."
        )
        status = "running"
        task.run_workload(schedule)
        workload_completed = True
        if bool(artifact_config.get("enabled", True)) and not args.skip_artifact:
            task.run_artifact_stress(artifact_config)
        status = "completed"
        task.show_and_wait(
            "Experiment complete. Thank you.\n\nPress SPACE to save and exit."
        )
    except ExperimentAborted as error:
        status = "aborted_after_workload" if workload_completed else "aborted"
        failure = str(error)
        events.append(
            lsl_timestamp=f"{recorder.clock():.9f}",
            phase="system",
            block_id="",
            condition="",
            event_type="abort",
            detail=failure,
        )
    except Exception as error:
        status = "failed"
        failure = repr(error)
        raise
    finally:
        if window is not None:
            window.close()
        recorder.stop()
        time.sleep(0.05)
        samples, timestamps, source_timestamps = recorder.snapshot()
        all_stream_samples = recorder.full_snapshot()
        if recorder.error is not None:
            status = "failed"
            failure = f"recorder thread failed: {recorder.error!r}"
        raw_partial = session_directory / "raw_eeg.partial.npz"
        np.savez_compressed(
            raw_partial,
            samples=samples,
            all_stream_samples=all_stream_samples,
            local_timestamps=timestamps,
            lsl_timestamps=source_timestamps,
            sampling_rate=np.asarray(recorder.metadata.nominal_sampling_rate),
            channel_names=np.asarray(["Fp1", "Fp2"]),
            all_channel_labels=np.asarray(recorder.metadata.channel_labels),
            all_channel_units=np.asarray(recorder.metadata.channel_units),
            selected_channel_indices=np.asarray(channel_indices),
            timestamp_repaired_samples=np.asarray(
                recorder.timestamp_diagnostics["repaired_samples"]
            ),
        )
        raw_partial.replace(session_directory / "raw_eeg.npz")
        events.write_tsv(session_directory / "events.tsv")
        values_uv = samples * (1e6 if input_unit == "volts" else 1.0)
        quality = (
            signal_quality_summary(
                values_uv, timestamps, recorder.metadata.nominal_sampling_rate
            )
            if len(samples)
            else {"samples": 0, "error": "no EEG samples received"}
        )
        metadata = {
            "schema_version": 3,
            "paradigm": "MATB-lite",
            "session_id": session_id,
            "participant_pseudonym": participant,
            "created_local_time": datetime.now().astimezone().isoformat(),
            "status": status,
            "failure": failure,
            "workload_completed": workload_completed,
            "simulated_eeg": args.simulate_eeg,
            "input_unit": input_unit,
            "selected_channel_indices": list(channel_indices),
            "selected_channel_names": ["Fp1", "Fp2"],
            "selected_lsl_channel_labels": [
                recorder.metadata.channel_labels[index] for index in channel_indices
            ],
            "all_lsl_channel_labels": list(recorder.metadata.channel_labels),
            "all_lsl_channel_units": list(recorder.metadata.channel_units),
            "sampling_rate": recorder.metadata.nominal_sampling_rate,
            "lsl_time_correction_seconds": recorder.metadata.time_correction_seconds,
            "stream_name": recorder.metadata.name,
            "protocol": dataclasses.asdict(protocol),
            "matb_config": dataclasses.asdict(matb_config),
            "schedule": schedule_as_dicts(schedule),
            "artifact_protocol_enabled": (
                bool(artifact_config.get("enabled", True)) and not args.skip_artifact
            ),
            "behavior": task.block_reports if task is not None else [],
            "warmup_quality": warmup_quality,
            "signal_quality": quality,
            "timestamp_diagnostics": recorder.timestamp_diagnostics,
            "recovery_file": "raw_eeg_recovery.csv",
            "privacy": "Raw EEG is private and excluded from version control.",
        }
        write_json(session_directory / "session_metadata.json", metadata)
        print(f"session={session_directory} status={status} samples={len(samples)}")
        if recorder.error is not None:
            print(f"recorder_error={recorder.error!r}", file=sys.stderr)
        if workload_completed and not len(samples):
            raise RuntimeError("experiment completed but no EEG samples were received")


if __name__ == "__main__":
    main()
