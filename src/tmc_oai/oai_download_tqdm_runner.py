from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


TRUE_TOKENS = {"1", "true", "yes", "y", "t"}
EXPECTED_TOTAL_RE = re.compile(r"Beginning download of (?:the remaining )?(\d+) files", re.IGNORECASE)
QUEUED_PROGRESS_RE = re.compile(r"(\d+)/(\d+) queued files downloaded so far", re.IGNORECASE)
SPEED_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:MiB|MB|M)\s*/\s*s", re.IGNORECASE)


@dataclass
class PackagePlan:
    package_id: str
    cmd: list[str]
    expected_total: int | None
    timepoint_label: str


def _parse_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _package_progress_root(package_id: str) -> Path:
    return (
        Path.home()
        / "NDA"
        / "nda-tools"
        / "downloadcmd"
        / "packages"
        / str(package_id)
        / ".download-progress"
    )


def _list_progress_reports(package_id: str) -> list[Path]:
    root = _package_progress_root(package_id)
    if not root.exists():
        return []
    return sorted(root.glob("*/download-progress-report.csv"))


def _choose_active_report(package_id: str, baseline_reports: set[Path]) -> Path | None:
    reports = _list_progress_reports(package_id)
    if not reports:
        return None
    new_reports = [path for path in reports if path not in baseline_reports]
    pool = new_reports if new_reports else reports
    return max(pool, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)


class ReportTracker:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.offset = 0
        self.col_file_id = -1
        self.col_exists = -1
        self.col_actual_size = -1
        self.col_expected_size = -1
        self.col_expected_location = -1
        self.seen_ids: set[str] = set()
        self.completed_files = 0
        self.completed_bytes = 0
        self.latest_file_path = ""

    def set_path(self, path: Path | None) -> None:
        if path is None:
            return
        if self.path is None or path != self.path:
            self.path = path
            self.offset = 0
            self.col_file_id = -1
            self.col_exists = -1
            self.col_actual_size = -1
            self.col_expected_size = -1
            self.col_expected_location = -1
            self.seen_ids.clear()
            self.completed_files = 0
            self.completed_bytes = 0
            self.latest_file_path = ""

    def _parse_header(self, header: list[str]) -> None:
        normalized = [str(col).strip().lower() for col in header]
        def idx(name: str) -> int:
            return normalized.index(name) if name in normalized else -1

        self.col_file_id = idx("package_file_id")
        self.col_exists = idx("exists")
        self.col_actual_size = idx("actual_file_size")
        self.col_expected_size = idx("expected_file_size")
        self.col_expected_location = idx("package_file_expected_location")

    def poll(self) -> tuple[int, int]:
        if self.path is None or not self.path.exists():
            return self.completed_files, self.completed_bytes

        try:
            file_size = self.path.stat().st_size
        except Exception:
            return self.completed_files, self.completed_bytes

        if file_size < self.offset:
            self.offset = 0
            self.seen_ids.clear()
            self.completed_files = 0
            self.completed_bytes = 0
            self.col_file_id = -1
            self.col_exists = -1
            self.col_actual_size = -1
            self.col_expected_size = -1
            self.col_expected_location = -1

        with self.path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            handle.seek(self.offset)
            reader = csv.reader(handle)
            if self.offset == 0:
                header = next(reader, None)
                if not header:
                    self.offset = handle.tell()
                    return self.completed_files, self.completed_bytes
                self._parse_header(list(header))

            for row in reader:
                if not row:
                    continue
                file_id = ""
                if 0 <= self.col_file_id < len(row):
                    file_id = str(row[self.col_file_id]).strip()
                dedupe_key = file_id or "|".join(row)
                if dedupe_key in self.seen_ids:
                    continue
                self.seen_ids.add(dedupe_key)

                exists_token = ""
                if 0 <= self.col_exists < len(row):
                    exists_token = str(row[self.col_exists]).strip().lower()
                is_complete = exists_token in TRUE_TOKENS if exists_token else True
                if not is_complete:
                    continue

                actual_size = 0
                if 0 <= self.col_actual_size < len(row):
                    actual_size = _parse_int(row[self.col_actual_size]) or 0
                if actual_size <= 0 and 0 <= self.col_expected_size < len(row):
                    actual_size = _parse_int(row[self.col_expected_size]) or 0

                expected_location = ""
                if 0 <= self.col_expected_location < len(row):
                    expected_location = str(row[self.col_expected_location]).strip()

                self.completed_files += 1
                self.completed_bytes += max(0, actual_size)
                if expected_location:
                    self.latest_file_path = expected_location

            self.offset = handle.tell()

        return self.completed_files, self.completed_bytes


def _read_stdout(proc: subprocess.Popen[str], out_queue: queue.Queue[str | None]) -> None:
    try:
        if proc.stdout is None:
            return
        for raw_line in proc.stdout:
            out_queue.put(str(raw_line))
    finally:
        out_queue.put(None)


def _load_plan(plan_path: Path) -> list[PackagePlan]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_packages = payload.get("packages")
    if not isinstance(raw_packages, list):
        raise ValueError(f"Invalid plan format: {plan_path}")

    plans: list[PackagePlan] = []
    for item in raw_packages:
        if not isinstance(item, dict):
            continue
        package_id = str(item.get("package_id") or "").strip()
        cmd_raw = item.get("cmd")
        if not package_id or not isinstance(cmd_raw, list) or not cmd_raw:
            continue
        cmd = [str(part) for part in cmd_raw]
        expected_total = _parse_int(item.get("expected_total"))
        timepoint_label = str(item.get("timepoint_label") or "").strip()
        plans.append(
            PackagePlan(
                package_id=package_id,
                cmd=cmd,
                expected_total=expected_total,
                timepoint_label=timepoint_label,
            )
        )
    return plans


def _format_speed(value_mb_s: float | None) -> str:
    if value_mb_s is None or value_mb_s < 0:
        return "-- MB/s"
    return f"{value_mb_s:0.2f} MB/s"


def _format_finish_time(eta_seconds: float | None) -> str:
    if eta_seconds is None or eta_seconds < 0:
        return "--:--:--"
    total_seconds = int(round(float(eta_seconds)))
    days, rem = divmod(max(0, total_seconds), 24 * 60 * 60)
    hours, rem = divmod(rem, 60 * 60)
    minutes, seconds = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _clean_status_line(text: str, max_len: int = 110) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").replace("\r", " ").replace("\n", " ")).strip()
    if not compact:
        return ""
    # Collapse noisy command echo lines from downloadcmd startup.
    if "downloadcmd" in compact.lower() and "-dp" in compact.lower():
        return "downloadcmd started"
    if len(compact) > max_len:
        return compact[: max_len - 3] + "..."
    return compact


def _shorten_middle(text: str, max_len: int) -> str:
    token = str(text or "")
    if len(token) <= max_len:
        return token
    if max_len <= 8:
        return token[:max_len]
    head = max_len // 2 - 2
    tail = max_len - head - 3
    return f"{token[:head]}...{token[-tail:]}"


def _resolved_ncols(requested_ncols: int | None) -> int:
    if requested_ncols and requested_ncols > 0:
        return int(requested_ncols)
    cols = shutil.get_terminal_size(fallback=(120, 24)).columns
    return max(40, int(cols))


def run_plan(
    plan_path: Path,
    poll_interval: float,
    stop_on_error: bool,
    *,
    ncols: int | None,
) -> int:
    plans = _load_plan(plan_path)
    if not plans:
        print(f"No package commands found in plan: {plan_path}")
        return 2

    use_dynamic_ncols = not (ncols and int(ncols) > 0)
    resolved_ncols = None if use_dynamic_ncols else int(ncols)
    base_width = _resolved_ncols(resolved_ncols)
    status_width = max(44, base_width - 6)

    package_bar = tqdm(
        total=len(plans),
        desc="Packages",
        position=0,
        unit="pkg",
        dynamic_ncols=use_dynamic_ncols,
        ncols=resolved_ncols,
        ascii=False,
        mininterval=max(0.5, poll_interval),
        bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}% {postfix}",
    )
    status_bars = [
        tqdm(
            total=0,
            desc="",
            bar_format="{desc}",
            position=2 + row_idx,
            dynamic_ncols=use_dynamic_ncols,
            ncols=resolved_ncols,
            mininterval=max(0.5, poll_interval),
            ascii=False,
        )
        for row_idx in range(3)
    ]
    status_rendered = [""] * 3

    def set_status_lines(lines: list[str]) -> None:
        nonlocal status_rendered
        width_now = _resolved_ncols(None if use_dynamic_ncols else resolved_ncols)
        status_width_now = max(44, width_now - 6)
        for row_idx in range(3):
            line = lines[row_idx] if row_idx < len(lines) else ""
            normalized = _clean_status_line(line, max_len=status_width_now).ljust(status_width_now)
            if normalized != status_rendered[row_idx]:
                status_bars[row_idx].set_description_str(normalized, refresh=False)
                status_bars[row_idx].refresh()
                status_rendered[row_idx] = normalized

    set_status_lines(["latest file: --", "event: idle", "recent: --"])

    overall_rc = 0
    try:
        for plan in plans:
            tp = plan.timepoint_label.strip() if plan.timepoint_label else ""
            package_text = tp or f"Package {plan.package_id}"
            package_bar.set_postfix_str(_shorten_middle(package_text, max(12, status_width // 2)))
            package_bar.refresh()

            baseline_reports = set(_list_progress_reports(plan.package_id))
            tracker = ReportTracker()

            expected_total = plan.expected_total if (plan.expected_total or 0) > 0 else None
            current_bar = tqdm(
                total=expected_total,
                desc="Files",
                position=1,
                unit="file",
                dynamic_ncols=use_dynamic_ncols,
                ncols=resolved_ncols,
                ascii=False,
                leave=False,
                mininterval=max(0.5, poll_interval),
                bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}% {postfix}",
            )

            stdout_queue: queue.Queue[str | None] = queue.Queue()
            proc = subprocess.Popen(
                plan.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            reader = threading.Thread(target=_read_stdout, args=(proc, stdout_queue), daemon=True)
            reader.start()

            seen_stdout_end = False
            last_completed_bytes = 0
            last_completed_files = 0
            last_speed_t = time.monotonic()
            last_render_t = 0.0
            current_speed_mb_s: float | None = None
            speed_initialized = False
            files_per_second: float | None = None
            eta_finish_text = "--:--:--"
            speed_history: deque[float] = deque(maxlen=50)
            event_history: deque[str] = deque(maxlen=50)
            render_interval = max(0.75, poll_interval)
            baseline_initialized = False
            latest_file_name = "--"
            last_reported_path = ""

            while True:
                drained = False
                while True:
                    try:
                        line = stdout_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    if line is None:
                        seen_stdout_end = True
                        continue

                    text = line.strip()
                    if not text:
                        continue

                    expected_match = EXPECTED_TOTAL_RE.search(text)
                    if expected_match:
                        parsed = _parse_int(expected_match.group(1))
                        if parsed and parsed > 0:
                            expected_total = parsed
                            current_bar.total = expected_total

                    queued_match = QUEUED_PROGRESS_RE.search(text)
                    if queued_match:
                        done = _parse_int(queued_match.group(1)) or 0
                        total = _parse_int(queued_match.group(2))
                        if total and total > 0 and (current_bar.total is None or current_bar.total <= 0):
                            current_bar.total = total
                        if done >= 0:
                            current_bar.n = max(current_bar.n, done)

                    speed_match = SPEED_RE.search(text)
                    if speed_match:
                        parsed_speed = float(speed_match.group(1))
                        speed_history.append(parsed_speed)
                        current_speed_mb_s = parsed_speed
                        speed_initialized = True
                    elif "error" in text.lower() or "failed" in text.lower():
                        cleaned = _clean_status_line(text)
                        if cleaned:
                            event_history.append(cleaned)
                    elif "beginning download" in text.lower():
                        cleaned = _clean_status_line(text)
                        if cleaned:
                            event_history.append(cleaned)

                active_report = _choose_active_report(plan.package_id, baseline_reports)
                tracker.set_path(active_report)
                completed_files, completed_bytes = tracker.poll()
                if tracker.latest_file_path and tracker.latest_file_path != last_reported_path:
                    last_reported_path = tracker.latest_file_path
                    latest_file_name = Path(last_reported_path).name or tracker.latest_file_path
                    event_history.append(f"downloaded: {latest_file_name}")

                now_t = time.monotonic()
                dt = now_t - last_speed_t
                if dt >= max(0.2, poll_interval):
                    if not baseline_initialized:
                        # Avoid huge fake speed bursts when resuming from an existing progress report.
                        last_completed_bytes = completed_bytes
                        last_completed_files = completed_files
                        baseline_initialized = True
                    else:
                        delta_bytes = completed_bytes - last_completed_bytes
                        delta_files = completed_files - last_completed_files
                        if delta_bytes > 0 and delta_files > 0:
                            speed = delta_bytes / max(dt, 1e-6) / (1024 * 1024)
                            current_speed_mb_s = speed
                            speed_history.append(speed)
                            speed_initialized = True
                            files_per_second = delta_files / max(dt, 1e-6)
                        elif delta_files > 0:
                            files_per_second = delta_files / max(dt, 1e-6)
                    last_completed_bytes = completed_bytes
                    last_completed_files = completed_files
                    last_speed_t = now_t

                if completed_files >= 0:
                    current_bar.n = max(current_bar.n, completed_files)

                files_done_now = max(current_bar.n, completed_files)
                total_now = current_bar.total
                if total_now is not None and files_per_second and files_per_second > 0:
                    remaining_files = max(0.0, float(total_now) - float(files_done_now))
                    eta_seconds = remaining_files / files_per_second
                    eta_finish_text = _format_finish_time(eta_seconds)
                else:
                    eta_finish_text = "--:--:--"

                done = proc.poll() is not None and seen_stdout_end and stdout_queue.empty()
                should_render = done or (now_t - last_render_t >= render_interval)
                if should_render:
                    speed_text = _format_speed(current_speed_mb_s) if speed_initialized else "-- MB/s"
                    current_bar.set_postfix_str(f"{speed_text} | ETA {eta_finish_text}")
                    current_bar.refresh()

                    latest_event = event_history[-1] if event_history else "waiting for progress..."
                    recent = " | ".join(list(event_history)[-3:]) if event_history else "--"
                    set_status_lines(
                        [
                            f"latest file: {_shorten_middle(latest_file_name, max(16, status_width - 13))}",
                            f"event: {_shorten_middle(latest_event, max(16, status_width - 7))}",
                            f"recent: {_shorten_middle(recent, max(16, status_width - 8))}",
                        ]
                    )
                    last_render_t = now_t

                if done:
                    break
                if not drained:
                    time.sleep(max(0.1, poll_interval))

            rc = proc.wait()
            current_bar.n = max(current_bar.n, current_bar.total or current_bar.n)
            current_bar.refresh()
            current_bar.close()

            package_bar.update(1)

            if rc != 0:
                overall_rc = rc
                set_status_lines(
                    [
                        f"latest file: {_shorten_middle(latest_file_name, max(16, status_width - 13))}",
                        f"event: package {plan.package_id} failed (exit {rc})",
                        "recent: see command output above for full error details",
                    ]
                )
                if stop_on_error:
                    break

        if overall_rc == 0 and int(package_bar.n) < int(package_bar.total or 0):
            package_bar.n = int(package_bar.total or package_bar.n)
            package_bar.refresh()
        if overall_rc == 0:
            package_bar.set_postfix_str("done")
            package_bar.refresh()
    finally:
        package_bar.close()
        for bar in status_bars:
            bar.close()

    return overall_rc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NDA downloadcmd plan with tqdm progress.")
    parser.add_argument("--plan", type=Path, required=True, help="Path to JSON plan generated by notebook.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Report polling interval in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--no-stop-on-error",
        action="store_true",
        help="Continue with remaining packages if a package command fails.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=0,
        help="Fixed terminal width for tqdm rendering; use 0 for autoscale (default: 0).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_plan(
        plan_path=args.plan.expanduser().resolve(),
        poll_interval=max(0.1, float(args.poll_interval)),
        stop_on_error=not bool(args.no_stop_on_error),
        ncols=int(args.ncols) if int(args.ncols) > 0 else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
