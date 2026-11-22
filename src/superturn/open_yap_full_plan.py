"""Select a speaker-broad Open Yap subset from the authorized delivery's inventory."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
from tempfile import NamedTemporaryFile

from .splits import speaker_groups, stratified_duration_groups
from .sources import budget_bytes, data_root

SOURCE = "open-yap-full"
CALLS_PER_SPEAKER = 2


def select(index, public_calls, allowance):
    """Maximize distinct voices in a slice-first ten-hour review."""
    import numpy as np
    from scipy import __version__ as solver_version
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    if type(allowance) is not int or allowance < 1:
        raise ValueError("there is no remaining acquisition budget")
    public_speakers = {voice for people in public_calls.values() for voice in people}
    if len(public_calls) != 16 or len(public_speakers) != 8:
        raise ValueError("the pinned public sample must resolve to sixteen calls and eight speakers")
    calls, options = {}, {}
    for row in index:
        call = row["conversation_id"]
        people = row["speaker_ids"]
        if (not isinstance(call, str) or not call or call in calls
                or not isinstance(people, list) or len(people) != 2
                or any(not isinstance(voice, str) or not voice for voice in people)
                or people[0] == people[1]):
            raise ValueError("the inventory must contain unique calls and ordered, paired speaker IDs")
        calls[call] = tuple(people)
        for mode in ("whole", "slice"):
            if mode not in row:
                continue
            option = row[mode]
            size, duration = option["transfer_bytes"], option["duration_seconds"]
            if (type(size) is not int or size <= 0 or type(duration) is not int or duration <= 0):
                raise ValueError("the inventory requires exact positive transfer costs and durations")
            start = option.get("start_second", 0)
            if mode == "slice" and (not 180 <= duration <= 240 or type(start) is not int or start < 0):
                raise ValueError("the slice needs a verified common start and a duration of three to four minutes")
            if mode == "whole" and (start != 0 or duration < 600):
                continue
            if size <= allowance and not public_speakers.intersection(people):
                options[call, mode] = (size, duration, start)
    if any(tuple(voices) != calls.get(call) for call, voices in public_calls.items()):
        raise ValueError("the public and delivered per-track identities do not match")
    options = dict(sorted(options.items(), key=lambda item: hashlib.sha256(
        f"open-yap-review-v1:{item[0][0]}:{item[0][1]}".encode()).digest()))
    if sum(mode == "whole" for _, mode in options) < 4:
        raise ValueError("four affordable whole calls outside the public roster are required")
    n_options = len(options)
    available = sorted({call for call, _ in options})
    people = sorted({voice for call in available for voice in calls[call]})
    pairs = sorted({tuple(sorted(calls[call])) for call, mode in options if mode == "whole"})
    call_index = {call: index for index, call in enumerate(available)}
    voice_index = {voice: index for index, voice in enumerate(people)}
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    voice_offset = 3 + len(available)
    pair_offset = voice_offset + 2 * len(people)
    variables = n_options + len(people)
    matrix = lil_matrix((pair_offset + len(pairs), variables), dtype=float)
    upper = np.ones(matrix.shape[0], dtype=float)
    lower = np.full(matrix.shape[0], -np.inf)
    lower[2], upper[2] = 4, 8
    upper[voice_offset:pair_offset:2] = CALLS_PER_SPEAKER
    upper[voice_offset + 1:pair_offset:2] = 0
    costs = np.full(variables, -1.0)
    for j, ((call, mode), (size, duration, _)) in enumerate(options.items()):
        matrix[0, j] = size / allowance
        matrix[1, j] = duration / 36000
        matrix[3 + call_index[call], j] = 1
        costs[j] = .25 * size / allowance
        if mode == "whole":
            matrix[2, j] = 1
            matrix[pair_offset + pair_index[tuple(sorted(calls[call]))], j] = 1
        for voice in calls[call]:
            i = voice_index[voice]
            matrix[voice_offset + 2 * i, j] = 1
            matrix[voice_offset + 2 * i + 1, j] = -1
    for i in range(len(people)):
        matrix[voice_offset + 2 * i + 1, n_options + i] = 1
    solution = milp(costs, integrality=np.ones(variables), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix.tocsc(), lower, upper),
                    options={"time_limit": 60, "mip_rel_gap": 0})
    if solution.x is None or solution.status not in (0, 1):
        raise ValueError("the solver did not produce a feasible slice-first selection")
    chosen = {key: options[key] for key, value in zip(options, solution.x[:n_options], strict=True) if value > .5}
    used = sum(size for size, _, _ in chosen.values())
    seconds = sum(duration for _, duration, _ in chosen.values())
    appearances = Counter(voice for call, _ in chosen for voice in calls[call])
    mode_counts = Counter(mode for _, mode in chosen)
    chosen_calls = {call for call, _ in chosen}
    whole_pairs = {tuple(sorted(calls[call])) for call, mode in chosen if mode == "whole"}
    if (not appearances or used > allowance or seconds > 36000 or max(appearances.values()) > CALLS_PER_SPEAKER
            or len(chosen_calls) != len(chosen) or not 4 <= mode_counts["whole"] <= 8
            or len(whole_pairs) != mode_counts["whole"] or public_speakers.intersection(appearances)):
        raise ValueError("the computed integer selection violates the exact acquisition limits")
    components = speaker_groups((call, calls[call]) for call in sorted(chosen_calls))
    split, _ = stratified_duration_groups(SOURCE, "train-conversation",
                                          ((components[call], "en", None) for call in sorted(chosen_calls)))
    partitions = {call: split[components[call]] for call in chosen_calls}
    voice_split = {}
    for call in sorted(chosen_calls):
        for voice in calls[call]:
            if voice_split.setdefault(voice, partitions[call]) != partitions[call]:
                raise ValueError("a speaker crosses the planned splits")
    return {"algorithm": "open-yap-slice-review-v1", "byte_allowance": allowance, "selected_bytes": used,
            "selected_seconds": seconds, "max_calls_per_speaker": CALLS_PER_SPEAKER,
            "solver_version": solver_version, "solver_proved_optimal": solution.status == 0,
            "solver_relative_gap": float(solution.mip_gap) if solution.mip_gap is not None else None,
            "public_sample_speakers_excluded": len(public_speakers), "eligible_calls": len(available),
            "selected_speakers": len(appearances), "selected_by_mode": dict(mode_counts),
            "speakers_by_split": dict(Counter(voice_split.values())),
            "components_by_split": dict(Counter(split.values())),
            "call_counts_per_speaker": dict(Counter(appearances.values())), "speaker_split": voice_split,
            "selected": [{"conversation_id": call, "speaker_ids": list(calls[call]),
                          "capture": mode, "start_second": start, "duration_seconds": duration,
                          "transfer_bytes": size, "split": partitions[call]}
                         for (call, mode), (size, duration, start) in sorted(chosen.items())]}


def protected_scratch(directory):
    """Private mode-0700 tmpfs in the current no-swap workload."""
    directory = directory.resolve()
    fs = subprocess.check_output(["findmnt", "--noheadings", "--output", "FSTYPE", "--target", directory],
                                 text=True).strip()
    swap_limit = Path("/sys/fs/cgroup/memory.swap.max")
    swap_usage = swap_limit.with_name("memory.swap.current")
    swap_disabled = (swap_limit.is_file() and swap_usage.is_file()
                     and swap_limit.read_text().strip() == swap_usage.read_text().strip() == "0")
    if (fs != "tmpfs" or directory.stat().st_mode & 0o077 or directory.stat().st_uid != os.getuid()
            or not swap_disabled):
        raise ValueError("use a private tmpfs directory in a verified no-swap workload")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return directory


def run(root, inventory, output, acquired_bytes, extra_reserve):
    """Run only inside approved confidential scratch; persistent copies must be encrypted."""
    if not (root / ".campaign-machine").is_file():
        raise ValueError("the private inventory stays on Vast")
    scratch = protected_scratch(inventory.resolve().parent)
    if output.resolve().parent != scratch:
        raise ValueError("work files must be inside the same private scratch directory")
    if inventory.resolve() == output.resolve():
        raise ValueError("inventory and selection must be separate files")
    previous = root / "runs/open-yap-preflight/metadata.json"
    preflight = json.loads((root / "reports/open-yap-preflight.json").read_text())
    if hashlib.sha256(previous.read_bytes()).hexdigest() != preflight["index_sha256"]:
        raise ValueError("the pinned public sample index changed")
    public_calls = {call: tuple(row["people"]) for call, row in json.loads(previous.read_text())["records"].items()}
    payload = inventory.read_bytes()
    used = sum(path.stat().st_size for path in (root / "raw").rglob("*") if path.is_file())
    if min(acquired_bytes, extra_reserve) < 0:
        raise ValueError("reserve must be nonnegative")
    allowance = min(5_000_000_000, budget_bytes() - max(used, acquired_bytes) - len(payload) - extra_reserve)
    report = select((json.loads(line) for line in payload.splitlines() if line.strip()), public_calls, allowance)
    report.update(inventory_sha256=hashlib.sha256(payload).hexdigest(), reserved_transfer_bytes=extra_reserve,
                  previous_acquisition_bytes=max(used, acquired_bytes))
    encoded = (json.dumps(report, indent=2) + "\n").encode()
    with NamedTemporaryFile(dir=output.parent) as temp:
        temp.write(encoded)
        temp.flush()
        try:
            os.link(temp.name, output)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise ValueError("the acquisition plan already exists and differs")
    return {key: value for key, value in report.items() if key not in {"selected", "speaker_split"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reserve-bytes", type=int, required=True, help="bound for uncounted metadata and overhead")
    parser.add_argument("--acquired-bytes", type=int, required=True, help="cumulative bytes from the acquisition ledger")
    args = parser.parse_args()
    print(json.dumps(run(data_root(), args.inventory, args.output, args.acquired_bytes, args.reserve_bytes), indent=2))
