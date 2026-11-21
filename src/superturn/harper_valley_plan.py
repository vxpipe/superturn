"""Plan a speaker-disjoint HarperValleyBank pilot from public metadata only."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import random

from .acquire import _request
from .ami_calibration import _write_json
from .progress import Progress
from .sources import data_root

REPOSITORY = "cricketclub/gridspace-stanford-harper-valley"
REVISION = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
TREE_SHA = "c0ed22efd81bcce435d7e0374bf747af8672e343"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}"
SPLITS = ("train", "validation", "test")
BUDGET = 1_000_000_000
CALLS_PER_SPEAKER = 20


@dataclass(frozen=True)
class Call:
    identifier: str
    caller: int
    agent: int
    size: int


def calls_within_budget(rows, budget):
    """Prefer underrepresented voices, using shorter calls when costs differ."""
    counts = Counter()
    chosen, used = [], 0
    candidates = sorted(rows, key=lambda row: (row.size, row.identifier))
    while candidates:
        candidates = [row for row in candidates
                      if row.size + used <= budget
                      and max(counts[row.caller], counts[row.agent]) < CALLS_PER_SPEAKER]
        if not candidates:
            break
        row = min(candidates, key=lambda item: (
            counts[item.caller] + counts[item.agent], item.size, item.identifier))
        candidates.remove(row)
        chosen.append(row)
        counts.update({row.caller, row.agent})
        used += row.size
    return chosen


def plan(rows, budget, *, attempts=2048):
    speakers = sorted({speaker for row in rows for speaker in (row.agent, row.caller)})
    if len(speakers) < 9:
        raise ValueError("at least nine speaker identities are required for this three-way split")
    held_out = max(3, round(len(speakers) * 0.17))
    rng = random.Random(48204)
    finalists = []
    for _ in range(attempts):
        ordered = rng.sample(speakers, len(speakers))
        partitions = (set(ordered[2 * held_out:]), set(ordered[:held_out]), set(ordered[held_out:2 * held_out]))
        groups = [[row for row in rows if row.caller in voices and row.agent in voices] for voices in partitions]
        counts = tuple(len(group) for group in groups)
        coverage = tuple(min(len({row.agent for row in group}), len({row.caller for row in group})) for group in groups)
        # Preserve both roles in each held-out split while keeping the fit pool sizable.
        rank = (min(coverage[1:]), min(counts[1:]), counts[0])
        finalists.append((rank, groups))
    shortlist = sorted(finalists, key=lambda candidate: candidate[0], reverse=True)[:32]
    holdout_budget = budget * 15 // 100
    limits = (budget - 2 * holdout_budget, holdout_budget, holdout_budget)
    selections = [[calls_within_budget(group, cap) for group, cap in zip(groups, limits, strict=True)]
                  for _, groups in shortlist]

    def rank(selection):
        coverage = [len({voice for row in rows for voice in (row.caller, row.agent)}) for rows in selection]
        return min(coverage[1:]), coverage[0], min(map(len, selection[1:])), sum(map(len, selection))

    return max(selections, key=rank)


def run(root):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("HarperValleyBank metadata planning uses the campaign machine")
    url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{TREE_SHA}?recursive=1"
    with _request(url) as response:
        tree = json.load(response)
    if tree["truncated"] or tree["sha"] != TREE_SHA:
        raise ValueError("could not verify complete HarperValleyBank tree")
    sizes = {item["path"]: item["size"] for item in tree["tree"] if item["type"] == "blob"}
    metadata = sorted(path for path in sizes if path.startswith("data/metadata/") and path.endswith(".json"))
    if len(metadata) != 1446:
        raise ValueError("HarperValleyBank metadata count changed")
    cache = root / "cache" / "harper-valley" / f"{REVISION}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)

    def read_call(path):
        with _request(f"{RAW}/{path}") as response:
            payload = response.read(sizes[path] + 1)
        if len(payload) != sizes[path]:
            raise ValueError("HarperValleyBank metadata size changed")
        record = json.loads(payload)
        sid = path.rsplit("/", 1)[1].removesuffix(".json")
        if record["sid"] != sid:
            raise ValueError("HarperValleyBank conversation identifier mismatch")
        return sid, int(record["caller"]["speaker_id"]), int(record["agent"]["speaker_id"])

    if cache.is_file():
        identities = json.loads(cache.read_text())
    else:
        identities = []
        meter = Progress("harper-valley-metadata", len(metadata), "calls")
        try:
            with ThreadPoolExecutor(max_workers=12) as pool:
                for row in pool.map(read_call, metadata):
                    identities.append(row)
                    meter.advance(1)
            _write_json(cache, identities)
        except BaseException:
            meter.finish("failed")
            raise
        else:
            meter.finish()

    def files(sid):
        return [f"data/audio/{role}/{sid}.wav" for role in ("caller", "agent")] + [f"data/transcript/{sid}.json"]

    if sorted(f"data/metadata/{sid}.json" for sid, _, _ in identities) != metadata:
        raise ValueError("HarperValleyBank speaker metadata does not match the provider inventory")
    rows = [Call(sid, caller, agent, sum(sizes[path] for path in files(sid))) for sid, caller, agent in identities]
    metadata_bytes = sum(sizes[path] for path in metadata)
    selection = dict(zip(SPLITS, plan(rows, BUDGET - metadata_bytes), strict=True))
    assignments = {split: {voice for row in cases for voice in (row.agent, row.caller)}
                   for split, cases in selection.items()}
    if sum(map(len, assignments.values())) != len(set.union(*assignments.values())):
        raise ValueError("speaker overlap in HarperValleyBank selection")
    splits = {split: {"calls": len(cases), "speakers": len(assignments[split]),
                      "caller_speakers": len({row.caller for row in cases}),
                      "agent_speakers": len({row.agent for row in cases}), "bytes": sum(row.size for row in cases)}
              for split, cases in selection.items()}
    selected_bytes = metadata_bytes + sum(split["bytes"] for split in splits.values())
    if selected_bytes > BUDGET or not all(split["calls"] for split in splits.values()):
        raise ValueError("HarperValleyBank plan exceeds its cap or is missing a split")
    manifest = cache.with_name(f"{REVISION}-selection.json")
    _write_json(manifest, {
        split: [{"id": row.identifier, "caller": row.caller, "agent": row.agent,
                 "files": [{"path": path, "bytes": sizes[path]} for path in files(row.identifier)]}
                for row in cases] for split, cases in selection.items()
    })
    report = root / "reports" / "harper-valley-plan.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report, {
        "revision": REVISION, "corpus_calls": len(rows),
        "corpus_speakers": len({voice for row in rows for voice in (row.agent, row.caller)}),
        "metadata_bytes": metadata_bytes, "projected_bytes": selected_bytes,
        "cap_per_speaker": CALLS_PER_SPEAKER, "splits": splits, "audio_bytes_downloaded": 0,
        "selection_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    })
    return report


if __name__ == "__main__":
    print(run(data_root()))
