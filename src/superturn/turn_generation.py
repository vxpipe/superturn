"""Generate and validate quota-balanced synthetic turn-completion dialogues."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from fcntl import LOCK_EX, LOCK_NB, flock
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .ami_calibration import _write_json
from .progress import Progress

CATALOGUE = Path("configs/turn-generation-catalogue.json")
GENERATOR = "gemini-3.8-flash"
QUOTA = 2
BATCH = 6
RETRIES = 4
GEN_RESERVE_USD = .038
OUTPUT_LIMIT = 3400
CLEANER_VERSION = "turn-text-nfkc-v2"
GEN_INPUT, GEN_OUTPUT = ((.75, 3.75) if date.today() <= date(2026, 12, 31) else (1.50, 7.50))
FIELDS = ("previous_agent", "caller_before_pause", "caller_after_pause", "next_agent",
          "short_caller_reply", "final_agent_reply")
SYSTEM = """Generate original, fictional telephone-style spoken conversations, not formal written
language. For each case write one two-person service exchange. Each JSON dialogue has a
completed agent turn, a caller's utterance split at a plausible 300–800 ms mid-thought silence,
and a substantive agent reply after the caller finishes. The two caller fields join smoothly
as one turn. The mid-turn hold and the full caller turn will form a matched pair. Never insert
overlap, a barge-in, a listener backchannel, silence markers, cues, names of actual people,
contact information, policy advice or stage directions. Use natural colloquial words, mild
disfluencies where appropriate and varied clause lengths. Use everyday borrowed words as
native speakers say them, written in their own language's script. Do not romanize non-Latin
languages or switch languages. For caller-first slots put an empty string in previous_agent.
For short-shift slots, after next_agent write a brief substantive caller reply to that turn
and a final substantive agent reply. Otherwise leave both short_caller_reply and
final_agent_reply empty.
Output just the requested JSON and preserve each requested slot."""
SCHEMA = {
    "type": "array", "items": {
        "type": "object",
        "properties": {"slot": {"type": "string"}, **{field: {"type": "string"} for field in FIELDS}},
        "required": ["slot", *FIELDS], "additionalProperties": False,
    },
}
SCRIPT_NAMES = {
    "Latn": ("LATIN",), "Deva": ("DEVANAGARI",), "Knda": ("KANNADA",),
    "Taml": ("TAMIL",), "Telu": ("TELUGU",), "Mlym": ("MALAYALAM",),
    "Hans": ("CJK UNIFIED IDEOGRAPH",), "Jpan": ("HIRAGANA", "KATAKANA", "CJK UNIFIED IDEOGRAPH"),
}


def make_plan(catalogue):
    plan = []
    scenarios = [(domain, scenario) for domain in catalogue["domains"] for scenario in domain["scenarios"]]
    for language in catalogue["languages"]:
        variants = language["regional_variants"] or [language["code"]]
        for index, (domain, scenario) in enumerate(scenarios):
            criterion = f'{language["code"]}/{scenario["id"]}'
            for sample in range(QUOTA):
                plan.append({
                    "slot": f"{criterion}/{sample + 1}", "criterion": criterion,
                    "language": language["code"], "language_name": language["name"],
                    "native_script": language["scripts"][0],
                    "variant": variants[(index + sample) % len(variants)],
                    "domain": domain["id"], "scenario": scenario["id"],
                    "situation": scenario["description"], "notes": domain.get("prompt_notes", ""),
                    "caller_first": (index + sample) % 7 == 0,
                    "short_shift": sample == 1,
                    "pause_style": (("complete-sounding answer with an added detail",
                                     "long caller turn with multiple natural pause opportunities",
                                     "short complete-sounding answer before a qualifier")[index % 3]
                                    if sample == 0 else
                                    ("planning before a name or time", "a self-repair after several words",
                                     "a short pause during a list")[index % 3]),
                })
    return plan


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(?<=\w)['’](?=\w)", "", text)
    return " ".join("".join(
        char if char in "\u200c\u200d" or unicodedata.category(char)[0] not in "PC" else " "
        for char in text
    ).split())


def clean_sample(row, spec):
    result = {}
    native = SCRIPT_NAMES[spec["native_script"]]
    for field in FIELDS:
        raw = row.get(field)
        if not isinstance(raw, str) or len(raw) > 360 or re.search(r"[@\[\]<>\n]|(?:\d[ .-]*){6}", raw):
            return None
        text = normalize(raw)
        optional_blank = (field == "previous_agent" and spec["caller_first"] or
                          field in {"short_caller_reply", "final_agent_reply"} and not spec["short_shift"])
        if optional_blank:
            if text:
                return None
            result[field] = ""
            continue
        letters = [unicodedata.name(char, "") for char in text if char.isalpha()]
        if not letters or sum(name.startswith(native) for name in letters) < .8 * len(letters):
            return None
        result[field] = text
    if result["caller_before_pause"] == result["caller_after_pause"]:
        return None
    short = result["short_caller_reply"]
    if len(short) > 65 or len(short.split()) > 9:
        return None
    return result


class ApiError(RuntimeError):
    pass


def request_json(path, key, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = Request("https://generativelanguage.googleapis.com/v1beta/" + path, data=data,
                  headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=90) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 or attempt == 2:
                raise ApiError(f"HTTP {error.code} from {path}") from None
            retry_after = error.headers.get("Retry-After", "")
            time.sleep(min(60, int(retry_after)) if retry_after.isdigit() else 2 ** (attempt + 2))
        except (URLError, TimeoutError):
            raise ApiError(f"network timeout or connection failure from {path}") from None


def generate_batch(specs, key, attempt):
    prompt = json.dumps({"attempt": attempt, "cases": specs}, ensure_ascii=False)
    response = request_json(f"models/{GENERATOR}:generateContent", key, {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": SCHEMA,
                             "maxOutputTokens": OUTPUT_LIMIT, "thinkingConfig": {"thinkingLevel": "LOW"}},
    })
    usage = response.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount")
    output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    spend = GEN_RESERVE_USD if input_tokens is None else (input_tokens * GEN_INPUT + output_tokens * GEN_OUTPUT) / 1_000_000
    candidates = response.get("candidates", [])
    if candidates and candidates[0].get("finishReason") == "MAX_TOKENS":
        print(f"output limit reached for {specs[0]['language']} batch of {len(specs)}", flush=True)
    text = "".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])
                   if not part.get("thought")) if candidates else ""
    try:
        generated = json.loads(text)
    except ValueError:
        return [], spend
    if not isinstance(generated, list) or any(not isinstance(row, dict) for row in generated):
        return [], spend
    by_slot = {row.get("slot"): row for row in generated if isinstance(row.get("slot"), str)}
    if len(by_slot) != len(generated):
        return [], spend
    checked = []
    for spec in specs:
        if spec["slot"] not in by_slot:
            continue
        cleaned = clean_sample(by_slot[spec["slot"]], spec)
        if cleaned is not None:
            checked.append({**spec, **cleaned, "generator_version": response.get("modelVersion")})
    return checked, spend


def sample_key(row):
    return row["language"], row["previous_agent"], row["caller_before_pause"], row["caller_after_pause"]


def paired_examples(row):
    """Keep observational evidence outside the two model inputs."""
    hold = {"slot": row["slot"], "language": row["language"], "label": "hold",
            "previous_turn": row["previous_agent"], "current_turn": row["caller_before_pause"]}
    joiner = "" if row["language"] in {"zh", "ja"} else " "
    # Use the completed opening for first-turn shifts so the formatting
    # marker appears equally often in holds and shifts.
    short = row["short_caller_reply"] if row["previous_agent"] else ""
    shift = {**hold, "label": "shift",
             "previous_turn": row["next_agent"] if short else row["previous_agent"],
             "current_turn": short or joiner.join((row["caller_before_pause"], row["caller_after_pause"]))}
    return hold, shift


def plan_checksum(plan):
    protocol = [plan, SYSTEM, SCHEMA, GENERATOR, CLEANER_VERSION]
    return sha256(json.dumps(protocol).encode()).hexdigest()


def smoke_plan(plan):
    choices = {}
    for spec in plan:
        choices.setdefault(spec["native_script"], []).append(spec)
    return [
        next(spec for spec in options if spec["caller_first"] == (index % 4 < 2)
             and spec["short_shift"] == (index % 2 == 1) and spec["domain"] != "banking")
        for index, options in enumerate(choices.values())
    ]


def similar_sample(row, prior):
    def bigrams(candidate):
        text = " ".join(sample_key(candidate)[1:])
        return set(zip(text, text[1:]))

    left, right = bigrams(row), bigrams(prior)
    return bool(left | right) and len(left & right) / len(left | right) >= .9


def schedule_batches(pending, attempts, available):
    languages = defaultdict(list)
    for spec in pending:
        languages[spec["language"]].append(spec)
    attempted = Counter()
    for slot, count in attempts.items():
        attempted[slot.split("/", 1)[0]] += count
    groups = [sorted(languages[language], key=lambda spec: attempts.get(spec["slot"], 0))
              for language in sorted(languages, key=lambda language: (attempted[language], language))]
    batches = []
    size = [3 if group[0]["native_script"] in {"Knda", "Taml", "Telu", "Mlym"} else BATCH for group in groups]
    for batch_index in range(max((len(group) + n - 1) // n for group, n in zip(groups, size, strict=True))):
        for group, n in zip(groups, size, strict=True):
            if batch := group[batch_index * n:(batch_index + 1) * n]:
                batches.append(batch)
                if len(batches) == available:
                    return batches
    return batches


def initial_state(plan, smoke_source):
    state = {"plan_sha256": plan_checksum(plan), "spend_usd": 0., "attempts": {}, "accepted": {}}
    if smoke_source is None:
        return state
    smoke = json.loads((smoke_source / "state.json").read_text())
    receipt = json.loads((smoke_source / "receipt.json").read_text())
    corpus = (smoke_source / "dialogues.jsonl").read_bytes()
    eligible = {row["slot"]: row for row in smoke_plan(plan)}
    rows = {row["slot"]: row for row in map(json.loads, corpus.splitlines())}
    if (receipt["dialogues_sha256"] != sha256(corpus).hexdigest()
            or receipt["generator_prompt_sha256"] != sha256(SYSTEM.encode()).hexdigest()
            or receipt["generator"] != GENERATOR or receipt["cleaner_version"] != CLEANER_VERSION
            or receipt["accepted_dialogues"] != len(rows) or smoke["accepted"] != rows):
        raise ValueError("the smoke rows or receipt differ from the compatible generation protocol")
    for slot, row in rows.items():
        spec = eligible.get(slot)
        cleaned = clean_sample(row, spec) if spec is not None else None
        if (spec is None or any(row.get(key) != value for key, value in spec.items())
                or cleaned is None):
            raise ValueError("a smoke dialogue does not match its approved slot")
        state["accepted"][slot] = {**spec, **cleaned, "generator_version": row.get("generator_version")}
        state["attempts"][slot] = smoke["attempts"][slot]
    state["spend_usd"] = smoke["spend_usd"]
    state["seeded_from_smoke"] = len(state["accepted"])
    return state


def run(plan, directory, budget, workers, key, *, smoke_source=None, attempts_per_slot=RETRIES):
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    digest = plan_checksum(plan)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != digest:
            raise ValueError("this output directory belongs to a different generation plan")
    else:
        state = initial_state(plan, smoke_source)
    accepted, attempts = state["accepted"], state["attempts"]
    seen = {sample_key(row) for row in accepted.values()}
    by_criterion = defaultdict(list)
    for row in accepted.values():
        by_criterion[row["criterion"]].append(row)
    meter = Progress("turn-generation-accepted", len(plan), "dialogues", path=directory / "progress.json")
    meter.skip(len(accepted))
    stop = "attempt_limit"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending := [spec for spec in plan if spec["slot"] not in accepted
                          and attempts.get(spec["slot"], 0) < attempts_per_slot]:
            reserve = GEN_RESERVE_USD
            available = min(workers, int((budget - state["spend_usd"] + 1e-9) / reserve))
            if available <= 0:
                stop = "budget_cap"
                break
            batches = schedule_batches(pending, attempts, available)
            for batch in batches:
                for spec in batch:
                    slot = spec["slot"]
                    attempts[slot] = attempts.get(slot, 0) + 1
            state["spend_usd"] += len(batches) * reserve
            _write_json(state_path, state)
            jobs = [pool.submit(generate_batch, batch, key, max(attempts[spec["slot"]] for spec in batch))
                    for batch in batches]
            for job in jobs:
                try:
                    rows, spend = job.result()
                except ApiError as error:
                    print(str(error), flush=True)
                    stop = "provider_error"
                    continue
                state["spend_usd"] += spend - reserve
                if spend > reserve:
                    stop = "reservation_exceeded"
                for row in rows:
                    fingerprint = sample_key(row)
                    related = by_criterion[row["criterion"]]
                    if fingerprint in seen or any(similar_sample(row, prior) for prior in related):
                        state["duplicates_rejected"] = state.get("duplicates_rejected", 0) + 1
                        continue
                    seen.add(fingerprint)
                    related.append(row)
                    accepted[row["slot"]] = row
                    meter.advance(1)
                _write_json(state_path, state)
            if stop in {"provider_error", "reservation_exceeded"}:
                break
    if len(accepted) == len(plan):
        stop = "complete"
    meter.finish(stop)
    ordered = sorted(accepted.values(), key=lambda row: row["slot"])
    examples = [example for row in ordered for example in paired_examples(row)]
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
    temporary = directory / "dialogues.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(directory / "dialogues.jsonl")
    criteria = Counter(row["criterion"] for row in ordered)
    targets = Counter(row["criterion"] for row in plan)
    attempts_by_style = Counter()
    for row in plan:
        attempts_by_style[row["pause_style"]] += attempts.get(row["slot"], 0)
    receipt = {
        "plan_sha256": digest, "dialogues_sha256": sha256(payload.encode()).hexdigest(),
        "generator_prompt_sha256": sha256(SYSTEM.encode()).hexdigest(),
        "cleaner_version": CLEANER_VERSION,
        "list_usd_per_million_tokens": {"generator_input": GEN_INPUT, "generator_output": GEN_OUTPUT},
        "generator": GENERATOR, "target_per_criterion": QUOTA,
        "target_dialogues": len(plan), "accepted_dialogues": len(ordered),
        "derived_example_counts": dict(Counter(example["label"] for example in examples)),
        "accepted_by_language": dict(Counter(row["language"] for row in ordered)),
        "accepted_by_pause_style": dict(Counter(row["pause_style"] for row in ordered)),
        "seeded_from_smoke": state.get("seeded_from_smoke", 0),
        "attempted_by_pause_style": dict(attempts_by_style),
        "duplicates_rejected": state.get("duplicates_rejected", 0),
        "complete_criteria": sum(criteria[criterion] == count for criterion, count in targets.items()),
        "target_criteria": len(targets),
        "estimated_spend_including_unknown_requests_usd": round(state["spend_usd"], 6),
        "stop": stop,
    }
    _write_json(directory / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="inspect the matrix without an API request")
    parser.add_argument("--check-access", action="store_true", help="read model metadata only")
    parser.add_argument("--smoke", action="store_true", help="run an isolated one-dialogue-per-script probe")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--seed-smoke", type=Path, help="reuse accepted dialogues and recorded spend from a smoke run")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--attempts-per-slot", type=int, default=RETRIES)
    args = parser.parse_args()
    plan = make_plan(json.loads(CATALOGUE.read_text()))
    if args.plan:
        print(json.dumps({"target_per_criterion": QUOTA, "criteria": len(plan) // QUOTA,
                          "dialogues": len(plan), "languages": Counter(row["language"] for row in plan)}, indent=2))
        return
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        parser.error("set GEMINI_API_KEY")
    if args.check_access:
        value = request_json(f"models/{GENERATOR}", key)
        print(value["name"])
        return
    if args.budget_usd is None or not math.isfinite(args.budget_usd) or args.budget_usd <= 0 or not 1 <= args.workers <= 10:
        parser.error("an approved positive --budget-usd and 1–10 workers are required")
    if args.attempts_per_slot <= 0:
        parser.error("--attempts-per-slot must be positive")
    if args.smoke:
        if args.seed_smoke is not None:
            parser.error("--seed-smoke applies to the full plan only")
        plan = smoke_plan(plan)
    if args.directory is None:
        args.directory = Path("data/turn-generation-smoke" if args.smoke else "data/turn-generation-v1")
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory / "run.lock").open("a") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError:
            parser.error("another process is generating into this directory")
        receipt = run(plan, args.directory, args.budget_usd, args.workers, key,
                      smoke_source=args.seed_smoke, attempts_per_slot=args.attempts_per_slot)
    if receipt["stop"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
