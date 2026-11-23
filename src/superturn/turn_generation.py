"""Generate and validate quota-balanced synthetic turn-completion dialogues."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fcntl import LOCK_EX, LOCK_NB, flock
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .ami_calibration import _write_json
from .progress import Progress

CATALOGUE = Path("configs/turn-generation-catalogue.json")
GENERATOR = "gemini-3.8-flash"
JUDGE = "gemini-decision-1"
QUOTA = 2
BATCH = 6
RETRIES = 4
RESERVE_USD = .05
OUTPUT_LIMIT = 3400
CLEANER_VERSION = "turn-text-nfkc-v1"
FIELDS = ("previous_agent", "caller_before_pause", "caller_after_pause", "next_agent")
SYSTEM = """Generate original, fictional telephone-style spoken conversations, not formal written
language. For each case write one two-person service exchange. Each JSON dialogue has a
completed agent turn, a caller's utterance split at a plausible 300–800 ms mid-thought silence,
and a substantive agent reply after the caller finishes. The two caller fields join smoothly
as one turn. The mid-turn hold and the full caller turn will form a matched pair. Never insert
overlap, a barge-in, a listener backchannel, silence markers, cues, names of actual people,
contact information, policy advice or stage directions. Use natural colloquial words, mild
disfluencies where appropriate and varied clause lengths. Write in the native script specified
for the conversation language. Do not transliterate non-Latin languages or add another language.
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
QUESTIONS = {
    "speech": "Is this a natural, informal two-person phone exchange in the requested language?",
    "scenario": "Does this exchange clearly fit its requested scenario?",
    "pause": "Could the caller plausibly pause at the specified point and then continue the same turn? Assess the boundary and the continuation.",
    "floor": "Does the last agent reply take a substantive turn after the complete caller utterance, without interruption or just a bare acknowledgment?",
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
                    "pause_style": ("complete-sounding phrase with an additional detail" if sample == 0
                                    else "a planning pause before a short substantive answer is finished"),
                })
    return plan


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join("".join(char for char in text if unicodedata.category(char)[0] not in "PC").split())


def clean_sample(row, spec):
    result = {}
    native = SCRIPT_NAMES[spec["native_script"]]
    for field in FIELDS:
        raw = row.get(field)
        if not isinstance(raw, str) or len(raw) > 360 or re.search(r"[@\[\]<>\n]|(?:\d[ .-]*){6}", raw):
            return None
        text = normalize(raw)
        letters = [unicodedata.name(char, "") for char in text if char.isalpha()]
        if not letters or sum(name.startswith(native) for name in letters) < .8 * len(letters):
            return None
        result[field] = text
    if result["caller_before_pause"] == result["caller_after_pause"]:
        return None
    return result


class ApiError(RuntimeError):
    pass


def request_json(path, key, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = Request("https://generativelanguage.googleapis.com/v1beta/" + path, data=data,
                  headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=90) as response:
            return json.load(response)
    except HTTPError as error:
        raise ApiError(f"HTTP {error.code} from {path}") from None
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
    spend = RESERVE_USD if input_tokens is None else (input_tokens * .75 + output_tokens * 3.75) / 1_000_000
    candidates = response.get("candidates", [])
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
            checked.append((spec, cleaned))
    if not checked:
        return [], spend
    input_rows = [
        {"record": i, "language": spec["language_name"], "situation": spec["situation"], **sample}
        for i, (spec, sample) in enumerate(checked)
    ]
    questions = {
        f"r{i}_{name}": {"type": "noul", "instructions": f"For record {i} only. {question}"}
        for i in range(len(checked)) for name, question in QUESTIONS.items()
    }
    judged = request_json("decisions", key, {
        "model": JUDGE, "input": json.dumps(input_rows, ensure_ascii=False), "questions": questions,
    })
    judge_tokens = judged.get("usage", {}).get("total_input_tokens")
    spend = RESERVE_USD if judge_tokens is None else spend + judge_tokens * .10 / 1_000_000
    answers = judged.get("answers", {})
    accepted = []
    for i, (spec, sample) in enumerate(checked):
        scores = {name: answers.get(f"r{i}_{name}", {}).get("noul") for name in QUESTIONS}
        if all(isinstance(score, (float, int)) and score >= .65 for score in scores.values()):
            accepted.append({**spec, **sample, "judge": scores, "generator_version": response.get("modelVersion")})
    return accepted, spend


def sample_key(row):
    return row["language"], row["previous_agent"], row["caller_before_pause"], row["caller_after_pause"]


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
    for start in range(0, max(map(len, groups)), BATCH):
        for group in groups:
            if batch := group[start:start + BATCH]:
                batches.append(batch)
                if len(batches) == available:
                    return batches
    return batches


def run(plan, directory, budget, workers, key):
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    digest = sha256(json.dumps([plan, SYSTEM, QUESTIONS, GENERATOR, JUDGE, CLEANER_VERSION]).encode()).hexdigest()
    state = {"plan_sha256": digest, "spend_usd": 0., "attempts": {}, "accepted": {}}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != digest:
            raise ValueError("this output directory belongs to a different generation plan")
    accepted, attempts = state["accepted"], state["attempts"]
    seen = {sample_key(row) for row in accepted.values()}
    meter = Progress("turn-generation-accepted", len(plan), "dialogues", path=directory / "progress.json")
    meter.skip(len(accepted))
    stop = "attempt_limit"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending := [spec for spec in plan if spec["slot"] not in accepted
                          and attempts.get(spec["slot"], 0) < RETRIES]:
            available = min(workers, int((budget - state["spend_usd"] + 1e-9) / RESERVE_USD))
            if available <= 0:
                stop = "budget_cap"
                break
            batches = schedule_batches(pending, attempts, available)
            for batch in batches:
                for spec in batch:
                    slot = spec["slot"]
                    attempts[slot] = attempts.get(slot, 0) + 1
            state["spend_usd"] += len(batches) * RESERVE_USD
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
                state["spend_usd"] += spend - RESERVE_USD
                if spend > RESERVE_USD:
                    stop = "reservation_exceeded"
                for row in rows:
                    fingerprint = sample_key(row)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    accepted[row["slot"]] = row
                    meter.advance(1)
                _write_json(state_path, state)
            if stop in {"provider_error", "reservation_exceeded"}:
                break
    if len(accepted) == len(plan):
        stop = "complete"
    meter.finish(stop)
    ordered = sorted(accepted.values(), key=lambda row: row["slot"])
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
    temporary = directory / "dialogues.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(directory / "dialogues.jsonl")
    criteria = Counter(row["criterion"] for row in ordered)
    targets = Counter(row["criterion"] for row in plan)
    receipt = {
        "plan_sha256": digest, "dialogues_sha256": sha256(payload.encode()).hexdigest(),
        "generator_prompt_sha256": sha256(SYSTEM.encode()).hexdigest(),
        "cleaner_version": CLEANER_VERSION,
        "generator": GENERATOR, "judge": JUDGE, "target_per_criterion": QUOTA,
        "target_dialogues": len(plan), "accepted_dialogues": len(ordered),
        "accepted_by_language": dict(Counter(row["language"] for row in ordered)),
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
    parser.add_argument("--directory", type=Path, default=Path("data/turn-generation-v1"))
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--workers", type=int, default=6)
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
        for model in (GENERATOR, JUDGE):
            value = request_json(f"models/{model}", key)
            print(value["name"])
        return
    if args.budget_usd is None or not math.isfinite(args.budget_usd) or args.budget_usd <= 0 or not 1 <= args.workers <= 10:
        parser.error("an approved positive --budget-usd and 1–10 workers are required")
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory / "run.lock").open("a") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError:
            parser.error("another process is generating into this directory")
        receipt = run(plan, args.directory, args.budget_usd, args.workers, key)
    if receipt["stop"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
