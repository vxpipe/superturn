"""Generate resumable English telephone-style training dialogues for en-scale-v1."""

import argparse
from collections import Counter
from fcntl import LOCK_EX, LOCK_NB, flock
import json
import os
from pathlib import Path
import re
import unicodedata

from .ami_calibration import _write_json
from .shards import seed_for
from .sources import data_root
from .turn_generation import (CATALOGUE, GENERATOR, GEN_INPUT, GEN_OUTPUT, GEN_RESERVE_USD,
                              normalize, plan_checksum, request_json, run, sample_key)

SCALE_VERSION = "en-scale-v1"
BUDGET = 25.
TOTALS = {"G1": 7000, "G2": 4000, "G3": 3000, "G4": 1000, "G-NUM": 2000, "G5": 3000}
SHAPES = {
    "G1": "A natural multi-clause narrative of 20–45 words. Mark 2 or 3 internal pauses; at least one follows a complete clause.",
    "G2": "A complete substantive answer of 3–12 words, one pause, then 6–20 words of related explanation by the same caller.",
    "G3-short": "A complete substantive 1–3-word answer. No pause. Do not use a listener backchannel.",
    "G3-medium": "A complete substantive 4–7-word answer. No pause.",
    "G4": "Start with 1–3 words that clearly leave a thought unfinished, one pause, then 6–20 more words.",
    "number-complete": "Complete the request with four or more spoken digits for a fictional reference. No pause.",
    "number-hold": "Dictate a fictional reference of at least four spoken digits, with one pause inside the digit sequence.",
}
VALIDATION_SCENARIOS = {
    "G1": "education_course_info", "G2": "home_repair_booking", "G3": "food_order_tracking",
    "G4": "utilities_outage_report", "G-NUM": "hospitality_table_reservation", "G5": "real_estate_site_visit",
}
TEST_SCENARIOS = {
    "G1": ("park_gate_closure", "A park ranger and a caller discuss a confusing walking trail closure."),
    "G2": ("costume_donation", "A community theatre is looking for a specific costume donation."),
    "G3": ("volunteer_cleanup_supplies", "A cleanup event organiser confirms what equipment a volunteer can bring."),
    "G4": ("wildlife_sighting_report", "A caller reports where they saw an injured wild animal."),
    "G-NUM": ("tool_library_inventory", "A tool library checks fictional inventory codes for borrowed equipment."),
    "G5": ("garden_tour_access", "A caller asks a community garden about access for a garden tour."),
}
SCALE_SYSTEM = """Write original fictional two-person telephone calls in everyday spoken English, not formal writing.
Follow each requested situation and caller shape exactly. Use varying wording, natural fillers, occasional false
starts and repairs when they fit. Do not include interruptions, barge-ins, listener backchannels, role or stage
notes, real names or contact information, banking or account servicing. Always spell every digit as a word.
Use fictional references. If the situation calls for a phone number use only a fictional 555-01xx number, spelled
digit by digit. Only mark a pause that the caller would actually make inside a turn with the separator ‖. Keep
the full caller turn intact with all its parts. Only end the caller turn when a listener could naturally take
the floor. next_agent is a substantive response that advances the conversation. If previous is 'none', leave
previous_agent blank. If previous is 'statement', use a real conversational statement or greeting rather than a
question. Otherwise give a natural preceding agent turn. Return the requested JSON only, one row per slot."""
SCALE_SCHEMA = {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                                "properties": {key: {"type": "string"} for key in (
                                    "slot", "previous_agent", "caller", "next_agent")},
                                "required": ["slot", "previous_agent", "caller", "next_agent"]},
}
DIGITS = set("zero oh one two three four five six seven eight nine".split())
PROHIBITED = re.compile(r"\b(?:banking|accounts?|bank balance|routing number|credit card|debit card)\b")
ACKNOWLEDGMENTS = {"okay", "ok", "got it", "thanks", "thank you", "i see", "uh huh", "mm hmm", "mhm", "all right"}
QUESTION_WORDS = set("what which when where who why how is are was were do does did can could will would have has should may".split())
DIGIT_RUN = re.compile(r"\b(?:" + "|".join(sorted(DIGITS)) + r")(?: (?:" + "|".join(sorted(DIGITS)) + r"))*\b")


def build_plan(catalogue, validation_path=Path("datasets/turn-completion-v1/validation.jsonl")):
    original_validation = {json.loads(line)["slot"].split("/")[1]
                           for line in validation_path.read_text().splitlines()}
    eligible = {scenario["id"]: scenario["description"] for domain in catalogue["domains"]
                if domain["id"] not in {"banking", "collections"} for scenario in domain["scenarios"]
                if all(word not in scenario["id"] for word in ("account", "kyc", "payment", "billing", "bill_"))}
    # The original B1 validation role remains held out globally. The six
    # additional test scenarios are new to the catalogue and the B1 recipe.
    if not set(VALIDATION_SCENARIOS.values()) <= original_validation & set(eligible):
        raise ValueError("the pinned validation scenarios are missing")
    training = sorted(eligible.keys() - original_validation, key=lambda value: seed_for(SCALE_VERSION, value))
    if not training or set(eligible) & {name for name, _ in TEST_SCENARIOS.values()}:
        raise ValueError("training and newly written test scenarios must be disjoint")
    plan = []
    for family, total in TOTALS.items():
        for index in range(total):
            split = "validation" if index % 20 == 0 else "test" if index % 20 == 1 else "train"
            if split == "test":
                scenario, situation = TEST_SCENARIOS[family]
            else:
                scenario = VALIDATION_SCENARIOS[family] if split == "validation" else training[index % len(training)]
                situation = eligible[scenario]
            shape = family
            prior = "usual"
            if family == "G3":
                shape = "G3-medium" if index % 3 == 0 else "G3-short"
            elif family == "G-NUM":
                shape = "number-complete" if (index // 20 + index) % 2 == 0 else "number-hold"
            elif family == "G5":
                prior = "none" if (index // 20 + index) % 2 == 0 else "statement"
                shape = ("G1", "G2", "G3-short", "G3-medium", "G4", "number-complete", "number-hold")[index % 7]
            criterion = f"en/{SCALE_VERSION}/{family}/{scenario}/{shape}/{prior}"
            plan.append({"slot": f"{criterion}/{index + 1:05d}", "criterion": criterion, "language": "en",
                         "native_script": "Latn", "scale_family": family, "scenario": scenario, "situation": situation,
                         "shape": shape, "prior": prior, "split": split, "pause_style": family})
    # Balance all six families in the first wave and on interrupted runs.
    return sorted(plan, key=lambda row: seed_for(SCALE_VERSION, "order", row["slot"]))


def clean_scale_row(row, spec):
    raw = {field: row.get(field) for field in ("previous_agent", "caller", "next_agent")}
    if any(not isinstance(value, str) or len(value) > 480 or re.search(r"[@\[\]<>\r\n\d]", value)
           for value in raw.values()):
        return None
    prior, following = normalize(raw["previous_agent"]), normalize(raw["next_agent"])
    if (bool(prior) != (spec["prior"] != "none") or not following
            or spec["prior"] == "statement" and "?" in raw["previous_agent"]):
        return None
    parts = [normalize(piece) for piece in re.split(r"‖|\|\|", raw["caller"])]
    if any(not part for part in parts) or PROHIBITED.search(" ".join([prior, following, *parts])):
        return None
    for text in [prior, following, " ".join(parts)]:
        letters = [char for char in text if char.isalpha()]
        if text and (not letters or sum(unicodedata.name(char, "").startswith("LATIN") for char in letters)
                     < .8 * len(letters)):
            return None
        for match in DIGIT_RUN.finditer(text):
            digits = match.group().replace("oh", "zero").split()
            if len(digits) >= 7 and (len(digits) != 7 or digits[:5] != ["five", "five", "five", "zero", "one"]):
                return None
    words = [part.split() for part in parts]
    size = [len(part) for part in words]
    shape = spec["shape"]
    if shape == "G1":
        valid = len(size) in {3, 4} and 20 <= sum(size) <= 45 and min(size) >= 2
    elif shape == "G2":
        valid = len(size) == 2 and 3 <= size[0] <= 12 and 6 <= size[1] <= 20
    elif shape == "G4":
        valid = len(size) == 2 and 1 <= size[0] <= 3 and 6 <= size[1] <= 20
    elif shape == "G3-short":
        is_answer = parts[0] not in ACKNOWLEDGMENTS and (parts[0] not in {"yes", "yeah", "no", "sure"}
                     or bool(prior) and ("?" in raw["previous_agent"] or prior.split()[0] in QUESTION_WORDS))
        valid = len(size) == 1 and 1 <= size[0] <= 3 and is_answer
    elif shape == "G3-medium":
        valid = len(size) == 1 and 4 <= size[0] <= 7
    else:
        digits = sum(word in DIGITS for part in words for word in part)
        valid = digits >= 4 and words[-1][-1] in DIGITS and (
            len(size) == 1 if shape == "number-complete" else
            len(size) == 2 and words[0][-1] in DIGITS and words[1][0] in DIGITS)
    if not valid:
        return None
    return {"previous_agent": prior, "caller_before_pause": parts[0],
            "caller_after_pause": "‖".join(parts[1:]), "next_agent": following}


def generate_scale_batch(specs, key, attempt):
    cases = [{"slot": row["slot"], "situation": (
        "A caller asks how long a prepaid mobile data voucher lasts before buying it. No account is accessed."
        if row["scenario"] == "telecom_recharge" else row["situation"]), "prior": row["prior"],
              "shape": SHAPES[row["shape"]]} for row in specs]
    response = request_json(f"models/{GENERATOR}:generateContent", key, {
        "system_instruction": {"parts": [{"text": SCALE_SYSTEM}]},
        "contents": [{"parts": [{"text": json.dumps({"attempt": attempt, "cases": cases})}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": SCALE_SCHEMA,
                             "maxOutputTokens": 3400, "thinkingConfig": {"thinkingLevel": "LOW"}},
    })
    usage = response.get("usageMetadata", {})
    counted_input = usage.get("promptTokenCount")
    output = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    spend = (counted_input * GEN_INPUT + output * GEN_OUTPUT) / 1_000_000 if counted_input is not None else GEN_RESERVE_USD
    candidate = next(iter(response.get("candidates", [])), {})
    content = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
                      if not part.get("thought"))
    try:
        rows = json.loads(content)
    except ValueError:
        return [], spend
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("slot"), str) for row in rows):
        return [], spend
    by_slot = {row["slot"]: row for row in rows}
    if len(by_slot) != len(rows):
        return [], spend
    checked = []
    for spec in specs:
        cleaned = clean_scale_row(by_slot[spec["slot"]], spec) if spec["slot"] in by_slot else None
        if cleaned is not None:
            checked.append({**spec, **cleaned, "generator_version": response.get("modelVersion"),
                            "generation_attempt": attempt})
    return checked, spend


def scale_examples(row):
    parts = [row["caller_before_pause"]]
    if row["caller_after_pause"]:
        parts.extend(row["caller_after_pause"].split("‖"))
    return [{"slot": row["slot"], "language": "en", "previous_turn": row["previous_agent"],
             "current_turn": " ".join(parts[:index + 1]), "label": "shift" if index == len(parts) - 1 else "hold"}
            for index in range(len(parts))]


def revalidate_state(directory, plan):
    path = directory / "state.json"
    if not path.is_file():
        return
    state = json.loads(path.read_text())
    if state["plan_sha256"] != plan_checksum(plan):
        raise ValueError("the ledger belongs to a different scale plan")
    specs = {row["slot"]: row for row in plan}
    rejected = []
    for slot, row in state["accepted"].items():
        caller = row["caller_before_pause"]
        if row["caller_after_pause"]:
            caller += "‖" + row["caller_after_pause"]
        if clean_scale_row({**row, "caller": caller}, specs[slot]) is None:
            rejected.append(slot)
    if rejected:
        for slot in rejected:
            del state["accepted"][slot]
        state["revalidation_rejected"] = state.get("revalidation_rejected", 0) + len(rejected)
        _write_json(path, state)
        print(f"revalidated the existing ledger and reopened {len(rejected)} slots", flush=True)


def reopen_excluded(directory, plan, audit_path):
    from .turn_text_train import digest

    audit = json.loads(audit_path.read_text())
    path = directory / "state.json"
    state = json.loads(path.read_text())
    if (audit["plan_sha256"] != state["plan_sha256"] or state["plan_sha256"] != plan_checksum(plan)
            or audit["dialogues_sha256"] != digest(directory / "dialogues.jsonl")):
        raise ValueError("refill requires the audit for the current accepted ledger")
    for slot in audit["excluded_slots"]:
        rejected = state["accepted"].pop(slot)
        state.setdefault("excluded_sample_keys", []).append(sample_key(rejected))
    _write_json(path, state)
    print(f"reopened {len(audit['excluded_slots'])} pretraining audit exclusions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=data_root() / "cache" / SCALE_VERSION)
    parser.add_argument("--budget-usd", type=float, default=BUDGET)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--refill-from", type=Path, help="reopen slots rejected in a matching Vast pretraining audit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="use a small portion of the same resumable ledger budget")
    args = parser.parse_args()
    plan = build_plan(json.loads(CATALOGUE.read_text()))
    if args.plan:
        print(json.dumps({"plan_sha256": plan_checksum(plan), "target": len(plan),
                          "counts": Counter(f"{row['scale_family']}/{row['split']}" for row in plan)}, indent=2))
        return
    key = os.environ.get("GEMINI_API_KEY")
    if not key or not 0 < args.budget_usd <= BUDGET or not 1 <= args.workers <= 10:
        parser.error("set GEMINI_API_KEY, use no more than $25 and use 1–10 workers")
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory / "run.lock").open("a") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError:
            parser.error("another process holds the en-scale-v1 ledger lock")
        if args.refill_from:
            reopen_excluded(args.directory, plan, args.refill_from)
        revalidate_state(args.directory, plan)
        if args.smoke:
            args.budget_usd = min(args.budget_usd, .20)
        receipt = run(plan, args.directory, args.budget_usd, args.workers, key, attempts_per_slot=6)
    if receipt["stop"] != "complete" and not args.smoke:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
