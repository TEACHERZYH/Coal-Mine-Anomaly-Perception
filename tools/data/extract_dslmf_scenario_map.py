from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCENARIO_PATTERN = re.compile(r"^Scenario\s+([0-9]+)$", re.IGNORECASE)
RANGE_PATTERN = re.compile(
    r"^(?P<start>[A-Za-z0-9_]+)[.]jpg-(?P<end>[A-Za-z0-9_]+)[.]jpg$"
)


class ScenarioMapError(ValueError):
    """Raised when the official scenario index cannot be parsed unambiguously."""


def _normalize_document_stem(value: str) -> str:
    text = value.strip()
    if re.fullmatch(r"[0-9]+", text):
        return text
    match = re.fullmatch(r"[pP]([0-9]+)_([0-9]+)", text)
    if match is None:
        raise ScenarioMapError(f"Unsupported scenario filename stem: {value}")
    return f"p{match.group(1)}-{match.group(2)}"


def _filename_key(value: str) -> tuple[int, int, int]:
    if re.fullmatch(r"[0-9]+", value):
        return (0, int(value), 0)
    match = re.fullmatch(r"[pP]([0-9]+)-([0-9]+)", value)
    if match is None:
        raise ScenarioMapError(f"Unsupported normalized filename stem: {value}")
    return (1, int(match.group(1)), int(match.group(2)))


def parse_scenario_text(text: str, *, expected_scenarios: int = 58) -> list[dict[str, Any]]:
    scenarios: dict[int, list[dict[str, str]]] = {}
    current: int | None = None
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        scenario_match = SCENARIO_PATTERN.fullmatch(line)
        if scenario_match:
            current = int(scenario_match.group(1))
            if current in scenarios:
                raise ScenarioMapError(f"Duplicate Scenario {current}")
            scenarios[current] = []
            continue
        range_match = RANGE_PATTERN.fullmatch(line)
        if range_match:
            if current is None:
                raise ScenarioMapError("Scenario range appears before a Scenario heading")
            start = _normalize_document_stem(range_match.group("start"))
            end = _normalize_document_stem(range_match.group("end"))
            start_key = _filename_key(start)
            end_key = _filename_key(end)
            if start_key[0] != end_key[0] or start_key > end_key:
                raise ScenarioMapError(
                    f"Scenario {current} contains incompatible or descending range {line}"
                )
            scenarios[current].append(
                {
                    "value": f"scenario-{current:02d}",
                    "start": start,
                    "end": end,
                }
            )
            continue
        if current is not None and ".jpg" in line.lower():
            raise ScenarioMapError(f"Unparsed filename range under Scenario {current}: {line}")

    expected = list(range(1, expected_scenarios + 1))
    if sorted(scenarios) != expected:
        missing = sorted(set(expected) - set(scenarios))
        extra = sorted(set(scenarios) - set(expected))
        raise ScenarioMapError(f"Scenario headings are incomplete: missing={missing}, extra={extra}")
    if any(not scenarios[index] for index in expected):
        empty = [index for index in expected if not scenarios[index]]
        raise ScenarioMapError(f"Scenarios contain no filename ranges: {empty}")

    return [
        {
            "scenario_id": f"scenario-{index:02d}",
            "scenario_number": index,
            "ranges": scenarios[index],
        }
        for index in expected
    ]


def _overlap_audit(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals = []
    for scenario in scenarios:
        for item in scenario["ranges"]:
            intervals.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "start_key": _filename_key(item["start"]),
                    "end_key": _filename_key(item["end"]),
                }
            )
    overlaps = []
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            if left["start_key"][0] != right["start_key"][0]:
                continue
            start = max(left["start_key"], right["start_key"])
            end = min(left["end_key"], right["end_key"])
            if start > end:
                continue
            if start[0] != 0:
                raise ScenarioMapError(
                    "Overlapping p<major>-<minor> ranges require manual source review"
                )
            width = end[1] - start[1] + 1
            if width > 1000:
                raise ScenarioMapError("Official numeric overlap is too large for bounded review")
            overlaps.append(
                {
                    "left_scenario_id": left["scenario_id"],
                    "right_scenario_id": right["scenario_id"],
                    "start": f"{start[1]:07d}",
                    "end": f"{end[1]:07d}",
                    "ambiguous_filename_stems": [
                        f"{value:07d}" for value in range(start[1], end[1] + 1)
                    ],
                }
            )
    return overlaps


def _adapter_ranges(
    scenarios: list[dict[str, Any]], ambiguous_stems: set[str]
) -> list[dict[str, str]]:
    ambiguous_numeric = sorted(
        int(value) for value in ambiguous_stems if re.fullmatch(r"[0-9]+", value)
    )
    result = []
    for scenario in scenarios:
        for item in scenario["ranges"]:
            start_key = _filename_key(item["start"])
            end_key = _filename_key(item["end"])
            if start_key[0] == 1:
                result.append(dict(item))
                continue
            cursor = start_key[1]
            for excluded in ambiguous_numeric:
                if excluded < cursor or excluded > end_key[1]:
                    continue
                if cursor <= excluded - 1:
                    result.append(
                        {
                            "value": item["value"],
                            "start": f"{cursor:07d}",
                            "end": f"{excluded - 1:07d}",
                        }
                    )
                cursor = excluded + 1
            if cursor <= end_key[1]:
                result.append(
                    {
                        "value": item["value"],
                        "start": f"{cursor:07d}",
                        "end": f"{end_key[1]:07d}",
                    }
                )
    return result


def extract_pdf_text(path: Path) -> tuple[str, int]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ScenarioMapError(
            "PDF extraction requires pdfplumber; use the bundled workspace Python runtime"
        ) from exc
    with pdfplumber.open(path) as document:
        page_text = [page.extract_text() or "" for page in document.pages]
    return "\n".join(page_text), len(page_text)


def build_payload(
    pdf: Path,
    *,
    expected_scenarios: int = 58,
    additional_excluded_stems: list[str] | None = None,
) -> dict[str, Any]:
    text, page_count = extract_pdf_text(pdf)
    scenarios = parse_scenario_text(text, expected_scenarios=expected_scenarios)
    overlaps = _overlap_audit(scenarios)
    ambiguous_stems = sorted(
        {
            stem
            for overlap in overlaps
            for stem in overlap["ambiguous_filename_stems"]
        }
    )
    reviewed_unmapped = sorted(set(additional_excluded_stems or []))
    for stem in reviewed_unmapped:
        _filename_key(stem)
    excluded_stems = sorted(set(ambiguous_stems) | set(reviewed_unmapped))
    adapter_ranges = _adapter_ranges(scenarios, set(ambiguous_stems))
    return {
        "schema_version": "mining1.dslmfplus_scenario_map.v1",
        "source_pdf": pdf.as_posix(),
        "source_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "page_count": page_count,
        "scenario_count": len(scenarios),
        "range_count": sum(len(item["ranges"]) for item in scenarios),
        "official_overlap_count": len(overlaps),
        "official_overlaps": overlaps,
        "ambiguity_policy": "drop_all_filename_ids_assigned_to_multiple_official_scenarios",
        "ambiguous_filename_stems": ambiguous_stems,
        "unmapped_policy": "drop_reviewed_filename_ids_absent_from_all_official_scenario_ranges",
        "reviewed_unmapped_filename_stems": reviewed_unmapped,
        "excluded_image_stems": excluded_stems,
        "adapter_range_count": len(adapter_ranges),
        "adapter_ranges": adapter_ranges,
        "document_filename_rule": (
            "The PDF replaces the hyphen inside dataset p<major>-<minor> stems "
            "with an underscore so the remaining hyphen denotes an inclusive range."
        ),
        "scenarios": scenarios,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--expected-scenarios", type=int, default=58)
    parser.add_argument("--additional-excluded-stem", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(
        args.pdf,
        expected_scenarios=args.expected_scenarios,
        additional_excluded_stems=args.additional_excluded_stem,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
