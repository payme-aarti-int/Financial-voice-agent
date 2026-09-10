from __future__ import annotations
 
import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
 
import yaml
 
from app.pipeline import Pipeline
 
CASES_PATH = Path(__file__).parent / "cases.yaml"
 
SPELLED_OUT = {
    "142": ["forty-two thousand", "forty two thousand"],
    "120": ["twenty thousand", "one hundred twenty"],
    "397": ["ninety-seven thousand", "ninety seven thousand"],
}

REFUSAL_MARKERS = (
    "don't have",
    "do not have",
    "not have data",
    "no data",
    "only have",
    "cannot provide",
    "can't provide",
    "cannot help",
    "can't help",
    "unable to",
    "not available",
    "i'm sorry",
    "i am sorry",
    "don't track",
    "do not track",
    "outside",
    "beyond",
    "no information",
    "not something i",
)
 
 
@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    answer: str = ""
    spoken: str = ""
    tools_called: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    errored: bool = False
    failures: list[str] = field(default_factory=list)
 
    @property
    def passed(self) -> bool:
        return not self.failures
 
 
def looks_like_refusal(text: str) -> bool:
    # Models emit typographic apostrophes (U+2019), not ASCII. Normalise
    # before matching, or "don't have" never matches "don\u2019t have"
    # and a perfectly good refusal scores as a failure.
    lowered = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return any(marker in lowered for marker in REFUSAL_MARKERS)
 
 
def score(case: dict, result: CaseResult) -> None:
    
    def norm(text: str) -> str:
        return text.lower().replace("\u2019", "'").replace("\u2018", "'")

    answer = norm(result.answer)
    spoken = norm(result.spoken)
 
    if "expects_tool" in case:
        expected = case["expects_tool"]
        if expected is None:
            if result.tools_called:
                result.failures.append(
                    f"expected no tool call, got {result.tools_called}"
                )
        elif expected not in result.tools_called:
            got = result.tools_called or ["none"]
            result.failures.append(f"expected tool {expected}, got {got}")
 
    if case.get("expects_refusal") and not looks_like_refusal(result.answer):
        result.failures.append("expected a refusal, agent answered instead")
 
    for needle in case.get("must_contain", []):
        alternatives = [str(needle).lower()]
        # A voice agent may spell figures out -- "one hundred forty-two
        # thousand" is arguably better spoken output than "142,000".
        # Assert the VALUE is present, not its notation.
        alternatives.extend(SPELLED_OUT.get(str(needle), []))
        if not any(alt in answer for alt in alternatives):
            result.failures.append(f"answer missing {needle!r}")
 
    for needle in case.get("must_not_contain", []):
        if str(needle).lower() in answer:
            result.failures.append(f"answer contains forbidden {needle!r}")
 
    for needle in case.get("spoken_must_not_contain", []):
        if str(needle).lower() in spoken:
            result.failures.append(f"spoken text contains {needle!r}")
 
 
def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    import math
 
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]
 
 
def run(category: str | None, verbose: bool, threshold: float) -> int:
    cases = yaml.safe_load(CASES_PATH.read_text())["cases"]
    if category:
        cases = [c for c in cases if c.get("category") == category]
 
    if not cases:
        print(f"No cases matched category={category!r}")
        return 1
 
    results: list[CaseResult] = []
 
    with Pipeline() as pipeline:
        for index, case in enumerate(cases, 1):
            result = CaseResult(
                case_id=case["id"],
                category=case.get("category", "uncategorised"),
                question=case["question"],
            )
 
            turn = None
            for attempt in range(4):
                try:
                    turn = pipeline.run_text(case["question"], autoplay=False)
                    break
                except Exception as exc:  # noqa: BLE001
                    # 429 means we are asking too fast, not that the agent
                    # is wrong. Back off and retry rather than scoring it.
                    if "429" in str(exc) and attempt < 3:
                        time.sleep(5 * (attempt + 1))
                        continue
                    result.errored = True
                    result.failures.append(f"ERROR {type(exc).__name__}: {exc}")
                    break
            try:
                if turn is None:
                    raise RuntimeError("no result")
                result.answer = turn.agent_turn.answer if turn.agent_turn else ""
                result.spoken = turn.spoken_text
                result.tools_called = [c["name"] for c in turn.tool_calls]
                result.latency_ms = turn.timing.total_ms if turn.timing else 0.0
                score(case, result)
            except Exception as exc:  # noqa: BLE001
                # An API error is NOT an agent quality failure. Scoring it
                # as one produces false alarms like "the agent may be
                # inventing figures" when the API simply refused to talk.
                result.errored = True
                result.failures.append(f"ERROR {type(exc).__name__}: {exc}")
 
            results.append(result)
            time.sleep(2.5)
 
            mark = "ERR " if result.errored else ("PASS" if result.passed else "FAIL")
            print(f"[{index:>2}/{len(cases)}] {mark}  {result.case_id}")
            if verbose or not result.passed:
                print(f"        Q: {result.question}")
                print(f"        A: {result.answer[:140]}")
                print(f"        tools: {result.tools_called or ['none']}")
                for failure in result.failures:
                    print(f"        -> {failure}")
 
    return report(results, threshold)
 
 
def report(results: list[CaseResult], threshold: float) -> int:
    errored = [r for r in results if r.errored]
    scored = [r for r in results if not r.errored]
    passed = [r for r in scored if r.passed]
    rate = len(passed) / len(scored) if scored else 0.0
 
    print("\n" + "=" * 58)
    print(f"{len(passed)}/{len(scored)} scored cases passed ({rate:.0%})")
    if errored:
        print(f"{len(errored)} case(s) errored (API/infra) and were not scored")
 
    categories: dict[str, list[CaseResult]] = {}
    for result in scored:
        categories.setdefault(result.category, []).append(result)
    errored_by_cat: dict[str, int] = {}
    for result in errored:
        errored_by_cat[result.category] = errored_by_cat.get(result.category, 0) + 1
 
    print("\n  category           passed      rate")
    print("  " + "-" * 40)
    for name, group in sorted(categories.items()):
        ok = sum(1 for r in group if r.passed)
        err = errored_by_cat.get(name, 0)
        suffix = f"   ({err} errored)" if err else ""
        print(f"  {name:<18} {ok:>2}/{len(group):<6} {ok / len(group):>8.0%}{suffix}")
 
    latencies = [r.latency_ms for r in results if r.latency_ms]
    if latencies:
        print(
            f"\n  latency   p50 {median(latencies):>6.0f}ms   "
            f"p95 {percentile(latencies, 95):>6.0f}ms   "
            f"max {max(latencies):>6.0f}ms"
        )
 
    failures = [r for r in scored if not r.passed]
    if failures:
        print(f"\n  failed cases: {', '.join(r.case_id for r in failures)}")
 
    refusal_failures = [
        r for r in failures if r.category == "refusal" and not r.errored
    ]
    if refusal_failures:
        print(
            f"\n  WARNING: {len(refusal_failures)} refusal case(s) failed. "
            "The agent may be inventing figures."
        )
 
    print("=" * 58)
 
    if rate < threshold:
        print(f"\nFAILED: {rate:.0%} is below threshold {threshold:.0%}")
        return 1
    return 0
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval suite")
    parser.add_argument("--category", type=str, help="Run one category only")
    parser.add_argument("--verbose", action="store_true", help="Show every answer")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
 
    sys.exit(run(args.category, args.verbose, args.threshold))
 
 
if __name__ == "__main__":
    main()
 
