"""
FlashInfer-Bench Local Benchmark Runner.

Packs a solution from solutions/<name>/ and runs benchmarks locally.

Usage:
    python scripts/run_local.py <solution_name>
    python scripts/run_local.py gdn_decode_qk4_v8_d128_k_last
    python scripts/run_local.py gdn_prefill_qk4_v8_d128_k_last
    python scripts/run_local.py all   # run all solutions
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark locally and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{'='*80}")
        print(f"  {def_name}")
        print(f"{'='*80}")

        passed = failed = errored = 0
        latencies = []
        speedups = []

        for workload_uuid, result in traces.items():
            status = result.get("status", "unknown")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if status == "pass":
                passed += 1
            elif status == "fail":
                failed += 1
            else:
                errored += 1

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")
                latencies.append(result["latency_ms"])

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")
                speedups.append(result["speedup_factor"])

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

        print(f"\n  Summary: {passed} passed, {failed} failed, {errored} errored"
              f" (total {len(traces)} workloads)")
        if latencies:
            print(f"  Latency: min={min(latencies):.3f}ms, max={max(latencies):.3f}ms, "
                  f"avg={sum(latencies)/len(latencies):.3f}ms")
        if speedups:
            print(f"  Speedup: min={min(speedups):.2f}x, max={max(speedups):.2f}x, "
                  f"avg={sum(speedups)/len(speedups):.2f}x")


def list_solutions() -> list[str]:
    """List available solution names."""
    solutions_dir = PROJECT_ROOT / "solutions"
    if not solutions_dir.exists():
        return []
    return [d.name for d in solutions_dir.iterdir() if d.is_dir() and (d / "config.toml").exists()]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run FlashInfer-Bench local benchmarks")
    parser.add_argument(
        "solution_name",
        nargs="?",
        default="all",
        help="Solution name under solutions/, or 'all' to run all (default: all)"
    )
    args = parser.parse_args()

    available = list_solutions()
    if not available:
        print("No solutions found in solutions/ directory!")
        sys.exit(1)

    if args.solution_name == "all":
        solution_names = available
    else:
        if args.solution_name not in available:
            print(f"Solution '{args.solution_name}' not found. Available: {available}")
            sys.exit(1)
        solution_names = [args.solution_name]

    for sol_name in solution_names:
        print(f"\n{'#'*80}")
        print(f"  Benchmarking: {sol_name}")
        print(f"{'#'*80}")

        print("\nPacking solution from source files...")
        solution_path = pack_solution(sol_name)

        print("\nLoading solution...")
        solution = Solution.model_validate_json(solution_path.read_text())
        print(f"Loaded: {solution.name} ({solution.definition})")

        print("\nRunning benchmark...")
        results = run_benchmark(solution)

        if not results:
            print("No results returned!")
            continue

        print_results(results)


if __name__ == "__main__":
    main()
