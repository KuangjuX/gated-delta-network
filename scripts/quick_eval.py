"""
Quick evaluation script: test one workload for correctness then benchmark.
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/quick_eval.py <solution_name> [--full]
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("solution_name")
    parser.add_argument("--full", action="store_true", help="Run all workloads (default: first 5)")
    parser.add_argument("--n", type=int, default=5, help="Number of workloads to test")
    args = parser.parse_args()

    dataset_path = os.environ.get("FIB_DATASET_PATH", "/home/chengqi/mlsys26-contest")

    print(f"Packing {args.solution_name}...")
    solution_path = pack_solution(args.solution_name)
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})\n")

    trace_set = TraceSet.from_path(dataset_path)
    definition = trace_set.definitions[solution.definition]
    all_workloads = trace_set.workloads.get(solution.definition, [])

    if not args.full:
        workloads = all_workloads[:args.n]
    else:
        workloads = all_workloads

    print(f"Testing {len(workloads)} / {len(all_workloads)} workloads\n")

    config = BenchmarkConfig(warmup_runs=2, iterations=20, num_trials=3)

    bench_ts = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    t0 = time.time()
    benchmark = Benchmark(bench_ts, config)
    result_ts = benchmark.run_all(dump_traces=True)
    elapsed = time.time() - t0

    traces = result_ts.traces.get(definition.name, [])
    passed = failed = errored = 0
    latencies = []
    speedups = []

    for trace in traces:
        if not trace.evaluation:
            continue
        status = trace.evaluation.status.value
        wl_id = trace.workload.name if hasattr(trace.workload, 'name') else str(trace.workload)[:40]

        line = f"  {wl_id}: {status}"
        if status.upper() == "PASSED":
            passed += 1
        else:
            failed += 1

        if trace.evaluation.performance:
            p = trace.evaluation.performance
            line += f" | {p.latency_ms:.3f}ms"
            latencies.append(p.latency_ms)
            if p.reference_latency_ms:
                line += f" (ref {p.reference_latency_ms:.3f}ms)"
            if p.speedup_factor:
                line += f" | {p.speedup_factor:.2f}x"
                speedups.append(p.speedup_factor)

        if trace.evaluation.correctness:
            c = trace.evaluation.correctness
            line += f" | abs={c.max_absolute_error:.2e} rel={c.max_relative_error:.2e}"

        print(line)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed (total {len(traces)})")
    if latencies:
        print(f"Latency: min={min(latencies):.3f}ms max={max(latencies):.3f}ms avg={sum(latencies)/len(latencies):.3f}ms")
    if speedups:
        print(f"Speedup: min={min(speedups):.2f}x max={max(speedups):.2f}x avg={sum(speedups)/len(speedups):.2f}x")
    print(f"Wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
