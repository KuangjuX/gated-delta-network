"""
Pack solution source files into solution.json.

Reads configuration from a solution's config.toml under solutions/<name>/
and packs the appropriate source files into a Solution JSON file.

Usage:
    python scripts/pack_solution.py <solution_name>
    python scripts/pack_solution.py gdn_decode_qk4_v8_d128_k_last
    python scripts/pack_solution.py gdn_prefill_qk4_v8_d128_k_last
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


def load_config(solution_dir: Path) -> dict:
    """Load configuration from a solution's config.toml."""
    config_path = solution_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(solution_name: str, output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    solution_dir = PROJECT_ROOT / "solutions" / solution_name
    if not solution_dir.exists():
        raise FileNotFoundError(
            f"Solution directory not found: {solution_dir}\n"
            f"Available solutions: {[d.name for d in (PROJECT_ROOT / 'solutions').iterdir() if d.is_dir()]}"
        )

    config = load_config(solution_dir)
    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    if language == "triton":
        source_dir = solution_dir / "solution" / "triton"
    elif language == "cuda":
        source_dir = solution_dir / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    dps = build_config.get("destination_passing_style", True)
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=dps,
    )

    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    if output_path is None:
        output_path = solution_dir / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "solution_name",
        help="Name of the solution directory under solutions/ (e.g. gdn_decode_qk4_v8_d128_k_last)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: solutions/<name>/solution.json)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.solution_name, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
