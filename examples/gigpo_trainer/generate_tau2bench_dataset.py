"""
Dataset generation from trained challenger for tau2-bench solver training.

Pipeline (following Tool-R0):
  1. Generate N raw samples using the trained challenger
  2. Assign pseudo-labels by running a reference solver on each sample
  3. Drop noisy samples (inconsistent pseudo-labels across multiple attempts)
  4. Optionally order samples easy-to-hard by solver success rate
  5. Select target number of samples

Usage:
    python -m examples.gigpo_trainer.generate_tau2bench_dataset \
        --challenger_checkpoint /path/to/challenger_ckpt \
        --domain retail \
        --num_generate 500 \
        --num_target 200 \
        --num_solver_attempts 4 \
        --noise_threshold 0.25 \
        --order_easy_to_hard \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def generate_challenger_samples(
    challenger_url: str,
    challenger_model: str,
    domain: str,
    num_generate: int,
    temperature: float = 0.8,
) -> List[Dict[str, Any]]:
    """Generate raw task samples using the trained challenger.

    Args:
        challenger_url: vLLM server URL for the challenger model
        challenger_model: model name served by the challenger vLLM server
        domain: tau2-bench domain (retail, airline, telecom)
        num_generate: number of samples to generate
        temperature: sampling temperature

    Returns:
        List of generated task specs (question, available_tools, tool_call_answer)
    """
    from openai import OpenAI
    from agent_system.environments.env_package.tau2bench.projection import challenger_projection
    from agent_system.environments.env_package.tau2bench.rewards import (
        compute_challenger_format_reward,
        compute_challenger_validity_reward,
    )

    # Load domain info for the challenger prompt
    from tau2.registry import registry
    env = registry.get_env_constructor(domain)()
    policy = env.get_policy()
    tools = env.get_tools()
    tool_schemas = [t.openai_schema for t in tools]
    tool_names = {t.name for t in tools}
    tools_json = json.dumps(tool_schemas, indent=2)

    domain_info = f"Domain: {domain}\n\nPolicy:\n{policy}\n\nAvailable Tools:\n{tools_json}"

    from agent_system.environments.prompts.tau2bench import (
        TAU2BENCH_CHALLENGER_SYSTEM,
        TAU2BENCH_CHALLENGER_TEMPLATE,
    )
    prompt = TAU2BENCH_CHALLENGER_TEMPLATE.format(
        system_prompt=TAU2BENCH_CHALLENGER_SYSTEM,
        domain_info=domain_info,
    )

    client = OpenAI(base_url=challenger_url, api_key="EMPTY")
    samples = []
    num_valid = 0

    print(f"[generate] Generating {num_generate} samples from challenger...")
    for i in range(num_generate):
        try:
            resp = client.chat.completions.create(
                model=challenger_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2048,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[generate] Error on sample {i}: {e}")
            continue

        # Parse with challenger projection
        parsed_list, valid_list = challenger_projection([text])
        parsed = parsed_list[0]
        valid = valid_list[0]

        if parsed.get("type") != "task_spec":
            continue

        # Compute quality scores
        format_score = compute_challenger_format_reward(parsed)
        validity_score = compute_challenger_validity_reward(parsed, tool_names)

        # Only keep samples with reasonable quality
        if format_score < 0.5 or validity_score < 0.3:
            continue

        sample = {
            "id": f"gen_{domain}_{i:04d}",
            "domain": domain,
            "question": parsed["question"],
            "available_tools": parsed["available_tools"],
            "tool_call_answer": parsed["tool_call_answer"],
            "format_score": format_score,
            "validity_score": validity_score,
            "raw_text": text,
        }
        samples.append(sample)
        num_valid += 1

        if (i + 1) % 50 == 0:
            print(f"[generate] Progress: {i+1}/{num_generate}, valid: {num_valid}")

    print(f"[generate] Generated {num_valid} valid samples out of {num_generate} attempts")
    return samples


def assign_pseudo_labels(
    samples: List[Dict],
    solver_url: str,
    solver_model: str,
    num_attempts: int = 4,
    temperature: float = 0.7,
) -> List[Dict]:
    """Assign pseudo-labels by running a reference solver on each sample.

    For each sample, runs the solver multiple times and records success/failure.
    This gives a success rate per sample which serves as:
      - pseudo-label (is this a solvable, well-formed task?)
      - difficulty estimate (lower success rate = harder)

    Args:
        samples: list of generated task specs
        solver_url: vLLM server URL for the reference solver
        solver_model: model name for the solver
        num_attempts: number of solver attempts per sample
        temperature: solver sampling temperature

    Returns:
        Samples annotated with pseudo_label_success_rate and per-attempt results
    """
    from openai import OpenAI
    from agent_system.environments.env_package.tau2bench.projection import solver_projection
    from agent_system.environments.env_package.tau2bench.rewards import (
        compute_solver_accuracy,
        normalize_tool_call,
        _safe_json_loads,
    )

    client = OpenAI(base_url=solver_url, api_key="EMPTY")

    print(f"[pseudo-label] Labeling {len(samples)} samples with {num_attempts} solver attempts each...")

    for idx, sample in enumerate(samples):
        successes = 0
        attempt_results = []

        # Parse ground truth from the challenger's answer
        gt_obj = _safe_json_loads(sample["tool_call_answer"])
        gt_norm = normalize_tool_call(gt_obj)
        if gt_norm is None:
            sample["pseudo_label_success_rate"] = 0.0
            sample["attempt_results"] = []
            continue

        gt_calls = [gt_norm]

        # Build a simple prompt for the solver
        tools_text = sample["available_tools"]
        question = sample["question"]
        solver_prompt = (
            f"You are a helpful customer service agent.\n\n"
            f"Available Tools:\n{tools_text}\n\n"
            f"Use <tool_call> tags to call tools.\n\n"
            f"Customer: {question}\n\n"
            f"Respond with the appropriate tool call."
        )

        for attempt in range(num_attempts):
            try:
                resp = client.chat.completions.create(
                    model=solver_model,
                    messages=[{"role": "user", "content": solver_prompt}],
                    temperature=temperature,
                    max_tokens=512,
                )
                solver_text = resp.choices[0].message.content or ""
            except Exception as e:
                attempt_results.append({"success": False, "error": str(e)})
                continue

            # Parse solver output
            parsed_list, valid_list = solver_projection([solver_text])
            parsed = parsed_list[0]

            if parsed.get("type") == "tool_call":
                pred_calls = parsed["calls"]
                accuracy, _ = compute_solver_accuracy(pred_calls, gt_calls)
                is_success = accuracy >= 0.99
            else:
                is_success = False
                accuracy = 0.0

            if is_success:
                successes += 1
            attempt_results.append({"success": is_success, "accuracy": accuracy})

        sample["pseudo_label_success_rate"] = successes / max(1, num_attempts)
        sample["attempt_results"] = attempt_results

        if (idx + 1) % 20 == 0:
            print(f"[pseudo-label] Progress: {idx+1}/{len(samples)}")

    return samples


def filter_and_select(
    samples: List[Dict],
    num_target: int,
    noise_threshold: float = 0.25,
    order_easy_to_hard: bool = True,
) -> List[Dict]:
    """Filter noisy samples and select target number, optionally ordered easy-to-hard.

    Following Tool-R0 methodology:
    1. Drop noisy samples (success rate in (noise_threshold, 1-noise_threshold))
       These are ambiguous - neither clearly solvable nor clearly unsolvable.
    2. Keep samples that are either clearly solvable (high success rate) or
       clearly challenging but valid (low but non-zero success rate).
    3. Optionally order easy-to-hard by descending success rate.
    4. Select num_target samples.

    Args:
        samples: pseudo-labeled samples
        num_target: target number of samples to select
        noise_threshold: drop samples with success rate in this middle band
        order_easy_to_hard: whether to sort by difficulty

    Returns:
        Selected samples
    """
    # Step 1: Drop noisy samples
    # Keep samples where success rate is clearly high (>= 1-noise_threshold)
    # or clearly low but with valid formatting (< noise_threshold but format ok)
    # This drops ambiguous samples in the middle band
    clean_samples = []
    dropped = 0
    for s in samples:
        rate = s.get("pseudo_label_success_rate", 0.0)
        # Drop samples with 0 success rate (likely broken/unsolvable)
        if rate == 0.0:
            dropped += 1
            continue
        # Drop noisy middle-band samples
        if noise_threshold < rate < (1.0 - noise_threshold):
            dropped += 1
            continue
        clean_samples.append(s)

    print(f"[filter] Dropped {dropped} noisy samples, {len(clean_samples)} remaining")

    # Step 2: Optionally order easy-to-hard
    if order_easy_to_hard:
        clean_samples.sort(key=lambda s: -s.get("pseudo_label_success_rate", 0.0))
        print(f"[filter] Ordered samples easy-to-hard by success rate")

    # Step 3: Select target number
    selected = clean_samples[:num_target]
    print(f"[filter] Selected {len(selected)} / {num_target} target samples")

    return selected


def save_dataset(samples: List[Dict], output_dir: str, domain: str):
    """Save the generated dataset as parquet and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON (human-readable)
    json_path = os.path.join(output_dir, f"tau2bench_{domain}_generated.json")
    with open(json_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[save] Saved JSON dataset: {json_path}")

    # Save as parquet (for verl-agent training)
    try:
        import pandas as pd
        records = []
        for i, s in enumerate(samples):
            records.append({
                "data_source": f"tau2bench_{domain}_challenger",
                "prompt": json.dumps({
                    "question": s["question"],
                    "available_tools": s["available_tools"],
                    "tool_call_answer": s["tool_call_answer"],
                }),
                "ability": "agent",
                "reward_model": {"style": "tau2bench", "ground_truth": s["tool_call_answer"]},
                "extra_info": {
                    "split": "train",
                    "index": i,
                    "domain": domain,
                    "pseudo_label_success_rate": s.get("pseudo_label_success_rate", 0.0),
                },
            })
        df = pd.DataFrame(records)
        parquet_path = os.path.join(output_dir, f"tau2bench_{domain}_train.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"[save] Saved parquet dataset: {parquet_path}")
    except ImportError:
        print("[save] Warning: pandas not available, skipped parquet output")

    # Save metadata
    meta = {
        "domain": domain,
        "num_samples": len(samples),
        "success_rate_stats": {
            "mean": float(np.mean([s.get("pseudo_label_success_rate", 0) for s in samples])) if samples else 0,
            "min": float(np.min([s.get("pseudo_label_success_rate", 0) for s in samples])) if samples else 0,
            "max": float(np.max([s.get("pseudo_label_success_rate", 0) for s in samples])) if samples else 0,
        },
    }
    meta_path = os.path.join(output_dir, f"tau2bench_{domain}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate tau2-bench training dataset from trained challenger")
    parser.add_argument("--challenger_url", type=str, default="http://localhost:8001/v1",
                        help="vLLM server URL for the trained challenger")
    parser.add_argument("--challenger_model", type=str, default="challenger",
                        help="Model name served by challenger vLLM server")
    parser.add_argument("--solver_url", type=str, default="http://localhost:8002/v1",
                        help="vLLM server URL for the reference solver (for pseudo-labeling)")
    parser.add_argument("--solver_model", type=str, default="solver",
                        help="Model name served by solver vLLM server")
    parser.add_argument("--domain", type=str, default="retail",
                        help="tau2-bench domain")
    parser.add_argument("--num_generate", type=int, default=500,
                        help="Number of samples to generate from challenger")
    parser.add_argument("--num_target", type=int, default=200,
                        help="Target number of samples after filtering")
    parser.add_argument("--num_solver_attempts", type=int, default=4,
                        help="Number of solver attempts per sample for pseudo-labeling")
    parser.add_argument("--noise_threshold", type=float, default=0.25,
                        help="Threshold for dropping noisy samples")
    parser.add_argument("--order_easy_to_hard", action="store_true", default=True,
                        help="Order samples from easy to hard")
    parser.add_argument("--no_order", action="store_true", default=False,
                        help="Disable easy-to-hard ordering")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: $HOME/data/verl-agent/tau2bench_generated)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.environ.get("HOME", "."), "data/verl-agent/tau2bench_generated")

    order = args.order_easy_to_hard and not args.no_order

    print("=" * 60)
    print("  tau2-bench Dataset Generation Pipeline")
    print("=" * 60)
    print(f"  Domain:              {args.domain}")
    print(f"  Num generate:        {args.num_generate}")
    print(f"  Num target:          {args.num_target}")
    print(f"  Solver attempts:     {args.num_solver_attempts}")
    print(f"  Noise threshold:     {args.noise_threshold}")
    print(f"  Easy-to-hard order:  {order}")
    print(f"  Output dir:          {args.output_dir}")
    print("=" * 60)

    # Step 1: Generate challenger samples
    samples = generate_challenger_samples(
        challenger_url=args.challenger_url,
        challenger_model=args.challenger_model,
        domain=args.domain,
        num_generate=args.num_generate,
    )

    if not samples:
        print("[ERROR] No valid samples generated. Check challenger model.")
        sys.exit(1)

    # Step 2: Assign pseudo-labels
    samples = assign_pseudo_labels(
        samples=samples,
        solver_url=args.solver_url,
        solver_model=args.solver_model,
        num_attempts=args.num_solver_attempts,
    )

    # Step 3: Filter and select
    selected = filter_and_select(
        samples=samples,
        num_target=args.num_target,
        noise_threshold=args.noise_threshold,
        order_easy_to_hard=order,
    )

    # Step 4: Save
    save_dataset(selected, args.output_dir, args.domain)

    print(f"\nDone! Generated {len(selected)} training samples.")


if __name__ == "__main__":
    main()
