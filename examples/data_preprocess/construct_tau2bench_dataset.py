"""
Dataset construction for tau2-bench solver training (Tool-R0 style).

Pipeline:
  1. Load trained challenger model and generate N candidate task specs
  2. Validate quality: check format, validity, and pseudo-label correctness
  3. (Optional) Order samples easy-to-hard based on difficulty proxy
  4. Drop noisy / low-quality samples
  5. Select target number of samples for solver training

Usage:
    python -m examples.data_preprocess.construct_tau2bench_dataset \
        --challenger_model /path/to/trained_challenger \
        --domain retail \
        --num_generate 500 \
        --num_target 200 \
        --order_easy_to_hard \
        --output_dir ~/data/verl-agent/tau2bench_solver
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


def generate_samples(
    challenger_model: str,
    domain: str,
    num_generate: int,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    challenger_url: Optional[str] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate candidate task specs using the trained challenger model.

    Args:
        challenger_model: path or name of trained challenger model
        domain: tau2-bench domain (retail, airline, telecom)
        num_generate: number of samples to generate
        temperature: sampling temperature
        max_tokens: max generation tokens
        challenger_url: vLLM API URL (if using API server)
        seed: random seed

    Returns:
        list of raw generated samples
    """
    from tau2.registry import registry

    # Get domain info for the prompt
    env = registry.get_env_constructor(domain)()
    policy = env.get_policy()
    tools = env.get_tools()
    tool_schemas = [t.openai_schema for t in tools]
    tools_json = json.dumps(tool_schemas, indent=2)

    domain_info = (
        f"Domain: {domain}\n\n"
        f"Policy:\n{policy}\n\n"
        f"Available Tools:\n{tools_json}"
    )

    from agent_system.environments.prompts.tau2bench import (
        TAU2BENCH_CHALLENGER_SYSTEM,
        TAU2BENCH_CHALLENGER_TEMPLATE,
    )

    prompt = TAU2BENCH_CHALLENGER_TEMPLATE.format(
        system_prompt=TAU2BENCH_CHALLENGER_SYSTEM,
        domain_info=domain_info,
    )

    # Use OpenAI-compatible API for generation
    from openai import OpenAI

    if challenger_url is None:
        challenger_url = "http://localhost:8001/v1"

    client = OpenAI(base_url=challenger_url, api_key="EMPTY")
    rng = random.Random(seed)

    raw_samples = []
    print(f"[DataConstruct] Generating {num_generate} samples with challenger...")

    for i in range(num_generate):
        try:
            resp = client.chat.completions.create(
                model=challenger_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=rng.randint(0, 2**31),
            )
            text = resp.choices[0].message.content or ""
            raw_samples.append({
                "idx": i,
                "raw_text": text,
                "domain": domain,
            })
        except Exception as e:
            print(f"[DataConstruct] Warning: generation {i} failed: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"[DataConstruct]   Generated {i + 1}/{num_generate}")

    print(f"[DataConstruct] Generated {len(raw_samples)} raw samples")
    return raw_samples


def parse_and_validate(
    raw_samples: List[Dict],
    domain: str,
    min_format_score: float = 0.5,
    min_validity_score: float = 0.5,
) -> List[Dict[str, Any]]:
    """Parse raw samples and validate quality with pseudo labels.

    Args:
        raw_samples: list of raw generated samples
        domain: tau2-bench domain
        min_format_score: minimum format reward to keep sample
        min_validity_score: minimum validity reward to keep sample

    Returns:
        list of validated samples with quality scores
    """
    from agent_system.environments.env_package.tau2bench.projection import (
        challenger_projection,
    )
    from agent_system.environments.env_package.tau2bench.rewards import (
        compute_challenger_format_reward,
        compute_challenger_validity_reward,
    )
    from tau2.registry import registry

    # Get domain tool names for validity check
    env = registry.get_env_constructor(domain)()
    tools = env.get_tools()
    domain_tool_names = {t.name for t in tools}

    validated = []
    texts = [s["raw_text"] for s in raw_samples]
    parsed_list, valids = challenger_projection(texts)

    for i, (sample, parsed, valid) in enumerate(
        zip(raw_samples, parsed_list, valids)
    ):
        if not valid:
            continue

        format_score = compute_challenger_format_reward(parsed)
        validity_score = compute_challenger_validity_reward(
            parsed, domain_tool_names
        )

        if format_score < min_format_score or validity_score < min_validity_score:
            continue

        # Extract pseudo label (the gold tool call answer)
        from agent_system.environments.env_package.tau2bench.rewards import (
            _safe_json_loads,
            normalize_tool_call,
        )

        answer_text = parsed.get("tool_call_answer", "")
        answer_obj = _safe_json_loads(answer_text)
        gold_call = normalize_tool_call(answer_obj)

        if gold_call is None:
            continue

        validated.append({
            **sample,
            "parsed": parsed,
            "gold_call": gold_call,
            "format_score": format_score,
            "validity_score": validity_score,
            "quality_score": (format_score + validity_score) / 2.0,
        })

    print(
        f"[DataConstruct] Validated {len(validated)}/{len(raw_samples)} samples "
        f"(dropped {len(raw_samples) - len(validated)} low quality)"
    )
    return validated


def estimate_difficulty(
    samples: List[Dict],
) -> List[Dict]:
    """Estimate difficulty of each sample for easy-to-hard ordering.

    Difficulty proxy: number of required arguments + number of tools available.
    More args and more tools = harder.

    Args:
        samples: validated samples

    Returns:
        samples with added 'difficulty' field
    """
    from agent_system.environments.env_package.tau2bench.rewards import (
        _safe_json_loads,
    )

    for sample in samples:
        parsed = sample["parsed"]
        gold_call = sample["gold_call"]

        # Factor 1: number of arguments in gold call
        num_args = len(gold_call.get("arguments", {}))

        # Factor 2: number of available tools (more tools = harder to pick)
        tools_text = parsed.get("available_tools", "[]")
        tools = _safe_json_loads(tools_text)
        num_tools = len(tools) if isinstance(tools, list) else 1

        # Factor 3: inverse quality (lower quality = potentially harder/noisier)
        quality_penalty = 1.0 - sample.get("quality_score", 0.5)

        # Difficulty score (higher = harder)
        sample["difficulty"] = num_args * 0.4 + num_tools * 0.1 + quality_penalty * 0.5

    return samples


def select_samples(
    samples: List[Dict],
    num_target: int,
    order_easy_to_hard: bool = True,
    noise_threshold: float = 0.4,
) -> List[Dict]:
    """Drop noisy samples and select target number.

    Args:
        samples: validated samples with quality scores
        num_target: target number of samples to select
        order_easy_to_hard: whether to sort by difficulty
        noise_threshold: quality_score below this is considered noisy

    Returns:
        selected samples
    """
    # Drop noisy samples (low quality score)
    clean = [s for s in samples if s.get("quality_score", 0.0) >= noise_threshold]
    dropped = len(samples) - len(clean)
    if dropped > 0:
        print(f"[DataConstruct] Dropped {dropped} noisy samples (quality < {noise_threshold})")

    if order_easy_to_hard:
        clean = estimate_difficulty(clean)
        clean.sort(key=lambda s: s.get("difficulty", 0.0))
        print("[DataConstruct] Ordered samples easy-to-hard")

    # Select target number
    if len(clean) > num_target:
        selected = clean[:num_target]
        print(f"[DataConstruct] Selected {num_target} from {len(clean)} clean samples")
    else:
        selected = clean
        print(
            f"[DataConstruct] Warning: only {len(clean)} clean samples available "
            f"(target was {num_target})"
        )

    return selected


def save_dataset(
    samples: List[Dict],
    output_dir: str,
    split: str = "train",
) -> str:
    """Save dataset as parquet file for verl-agent training.

    Args:
        samples: selected samples
        output_dir: output directory
        split: train or test

    Returns:
        path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build records for parquet
    records = []
    for i, sample in enumerate(samples):
        parsed = sample["parsed"]
        records.append({
            "idx": i,
            "question": parsed.get("question", ""),
            "available_tools": parsed.get("available_tools", "[]"),
            "tool_call_answer": parsed.get("tool_call_answer", "{}"),
            "domain": sample.get("domain", "unknown"),
            "quality_score": sample.get("quality_score", 0.0),
            "difficulty": sample.get("difficulty", 0.0),
            "data_source": f"tau2bench_challenger_{sample.get('domain', 'unknown')}",
            "modality": "text",
        })

    output_path = os.path.join(output_dir, f"{split}.parquet")

    if pd is not None:
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)
    else:
        # Fallback: save as JSON lines
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    print(f"[DataConstruct] Saved {len(records)} samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Construct tau2-bench solver training dataset from trained challenger"
    )
    parser.add_argument(
        "--challenger_model",
        type=str,
        required=True,
        help="Path or name of trained challenger model",
    )
    parser.add_argument(
        "--challenger_url",
        type=str,
        default="http://localhost:8001/v1",
        help="vLLM API URL for challenger model",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="retail",
        choices=["retail", "airline", "telecom"],
    )
    parser.add_argument(
        "--num_generate",
        type=int,
        default=500,
        help="Number of raw samples to generate",
    )
    parser.add_argument(
        "--num_target",
        type=int,
        default=200,
        help="Target number of samples to select",
    )
    parser.add_argument(
        "--order_easy_to_hard",
        action="store_true",
        default=False,
        help="Order samples from easy to hard",
    )
    parser.add_argument(
        "--noise_threshold",
        type=float,
        default=0.4,
        help="Quality score threshold for dropping noisy samples",
    )
    parser.add_argument(
        "--min_format_score",
        type=float,
        default=0.5,
        help="Minimum format reward to keep a sample",
    )
    parser.add_argument(
        "--min_validity_score",
        type=float,
        default=0.5,
        help="Minimum validity reward to keep a sample",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ~/data/verl-agent/tau2bench_solver)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.expanduser(
            f"~/data/verl-agent/tau2bench_solver/{args.domain}"
        )

    print("=" * 60)
    print("  tau2-bench Dataset Construction")
    print("=" * 60)
    print(f"  Challenger: {args.challenger_model}")
    print(f"  Domain:     {args.domain}")
    print(f"  Generate:   {args.num_generate}")
    print(f"  Target:     {args.num_target}")
    print(f"  Easy→Hard:  {args.order_easy_to_hard}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 60)

    # Step 1: Generate
    raw_samples = generate_samples(
        challenger_model=args.challenger_model,
        domain=args.domain,
        num_generate=args.num_generate,
        temperature=args.temperature,
        challenger_url=args.challenger_url,
        seed=args.seed,
    )

    # Step 2: Validate with pseudo labels
    validated = parse_and_validate(
        raw_samples,
        domain=args.domain,
        min_format_score=args.min_format_score,
        min_validity_score=args.min_validity_score,
    )

    if not validated:
        print("[DataConstruct] ERROR: No valid samples after validation. Exiting.")
        return

    # Step 3 & 4: Drop noisy, order, and select
    selected = select_samples(
        validated,
        num_target=args.num_target,
        order_easy_to_hard=args.order_easy_to_hard,
        noise_threshold=args.noise_threshold,
    )

    # Step 5: Save
    # Save train split (90%) and test split (10%)
    split_idx = max(1, int(len(selected) * 0.9))
    train_samples = selected[:split_idx]
    test_samples = selected[split_idx:]

    save_dataset(train_samples, args.output_dir, split="train")
    if test_samples:
        save_dataset(test_samples, args.output_dir, split="test")

    print(f"\n[DataConstruct] Done! Train: {len(train_samples)}, Test: {len(test_samples)}")


if __name__ == "__main__":
    main()
