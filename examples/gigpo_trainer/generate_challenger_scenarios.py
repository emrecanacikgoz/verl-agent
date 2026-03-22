"""
TOD-Zero: Offline Challenger Scenario Generation
=================================================
Load a trained challenger checkpoint and generate N user scenarios
(instructions + expected actions) for a given tau2-bench domain.
Each generation is grounded in a fresh DB context sample.

Output format (matches TRL self-play intermediate.json):
    [
        {
            "domain": "airline",
            "iter": 1,
            "instructions": "You are a customer named ...",
            "actions": [{"name": "get_user_details", "arguments": {...}}, ...],
            "context": {"user": {...}, "reservation": {...}}
        },
        ...
    ]

Usage:
    python generate_challenger_scenarios.py \
        --model ./tod_zero/iter1_challenger/final \
        --domain airline \
        --n 1000 \
        --output ./tod_zero/iter1_scenarios.json
"""

import argparse
import hashlib
import json
import os
import re
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Generate challenger scenarios (DB-grounded)")
    p.add_argument("--model", required=True, help="Path to trained challenger model (HF or local)")
    p.add_argument("--domain", required=True, help="tau2-bench domain (retail, airline)")
    p.add_argument("--n", type=int, default=1000, help="Number of valid scenarios to generate")
    p.add_argument("--output", required=True, help="Output JSON file path")
    p.add_argument("--iter", type=int, default=1, help="Self-play iteration number (metadata)")
    p.add_argument("--batch_size", type=int, default=64, help="vLLM batch size")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per generation")
    p.add_argument("--max_model_len", type=int, default=8192, help="Max model context length")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

RE_INSTRUCTIONS = re.compile(r"<instructions>(.*?)</instructions>", re.DOTALL | re.IGNORECASE)
RE_ACTIONS_TAG  = re.compile(r"<actions>(.*?)</actions>",          re.DOTALL | re.IGNORECASE)


def _safe_json_loads(s: str):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_challenger_output(text: str):
    """Extract instructions + actions from model output.

    Returns None if parsing fails, otherwise:
        {"instructions": str, "actions": list[dict]}
    """
    instr_matches = RE_INSTRUCTIONS.findall(text)
    act_matches   = RE_ACTIONS_TAG.findall(text)

    if not instr_matches or not act_matches:
        return None

    instructions = instr_matches[-1].strip()
    actions = _safe_json_loads(act_matches[-1].strip())

    if len(instructions) < 20:
        return None
    if not isinstance(actions, list) or len(actions) == 0:
        return None
    if len(actions) > 10:
        return None

    # Validate each action has name + arguments
    parsed_actions = []
    for a in actions:
        if not isinstance(a, dict):
            return None
        name = a.get("name")
        arguments = a.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(arguments, dict):
            arguments = {}
        parsed_actions.append({"name": name.strip(), "arguments": arguments})

    return {"instructions": instructions, "actions": parsed_actions}


def validate_actions(actions: list, domain: str) -> tuple:
    """Validate action names are real domain tools. Returns (ok, reason)."""
    from agent_system.environments.env_package.tau2bench.db_sampler import DOMAIN_TOOLS
    valid_names = {t["name"] for t in DOMAIN_TOOLS.get(domain, [])}
    for a in actions:
        if a["name"] not in valid_names:
            return False, f"invalid_tool:{a['name']}"
    return True, "ok"


def fingerprint(actions: list) -> str:
    canon = json.dumps(actions, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(canon.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_scenarios_vllm(
    model_path: str,
    domain: str,
    n: int,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    iteration: int,
    seed: int,
) -> list:
    import random
    random.seed(seed)

    from vllm import LLM, SamplingParams
    from agent_system.environments.env_package.tau2bench.db_sampler import (
        sample_context, get_tools, get_policy,
        format_tools_for_prompt, format_context_for_prompt,
    )
    from agent_system.environments.prompts.tau2bench import (
        TAU2BENCH_CHALLENGER_SYSTEM,
        TAU2BENCH_CHALLENGER_USER,
        TAU2BENCH_CHALLENGER_TEMPLATE,
    )

    print(f"[generate] Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    policy     = get_policy(domain)
    tools      = get_tools(domain)
    tools_text = format_tools_for_prompt(tools)

    scenarios     = []
    seen_fps      = set()
    n_generated   = 0
    rejection_reasons: dict = {}

    print(f"[generate] Target: {n} valid scenarios (batch_size={batch_size})")

    while len(scenarios) < n:
        cur_batch = min(batch_size, (n - len(scenarios)) * 3)  # oversample 3x

        # Sample fresh DB context for each item → diversity
        contexts = [sample_context(domain) for _ in range(cur_batch)]

        # Use the same flat-text format as the training env (no chat template)
        # so the model sees identical prompts during training and generation.
        prompts = []
        for ctx in contexts:
            sys_prompt = TAU2BENCH_CHALLENGER_SYSTEM.format(
                domain=domain,
                policy=policy,
                tools_text=tools_text,
                context_text=format_context_for_prompt(ctx),
            )
            prompts.append(TAU2BENCH_CHALLENGER_TEMPLATE.format(
                system_prompt=sys_prompt,
                user_prompt=TAU2BENCH_CHALLENGER_USER,
            ))

        outputs = llm.generate(prompts, sampling_params)
        n_generated += len(outputs)

        for out, ctx in zip(outputs, contexts):
            raw = out.outputs[0].text if out.outputs else ""
            if not raw.strip():
                rejection_reasons["empty"] = rejection_reasons.get("empty", 0) + 1
                continue

            parsed = parse_challenger_output(raw)
            if parsed is None:
                rejection_reasons["parse_fail"] = rejection_reasons.get("parse_fail", 0) + 1
                continue

            ok, reason = validate_actions(parsed["actions"], domain)
            if not ok:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue

            fp = fingerprint(parsed["actions"])
            if fp in seen_fps:
                rejection_reasons["duplicate"] = rejection_reasons.get("duplicate", 0) + 1
                continue
            seen_fps.add(fp)

            scenarios.append({
                "domain": domain,
                "iter": iteration,
                "instructions": parsed["instructions"],
                "actions": parsed["actions"],
                "context": ctx,
            })

            if len(scenarios) >= n:
                break

        acceptance = len(scenarios) / max(1, n_generated)
        print(
            f"[generate] generated={n_generated} | valid={len(scenarios)}/{n} "
            f"(acceptance={acceptance:.1%})"
        )

        if n_generated > n * 30 and len(scenarios) < n * 0.1:
            print(f"[generate] WARNING: very low acceptance rate. Stopping with {len(scenarios)} scenarios.")
            break

    print("[generate] Rejection breakdown:")
    for reason, cnt in sorted(rejection_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason}: {cnt}")

    return scenarios


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"[generate] Domain: {args.domain}")
    print(f"[generate] Model:  {args.model}")
    print(f"[generate] Target: {args.n} scenarios → {args.output}")

    scenarios = generate_scenarios_vllm(
        model_path=args.model,
        domain=args.domain,
        n=args.n,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        iteration=args.iter,
        seed=args.seed,
    )

    with open(args.output, "w") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)

    print(f"[generate] Saved {len(scenarios)} scenarios to {args.output}")
    print("\n[generate] Sample scenarios:")
    for s in scenarios[:3]:
        print(f"  instructions: {s['instructions'][:120]}...")
        print(f"  actions: {[a['name'] for a in s['actions']]}")
        print()


if __name__ == "__main__":
    main()
