"""
Lightweight user simulator for tau2-bench solver environment.

Calls an external LLM API (vLLM-compatible OpenAI endpoint) to generate
user responses. The user simulator is FIXED (no training) throughout.
"""

import json
import time
from typing import Any, Dict, List, Optional


# Default system prompt template for user simulation
USER_SIM_SYSTEM_PROMPT = """You are simulating a customer who is contacting customer service.
You must follow the scenario below and act as this customer would.

IMPORTANT RULES:
- Stay in character and follow the scenario instructions precisely
- Respond naturally as a real customer would
- Only reveal information that the scenario says you know
- Do not reveal information the scenario says is unknown to you
- If the agent has fully resolved your issue, respond with exactly: [STOP]
- If you are being transferred to another department, respond with: [TRANSFER]
- If the agent says your request is out of scope, respond with: [STOP]
- Keep responses concise and natural (1-3 sentences typically)
- Do not make up information not in the scenario

<scenario>
{instructions}
</scenario>"""

# Number of consecutive failures before raising an error during training
_MAX_CONSECUTIVE_FAILURES = 3


def check_user_sim_connection(api_url: str, timeout: float = 10.0) -> bool:
    """Check if the user sim server is reachable. Returns True if healthy."""
    try:
        import urllib.request
        health_url = api_url.rstrip("/").replace("/v1", "") + "/health"
        req = urllib.request.urlopen(health_url, timeout=timeout)
        return req.status == 200
    except Exception:
        return False


class LightUserSimulator:
    """Lightweight user simulator using an external OpenAI-compatible API."""

    # Class-level cache: api_url -> resolved model name (avoids per-instance API calls + log spam)
    _resolved_model_cache: Dict[str, str] = {}

    def __init__(
        self,
        api_url: str,
        model: str,
        user_instructions: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.system_prompt = USER_SIM_SYSTEM_PROMPT.format(
            instructions=user_instructions,
        )

        # Conversation history (from user sim's perspective)
        # user sim's messages are "assistant", agent's messages are "user"
        self.history: List[Dict[str, str]] = []
        self._client = None
        self._consecutive_failures = 0

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_url, api_key="EMPTY")
            self._resolve_model_name()
        return self._client

    def _resolve_model_name(self):
        """Auto-discover the served model name if the configured one isn't available.

        Results are cached at the class level so that parallel workers sharing the
        same API URL only hit the server once and only print the warning once.
        """
        if self.api_url in LightUserSimulator._resolved_model_cache:
            self.model = LightUserSimulator._resolved_model_cache[self.api_url]
            return
        try:
            models = self._client.models.list()
            available = [m.id for m in models.data]
            if self.model not in available and available:
                print(
                    f"[UserSim] Model '{self.model}' not found on server. "
                    f"Available: {available}. Using '{available[0]}'."
                )
                self.model = available[0]
            LightUserSimulator._resolved_model_cache[self.api_url] = self.model
        except Exception as e:
            raise RuntimeError(
                f"[UserSim] FATAL: Cannot connect to user simulator at {self.api_url}. "
                f"Error: {e}\n"
                f"Make sure the vLLM user sim server is running:\n"
                f"  python -m vllm.entrypoints.openai.api_server "
                f"--model <model> --port 8000 &"
            )

    def reset(self, user_instructions: str):
        """Reset with new instructions."""
        self.system_prompt = USER_SIM_SYSTEM_PROMPT.format(
            instructions=user_instructions,
        )
        self.history = []
        self._consecutive_failures = 0

    def _call_api(self, messages: List[Dict]) -> str:
        """Call the API with failure tracking. Raises after too many consecutive failures."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self._consecutive_failures = 0
            return resp.choices[0].message.content or ""
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"[UserSim] FATAL: {_MAX_CONSECUTIVE_FAILURES} consecutive API failures "
                    f"at {self.api_url}. Last error: {e}\n"
                    f"All rollout conversations will produce [STOP] immediately — "
                    f"training data is garbage. Check the user sim server."
                ) from e
            print(
                f"[UserSim] WARNING: API call failed (failure {self._consecutive_failures}/"
                f"{_MAX_CONSECUTIVE_FAILURES}): {e}"
            )
            raise

    @staticmethod
    def _is_context_length_error(e: Exception) -> bool:
        """Return True if the exception is a context-length / token-limit error."""
        msg = str(e).lower()
        return "context length" in msg or "max_tokens" in msg or "input tokens" in msg or "reduce the length" in msg

    def generate_first_message(self) -> str:
        """Generate the user's opening message to start the conversation."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Generate your opening message as the customer. "
                    "Introduce yourself and state what you need help with. "
                    "Keep it natural and concise."
                ),
            },
        ]
        try:
            first_msg = self._call_api(messages)
            if not first_msg:
                first_msg = "Hi, I need help."
        except RuntimeError:
            raise
        except Exception:
            first_msg = "Hi, I need help."

        self.history.append({"role": "assistant", "content": first_msg})
        return first_msg

    @staticmethod
    def _parse_overshoot_tokens(e: Exception) -> Optional[int]:
        """Try to extract how many tokens over the limit we are from the error message."""
        import re
        msg = str(e)
        has_match = re.search(r"has (\d+) input tokens", msg)
        max_match = re.search(r"maximum context length is (\d+) tokens", msg)
        if has_match and max_match:
            input_tokens = int(has_match.group(1))
            max_context = int(max_match.group(1))
            overshoot = input_tokens - max_context
            # For "max_tokens too large" errors, input fits but input + max_completion > limit.
            # In that case overshoot is negative; add max_tokens (256) to get the real gap.
            max_tok_match = re.search(r"too large: (\d+)", msg)
            if overshoot < 0 and max_tok_match:
                overshoot = input_tokens + int(max_tok_match.group(1)) - max_context
            return max(overshoot, 1)  # always positive so caller drops at least a few turns
        return None

    def respond(self, agent_message: str) -> str:
        """Generate user's response to the agent's message."""
        self.history.append({"role": "user", "content": agent_message})

        # Try with progressively shorter history if we hit context length limits.
        history_view = list(self.history)
        while True:
            messages = [{"role": "system", "content": self.system_prompt}] + history_view
            try:
                user_reply = self._call_api(messages)
                if not user_reply:
                    user_reply = "[STOP]"
                break
            except RuntimeError:
                raise
            except Exception as e:
                if self._is_context_length_error(e) and len(history_view) > 2:
                    # Reset the failure counter so context-length retries don't count.
                    self._consecutive_failures = 0

                    # Estimate how many turns to drop based on the overshoot.
                    # Average ~80 tokens per turn; drop enough to clear the overshoot
                    # plus a buffer, with a minimum of 2 turns.
                    overshoot = self._parse_overshoot_tokens(e)
                    if overshoot is not None and overshoot > 0:
                        avg_tokens_per_turn = max(
                            1, sum(len(m["content"]) for m in history_view) // (4 * max(len(history_view), 1))
                        )
                        # Add 20% buffer to avoid repeated retries
                        turns_to_drop = max(2, int((overshoot * 1.2) / avg_tokens_per_turn))
                        # Round up to even number (user+assistant pairs)
                        turns_to_drop += turns_to_drop % 2
                    else:
                        # Fallback: drop half the history
                        turns_to_drop = max(2, (len(history_view) // 2) & ~1)

                    # Don't drop more than we have (keep at least 2 turns)
                    turns_to_drop = min(turns_to_drop, len(history_view) - 2)
                    history_view = history_view[turns_to_drop:]
                    print(
                        f"[UserSim] Context too long (overshoot={overshoot}), "
                        f"dropped {turns_to_drop} turns. "
                        f"History size: {len(history_view)} turns."
                    )
                    continue
                # Non-context-length API error: end episode with a sentinel that
                # signals a failed stop (not genuine user satisfaction) so the
                # environment can assign 0 reward instead of USER_STOP reward.
                user_reply = "[API_FAIL]"
                break

        self.history.append({"role": "assistant", "content": user_reply})
        return user_reply

    def is_stop(self, message: str) -> bool:
        """Check if message indicates conversation should end."""
        return any(sig in message for sig in ("[STOP]", "[TRANSFER]", "[OUT_OF_SCOPE]", "[API_FAIL]"))

    def is_api_fail(self, message: str) -> bool:
        """Check if the stop was caused by an API failure (not genuine user satisfaction)."""
        return "[API_FAIL]" in message
