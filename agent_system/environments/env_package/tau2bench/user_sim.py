"""
Lightweight user simulator for tau2-bench solver environment.

Calls an external LLM API (vLLM-compatible OpenAI endpoint) to generate
user responses. The user simulator is FIXED (no training) throughout.
"""

import json
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


class LightUserSimulator:
    """Lightweight user simulator using an external OpenAI-compatible API."""

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

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_url, api_key="EMPTY")
            self._resolve_model_name()
        return self._client

    def _resolve_model_name(self):
        """Auto-discover the served model name if the configured one isn't available."""
        try:
            models = self._client.models.list()
            available = [m.id for m in models.data]
            if self.model not in available and available:
                print(
                    f"[UserSim] Model '{self.model}' not found on server. "
                    f"Available: {available}. Using '{available[0]}'."
                )
                self.model = available[0]
        except Exception:
            pass  # server may not support /v1/models; keep configured name

    def reset(self, user_instructions: str):
        """Reset with new instructions."""
        self.system_prompt = USER_SIM_SYSTEM_PROMPT.format(
            instructions=user_instructions,
        )
        self.history = []

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
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            first_msg = resp.choices[0].message.content or "Hi, I need help with something."
        except Exception as e:
            print(f"[UserSim] Error generating first message: {e}")
            first_msg = "Hi, I need help with something."

        # Store as assistant message (user sim's role)
        self.history.append({"role": "assistant", "content": first_msg})
        return first_msg

    def respond(self, agent_message: str) -> str:
        """Generate user's response to the agent's message.

        Args:
            agent_message: The agent's text response

        Returns:
            The user simulator's reply
        """
        # Agent's message is "user" from user sim's perspective
        self.history.append({"role": "user", "content": agent_message})

        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self.history

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            user_reply = resp.choices[0].message.content or "[STOP]"
        except Exception as e:
            print(f"[UserSim] Error generating response: {e}")
            user_reply = "[STOP]"

        self.history.append({"role": "assistant", "content": user_reply})
        return user_reply

    def is_stop(self, message: str) -> bool:
        """Check if message indicates conversation should end."""
        return any(sig in message for sig in ("[STOP]", "[TRANSFER]", "[OUT_OF_SCOPE]"))
