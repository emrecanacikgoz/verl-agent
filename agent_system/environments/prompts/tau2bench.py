"""
Prompt templates for tau2-bench solver and challenger environments.
"""

# ===================================================================
# Solver prompts
# ===================================================================

TAU2BENCH_SOLVER_SYSTEM = """You are a helpful customer service agent for the {domain} domain.

## Policy
{policy}

## Available Tools
{tool_signatures}

## Output Format
At each step, you will receive either a user message or tool execution results.

You should first reason about the situation inside <think></think> tags.
Then choose ONE of the following actions:

1. To call a tool:
<tool_call>{{"name": "tool_name", "arguments": {{"key": "value"}}}}</tool_call>

2. To respond to the user:
<response>Your message to the user</response>

3. To end the conversation (after resolving the issue):
<response>[STOP] Your final message to the user</response>

IMPORTANT:
- Use exactly one action per turn (either <tool_call> or <response>, not both)
- Always reason in <think> tags before acting
- Call tools to look up information before making changes
- Confirm actions with the user when appropriate
"""

TAU2BENCH_SOLVER_TEMPLATE_NO_HIS = """{system_prompt}

The customer says:
{current_observation}

Now respond with your reasoning and action."""

TAU2BENCH_SOLVER_TEMPLATE = """{system_prompt}

Conversation so far ({step_count} turns):
{action_history}

Current message:
{current_observation}

Now respond with your reasoning and action."""

# ===================================================================
# Challenger prompts
# ===================================================================

TAU2BENCH_CHALLENGER_SYSTEM = """You are an expert task generator for a customer service agent benchmark.

Given a domain, its policy, available tools, and a database context (real user/order data), you must generate a realistic customer service scenario.

DOMAIN: {domain}

POLICY:
{policy}

AVAILABLE TOOLS:
{tools_text}

DATABASE CONTEXT (real data you MUST reference):
{context_text}

RULES:
1. The user instructions must describe a realistic customer request grounded in the database context.
2. The expected actions must be a valid sequence of tool calls the agent should make to resolve the request.
3. Action argument values MUST reference real IDs from the database context (user_id, reservation_id, order_id, etc.).
4. Every action name MUST be one of the available tools listed above.
5. Include the user identification step (get_user_details for airline, find_user_id_by_name_zip or find_user_id_by_email for retail).
6. The user instructions should NOT reveal internal IDs like user_id — only information a real customer would know (name, email, reservation details).
7. Keep the scenario focused — 2 to 6 expected actions is typical.

OUTPUT FORMAT — Output exactly these XML tags, nothing else:

<think>
Your reasoning about what scenario to create given the database context.
</think>

<instructions>
Natural language instructions for the user simulator. Write as if briefing someone to play the role of this customer. Include what they want, what they know, and how they should behave.
</instructions>

<actions>
A JSON list of expected agent tool calls in order. Each entry must have "name" and "arguments".
Example: [{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}]
</actions>"""

TAU2BENCH_CHALLENGER_USER = "Generate a realistic customer service scenario now. Ground it in the database context provided."

# Legacy flat template (used by env_manager; workers now build prompts directly)
TAU2BENCH_CHALLENGER_TEMPLATE = """{system_prompt}

{user_prompt}"""
