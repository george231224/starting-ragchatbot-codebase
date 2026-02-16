import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import anthropic
from openai import OpenAI


@dataclass
class ToolCall:
    """Normalized tool call across providers."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class NormalizedResponse:
    """Normalized LLM response across providers."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    has_tool_use: bool = False


class AIGenerator:
    """Handles interactions with LLM APIs for generating responses.

    Supports two providers:
    - "anthropic": Direct Anthropic SDK
    - "openrouter": OpenAI-compatible API (OpenRouter)
    """

    MAX_TOOL_ROUNDS = 2

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Course outline/structure questions**: Use the get_course_outline tool to retrieve the full lesson list
- When returning outline results, include the course title, course link, and list every lesson with its number and title
- **Up to two sequential tool calls per query** - you may refine or follow up on initial search results with a second tool call if needed
- Use the minimum number of tool calls necessary
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ):
        self.provider = provider
        self.model = model

        if provider == "openrouter":
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=base_url or "https://openrouter.ai/api/v1",
            )
        else:
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.anthropic_client = anthropic.Anthropic(**client_kwargs)

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """Generate AI response with optional tool usage and conversation context."""

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        if self.provider == "openrouter":
            return self._generate_openai(
                query, system_content, tools, tool_manager
            )
        else:
            return self._generate_anthropic(
                query, system_content, tools, tool_manager
            )

    # ── Anthropic path ──────────────────────────────────────────────

    def _generate_anthropic(
        self, query: str, system_content: str, tools, tool_manager
    ) -> str:
        messages = [{"role": "user", "content": query}]
        base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

        for _ in range(self.MAX_TOOL_ROUNDS):
            api_params = {
                **base_params,
                "messages": messages,
                "system": system_content,
            }
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self.anthropic_client.messages.create(**api_params)

            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_anthropic_text(response)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = self._execute_anthropic_tool_calls(
                response, tool_manager
            )
            messages.append({"role": "user", "content": tool_results})

        # All rounds exhausted — final call without tools
        final_params = {
            **base_params,
            "messages": messages,
            "system": system_content,
        }
        return self._extract_anthropic_text(
            self.anthropic_client.messages.create(**final_params)
        )

    def _execute_anthropic_tool_calls(self, response, tool_manager) -> list:
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                except Exception as e:
                    result = f"Error: {e}"
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
        return tool_results

    def _extract_anthropic_text(self, response) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    # ── OpenAI-compatible path (OpenRouter) ─────────────────────────

    def _generate_openai(
        self, query: str, system_content: str, tools, tool_manager
    ) -> str:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        openai_tools = self._convert_tools_to_openai(tools) if tools else None

        for _ in range(self.MAX_TOOL_ROUNDS):
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 800,
            }
            if openai_tools:
                kwargs["tools"] = openai_tools
                kwargs["tool_choice"] = "auto"

            response = self.openai_client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            if choice.finish_reason != "tool_calls" or not tool_manager:
                return choice.message.content or ""

            # Append assistant message with tool calls
            messages.append(choice.message)

            # Execute each tool call and append results
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                try:
                    result = tool_manager.execute_tool(tc.function.name, **args)
                except Exception as e:
                    result = f"Error: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        # All rounds exhausted — final call without tools
        final_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }
        final_response = self.openai_client.chat.completions.create(**final_kwargs)
        return final_response.choices[0].message.content or ""

    # ── Format conversion helpers ───────────────────────────────────

    @staticmethod
    def _convert_tools_to_openai(
        anthropic_tools: List[Dict],
    ) -> List[Dict]:
        """Convert Anthropic tool definitions to OpenAI function-calling format."""
        openai_tools = []
        for tool in anthropic_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        return openai_tools
