"""
LiteLLM Wrapper for Financial Agent.

Provides a unified interface for LLM calls using litellm,
replacing the model_library dependency.
"""

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import litellm


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    name: str
    args: dict
    id: str


@dataclass
class ToolResult:
    """Represents the result of a tool call."""
    tool_call: ToolCall
    result: str


@dataclass
class TextInput:
    """Represents a text input message."""
    text: str
    role: str = "user"


@dataclass
class CostInfo:
    """Token cost information."""
    prompt: float = 0.0
    completion: float = 0.0
    total: float = 0.0


@dataclass
class Metadata:
    """Response metadata."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: CostInfo = field(default_factory=CostInfo)
    
    def model_dump(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": {
                "prompt": self.cost.prompt,
                "completion": self.cost.completion,
                "total": self.cost.total,
            }
        }


@dataclass
class QueryResult:
    """Result from an LLM query."""
    output_text: Optional[str]
    reasoning: Optional[str]
    tool_calls: List[ToolCall]
    history: List[Any]
    metadata: Metadata
    
    @property
    def output_text_str(self) -> str:
        return self.output_text or ""


@dataclass
class ToolDefinition:
    """OpenAI-compatible tool definition."""
    name: str
    description: str
    parameters: dict
    
    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters.get("properties", {}),
                    "required": self.parameters.get("required", []),
                }
            }
        }


class MaxContextWindowExceededError(Exception):
    """Raised when the context window is exceeded."""
    pass


class LiteLLMWrapper:
    """
    Wrapper around litellm for agent LLM calls.
    
    Provides async query interface with tool calling support.
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 32000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._registry_key = model  # For compatibility
        self.logger = None  # Will be set by agent
    
    def _convert_messages(self, input_items: List[Any]) -> List[dict]:
        """Convert internal message types to OpenAI format."""
        messages = []
        
        for item in input_items:
            if isinstance(item, TextInput):
                messages.append({
                    "role": item.role,
                    "content": item.text
                })
            elif isinstance(item, ToolResult):
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.tool_call.id,
                    "content": item.result
                })
            elif isinstance(item, dict):
                # Already in dict format (from history)
                messages.append(item)
            elif hasattr(item, 'role'):
                # RawResponse or similar - convert to dict
                if hasattr(item, 'content'):
                    msg = {"role": item.role, "content": item.content}
                    if hasattr(item, 'tool_calls') and item.tool_calls:
                        msg["tool_calls"] = item.tool_calls
                    messages.append(msg)
        
        return messages
    
    def _convert_tools(self, tools: List[ToolDefinition]) -> List[dict]:
        """Convert tool definitions to OpenAI format."""
        return [tool.to_openai_format() for tool in tools]
    
    async def query(
        self,
        input: Any,
        tools: Optional[List[ToolDefinition]] = None
    ) -> QueryResult:
        """
        Query the LLM with optional tool calling.
        
        Args:
            input: Either a string or list of message items
            tools: Optional list of tool definitions
            
        Returns:
            QueryResult with response and metadata
        """
        # Handle string input (simple query)
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        else:
            messages = self._convert_messages(input)
        
        # Prepare kwargs
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            kwargs["tool_choice"] = "auto"
        
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "context" in error_str and ("length" in error_str or "window" in error_str or "exceeded" in error_str):
                raise MaxContextWindowExceededError(str(e))
            raise
        
        # Extract response content
        choice = response.choices[0]
        message = choice.message
        
        output_text = message.content
        reasoning = None  # litellm doesn't separate reasoning
        
        # Extract tool calls
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    args=args,
                    id=tc.id
                ))
        
        # Build history with assistant response
        history = list(messages)
        assistant_msg = {"role": "assistant", "content": output_text}
        if hasattr(message, 'tool_calls') and message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        history.append(assistant_msg)
        
        # Calculate metadata
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        
        # Estimate cost (rough estimate)
        cost = CostInfo(
            prompt=prompt_tokens * 0.000003,  # ~$3/1M tokens
            completion=completion_tokens * 0.000015,  # ~$15/1M tokens
            total=prompt_tokens * 0.000003 + completion_tokens * 0.000015
        )
        
        metadata = Metadata(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost
        )
        
        return QueryResult(
            output_text=output_text,
            reasoning=reasoning,
            tool_calls=tool_calls,
            history=history,
            metadata=metadata
        )
