from datetime import datetime

# Token keys aligned with OpenAI/litellm response format
TOKEN_KEYS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]

COST_KEYS = [
    "prompt",
    "completion",
    "total",
]



def _merge_statistics(metadata: dict) -> dict:
    """
    Merge turn-level statistics into session-level statistics.

    Args:
        metadata (dict): The metadata with turn-level statistics

    Returns:
        dict: Updated metadata with merged statistics
    """
    # Aggregate statistics from all turns
    for turn in metadata["turns"]:
        metadata["total_cost"] += turn.get("total_cost", 0) or 0
        for key in TOKEN_KEYS:
            metadata["total_tokens"][key] += turn["combined_metadata"].get(key, 0) or 0

        for key in TOKEN_KEYS:
            metadata["total_tokens_query"][key] += (
                turn["query_metadata"].get(key, 0) or 0
            )

        if "retrieval_metadata" in turn:
            rm = turn["retrieval_metadata"]
            for key in TOKEN_KEYS:
                metadata["total_tokens_retrieval"][key] += rm.get(key, 0) or 0

        metadata["error_count"] += len(turn.get("errors", []))

        # Aggregate tool usage
        for tool_call in turn.get("tool_calls", []):
            tool_name = tool_call["tool_name"]
            if tool_name not in metadata["tool_usage"]:
                metadata["tool_usage"][tool_name] = 0
            metadata["tool_usage"][tool_name] += 1
            metadata["tool_calls_count"] += 1

    # Calculate total duration
    if metadata["start_time"] and metadata["end_time"]:
        start = datetime.fromisoformat(metadata["start_time"])
        end = datetime.fromisoformat(metadata["end_time"])
        metadata["total_duration_seconds"] = (end - start).total_seconds()

    return metadata
