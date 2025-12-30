"""Utility functions for parsing LLM responses."""

import json
from typing import Any


def extract_json_from_response(response: str) -> dict[str, Any]:
    """
    Extract and parse JSON from an LLM response.

    Handles cases where the LLM wraps JSON in markdown code blocks or adds extra text.

    Args:
        response: Raw response string from LLM

    Returns:
        Parsed JSON as dictionary

    Raises:
        json.JSONDecodeError: If no valid JSON is found in the response
    """
    # Try to extract JSON from markdown code blocks
    if "```json" in response:
        json_str = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        json_str = response.split("```")[1].split("```")[0].strip()
    else:
        json_str = response.strip()

    return json.loads(json_str)
