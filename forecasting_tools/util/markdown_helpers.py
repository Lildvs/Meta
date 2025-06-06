def clean_markdown(text: str) -> str:
    """Removes code block specifiers for cleaner display."""
    text = text.replace("```json", "```")
    text = text.replace("```python", "```")
    return text