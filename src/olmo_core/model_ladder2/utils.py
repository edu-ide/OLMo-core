def format_tokens(tokens: int) -> str:
    """Format number of tokens into a human-readable string."""
    if tokens < 1_000:
        return f"{tokens} kokens"
    elif tokens < 1_000_000:
        return f"{tokens / 1_000:.1f}K tokens"
    elif tokens < 1_000_000_000:
        return f"{tokens / 1_000_000:.1f}M tokens"
    elif tokens < 1_000_000_000_000:
        return f"{tokens / 1_000_000_000:.1f}B tokens"
    else:
        return f"{tokens / 1_000_000_000_000:.1f}T tokens"
