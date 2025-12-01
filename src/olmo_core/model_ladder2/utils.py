def format_count(count: int) -> str:
    """Format a large count into a human-readable string."""
    if count < 1_000:
        return f"{count}"
    elif count < 1_000_000:
        return f"{count / 1_000:.1f}K"
    elif count < 1_000_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count < 1_000_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    else:
        return f"{count / 1_000_000_000_000:.1f}T"


def format_tokens(tokens: int) -> str:
    """Format number of tokens into a human-readable string."""
    return f"{format_count(tokens)} tokens"
