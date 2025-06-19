from config import USE_EMOJI


def emoji_label(icon: str, label: str) -> str:
    """Return a label optionally prefixed with an emoji icon."""
    return f"{icon} {label}" if USE_EMOJI else label
