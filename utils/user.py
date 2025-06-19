from dataclasses import dataclass
from config import PLANS


@dataclass
class User:
    """Simple user representation with quota tracking."""

    name: str = "Guest"
    plan: str = "free"
    char_usage: int = 0

    @property
    def char_limit(self) -> int:
        return PLANS.get(self.plan, 0)

    def has_quota(self, additional_chars: int = 0) -> bool:
        limit = self.char_limit
        if limit == 0:
            return False
        return (self.char_usage + additional_chars) <= limit
