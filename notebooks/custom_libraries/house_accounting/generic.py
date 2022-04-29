from datetime import datetime
from typing import Optional
from collections.abc import Iterator

from .enumerators import MainCategory, SubCategory, TimeCategory


class Cashflow:
    def __init__(
        self,
        timestamp: datetime,
        amount: float,
        sub_category: SubCategory,
        time_category: TimeCategory,
        tags: Optional[Iterator[str]] = [],
        description: Optional[str] = "",
    ) -> None:

        self.timestamp = timestamp
        self.amount = amount
        self.sub_category = sub_category
        self.time_category = time_category
        self.tags = tags
        self.description = description

    def add_tag(self, tag: str):
        self.tags.append(tag)
