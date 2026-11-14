"""
THE QUERY PLAN - the only thing a model (or a rule) is allowed to produce.

A plan is plain JSON data, never code. It looks like this:

  {"intent": "aggregate", "dataset": "orders_wide",
   "filters": [{"field": "region", "op": "eq", "value": "West"}],
   "group_by": ["category"],
   "metrics": [{"agg": "named", "name": "return_rate_pct", "as": "return_rate"}],
   "sort": [{"field": "return_rate", "dir": "desc"}], "limit": 5}

Safety by design
  * extra="forbid": any key we did not define (like "sql", "join", "code") is rejected.
  * Every operator, aggregation, intent and named metric is on a fixed list below.
  * Hard limits stop expensive or huge requests.
  * Joins cannot be written in a plan. The plan can only pick a ready-made, pre-checked
    view such as "orders_wide" (see catalog.py), so a fan-out join is impossible to request.
"""
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt

# ----------------------------------------------------------------- limits
MAX_LIMIT = 100             # rows returned
DEFAULT_LIMIT = 10
MAX_FILTERS = 8
MAX_GROUP_BY = 3
MAX_METRICS = 6
MAX_SORT = 3
MAX_SELECT = 8
MAX_IN_VALUES = 50
MAX_TEXT_VALUE_LEN = 100
MAX_GROUPS = 10_000          # estimated number of result groups
QUERY_SECONDS = 10.0         # soft time budget, checked between steps

# ------------------------------------------------------------ allow-lists
INTENTS = ("aggregate", "list", "distinct", "describe")
FILTER_OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "between", "contains", "is_null", "not_null")
AGGS = ("sum", "mean", "median", "min", "max", "count", "count_distinct", "named")

# named retail metrics (computed by app/retail/metrics.py) and the columns each needs
NAMED_METRICS = {
    "revenue": ["revenue"],
    "orders": ["revenue"],
    "units": ["revenue", "quantity"],
    "aov": ["revenue"],
    "returned_orders": ["revenue", "returned"],
    "return_rate_pct": ["revenue", "returned"],
    "returned_revenue": ["revenue", "returned"],
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Filter(_Strict):
    field: str = Field(min_length=1, max_length=64)
    op: Literal[FILTER_OPS]
    value: Any = None


class Metric(_Strict):
    agg: Literal[AGGS]
    field: Optional[str] = Field(default=None, max_length=64)
    name: Optional[str] = Field(default=None, max_length=32)       # only for agg="named"
    as_: Optional[str] = Field(default=None, alias="as", max_length=32)


class Sort(_Strict):
    field: str = Field(min_length=1, max_length=64)
    dir: Literal["asc", "desc"] = "desc"


class TimeBucket(_Strict):
    field: str = Field(min_length=1, max_length=64)
    unit: Literal["month", "year"]


class Plan(_Strict):
    intent: Literal[INTENTS]
    dataset: str = Field(min_length=1, max_length=64)
    filters: List[Filter] = []
    group_by: List[str] = []
    metrics: List[Metric] = []
    select: List[str] = []
    sort: List[Sort] = []
    limit: StrictInt = DEFAULT_LIMIT
    time_bucket: Optional[TimeBucket] = None
