"""
Ibis + DuckDB + pandas Pipeline for Semantic Query Processing.

This pipeline implements semantic operations as first-class citizens in a
planner-executor architecture with:
- Ibis as the logical IR for both relational and semantic operators
- DuckDB as the high-performance relational executor
- pandas as the semantic boundary contract for SDPS backends
"""

from .operators import (
    SemOp, SemFilter, SemProject, SemJoin, SemGroupBy,
    SemAgg, SemDistinct, SemOrderBy, SemIntersect, SemExcept
)
from .logical_plan import LogicalPlan, PlanNode, RelationalNode, SemanticNode
from .optimizer import PlanOptimizer, OptimizationRule
from .segmenter import PlanSegmenter, Segment, RelationalSegment, SemanticSegment
from .bridge import BridgeOperator, ToPandas, FromPandas
from .executor import PipelineExecutor
from .parser import IbisSQLParser

__all__ = [
    'SemOp', 'SemFilter', 'SemProject', 'SemJoin', 'SemGroupBy',
    'SemAgg', 'SemDistinct', 'SemOrderBy', 'SemIntersect', 'SemExcept',
    'LogicalPlan', 'PlanNode', 'RelationalNode', 'SemanticNode',
    'PlanOptimizer', 'OptimizationRule',
    'PlanSegmenter', 'Segment', 'RelationalSegment', 'SemanticSegment',
    'BridgeOperator', 'ToPandas', 'FromPandas',
    'PipelineExecutor',
    'IbisSQLParser',
]
