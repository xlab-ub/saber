"""
Semantic Operators as First-Class Ibis Operations.

Each semantic operator is a logical node with well-defined:
- Schema behavior (input/output columns)
- Relational properties (order-preserving, blocking, etc.)
- Semantic kernel signature (prompt, model, threshold, etc.)
- Cost estimation hooks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import pandas as pd


class DeterminismClass(Enum):
    """Classification of semantic operator determinism."""
    DETERMINISTIC = "deterministic"  # Embedding similarity with fixed model
    STOCHASTIC_BOUNDED = "stochastic_bounded"  # LLM with temp>0 but stable prompt
    STOCHASTIC_UNBOUNDED = "stochastic_unbounded"  # Tool calls, external state


@dataclass
class CostEstimate:
    """Cost estimate for a semantic operation."""
    cpu_time_ms: float = 0.0
    expected_selectivity: float = 1.0
    llm_tokens: int = 0
    llm_calls: int = 0
    embedding_calls: int = 0
    retrieval_complexity: float = 1.0


@dataclass
class SemanticKernelSignature:
    """Signature for semantic kernel execution."""
    prompt_template: str = ""
    model: Optional[str] = None
    embedding_space: Optional[str] = None
    threshold: float = 0.85
    top_k: Optional[int] = None
    temperature: float = 0.0


@dataclass 
class SemOp(ABC):
    """Base class for semantic operators."""
    
    backend: str = "lotus"
    determinism: DeterminismClass = DeterminismClass.STOCHASTIC_BOUNDED
    kernel: SemanticKernelSignature = field(default_factory=SemanticKernelSignature)
    cost_estimate: CostEstimate = field(default_factory=CostEstimate)
    
    # Relational properties
    order_preserving: bool = False
    duplicate_preserving: bool = True
    blocking: bool = False
    
    # Limit pushdown hint for early termination
    limit_hint: Optional[int] = None
    
    @abstractmethod
    def get_op_type(self) -> str:
        """Return the operation type string."""
        pass
    
    @abstractmethod
    def get_required_columns(self) -> Set[str]:
        """Return columns required from input."""
        pass
    
    @abstractmethod
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        """Return output column list given input columns."""
        pass
    
    @abstractmethod
    def execute(self, df: pd.DataFrame, backend_impl: Any) -> pd.DataFrame:
        """Execute the semantic operation on a DataFrame."""
        pass
    
    def get_cache_key(self, input_fingerprint: str) -> str:
        """Generate cache key for this operation."""
        return f"{self.get_op_type()}:{self.backend}:{self.kernel.prompt_template}:{input_fingerprint}"


@dataclass
class SemFilter(SemOp):
    """Semantic WHERE operation."""
    
    prompt: str = ""
    
    def __post_init__(self):
        self.kernel = SemanticKernelSignature(prompt_template=self.prompt)
        self.cost_estimate = CostEstimate(expected_selectivity=0.5)  # Prior
    
    def get_op_type(self) -> str:
        return "where"
    
    def get_required_columns(self) -> Set[str]:
        import re
        return set(re.findall(r'\{([^}]+)\}', self.prompt))
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        return input_columns  # Filter preserves columns
    
    def execute(self, df: pd.DataFrame, backend_impl: Any) -> pd.DataFrame:
        # Pass limit_hint to backend for early termination
        return backend_impl.sem_where(df, self.prompt, limit_hint=self.limit_hint)


@dataclass
class SemFilterMarker(SemOp):
    """Semantic WHERE used as a marker (adds boolean column instead of filtering).
    
    Used when SEM_WHERE appears in SELECT clause (e.g., CASE WHEN SEM_WHERE(...)).
    Preserves all rows and adds a _sem_where_result column with True/False.
    """
    
    prompt: str = ""
    result_column: str = "_sem_where_result"
    
    def __post_init__(self):
        self.kernel = SemanticKernelSignature(prompt_template=self.prompt)
        self.cost_estimate = CostEstimate(expected_selectivity=1.0)  # Preserves all rows
    
    def get_op_type(self) -> str:
        return "where_marker"
    
    def get_required_columns(self) -> Set[str]:
        import re
        return set(re.findall(r'\{([^}]+)\}', self.prompt))
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        return input_columns + [self.result_column]  # Adds marker column
    
    def execute(self, df: pd.DataFrame, backend_impl: Any) -> pd.DataFrame:
        # Use sem_where_marker to get boolean results without filtering
        return backend_impl.sem_where_marker(df, self.prompt, self.result_column)


@dataclass
class SemProject(SemOp):
    """Semantic SELECT/projection operation."""
    
    prompt: str = ""
    alias: str = "extracted"
    
    def __post_init__(self):
        self.kernel = SemanticKernelSignature(prompt_template=self.prompt)
    
    def get_op_type(self) -> str:
        return "select"
    
    def get_required_columns(self) -> Set[str]:
        import re
        return set(re.findall(r'\{([^}]+)\}', self.prompt))
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        return input_columns + [self.alias]
    
    def execute(self, df: pd.DataFrame, backend_impl: Any) -> pd.DataFrame:
        return backend_impl.sem_select(df, self.prompt, self.alias)


@dataclass
class SemJoin(SemOp):
    """Semantic JOIN operation."""
    
    prompt: str = ""
    left_table: str = ""
    right_table: str = ""
    left_alias: str = ""  # Alias for left table in output columns
    right_alias: str = ""  # Alias for right table in output columns
    
    def __post_init__(self):
        self.kernel = SemanticKernelSignature(prompt_template=self.prompt)
        self.blocking = True  # Join is blocking
        # Default aliases to table names if not set
        if not self.left_alias:
            self.left_alias = self.left_table
        if not self.right_alias:
            self.right_alias = self.right_table
    
    def get_op_type(self) -> str:
        return "join"
    
    def get_required_columns(self) -> Set[str]:
        import re
        return set(re.findall(r'\{([^}:]+)(?::[^}]+)?\}', self.prompt))
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        # Join produces prefixed columns from both tables
        return input_columns  # Actual columns determined at runtime
    
    def execute(self, df1: pd.DataFrame, df2: pd.DataFrame, backend_impl: Any) -> pd.DataFrame:
        return backend_impl.sem_join(df1, df2, self.prompt, self.left_table, self.right_table, 
                                      limit_hint=self.limit_hint)


@dataclass
class SemGroupBy(SemOp):
    """Semantic GROUP BY operation."""
    
    column: str = ""
    k: int = 8  # Number of clusters
    
    def __post_init__(self):
        self.kernel = SemanticKernelSignature()
        self.blocking = True
    
    def get_op_type(self) -> str:
        return "groupby"
    
    def get_required_columns(self) -> Set[str]:
        return {self.column} if self.column and self.column != '*' else set()
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        return input_columns + ['cluster_id']
    
    def execute(self, df: pd.DataFrame, backend_impl: Any) -> pd.DataFrame:
        return backend_impl.sem_group_by(df, self.column, self.k)


@dataclass
class SemAgg(SemOp):
    """Semantic aggregation operation."""
    
    prompt: str = ""
    alias: str = "agg_result"
    column: Optional[str] = None
    group_by_cols: Optional[List[str]] = None
    
    def __post_init__(self):
        self.kernel = SemanticKernelSignature(prompt_template=self.prompt)
        self.blocking = True
    
    def get_op_type(self) -> str:
        return "agg"
    
    def get_required_columns(self) -> Set[str]:
        import re
        cols = set(re.findall(r'\{([^}]+)\}', self.prompt))
        if self.column:
            cols.add(self.column)
        if self.group_by_cols:
            cols.update(self.group_by_cols)
        return cols
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        out = list(self.group_by_cols) if self.group_by_cols else []
        out.append(self.alias)
        return out
    
    def execute(self, df: pd.DataFrame, backend_impl: Any) -> pd.DataFrame:
        return backend_impl.sem_agg(df, self.prompt, self.alias, self.group_by_cols, self.column)


@dataclass
class SemDistinct(SemOp):
    """Semantic DISTINCT/deduplication operation."""
    
    column: str = ""
    alias: Optional[str] = None
    
    def __post_init__(self):
        self.kernel = SemanticKernelSignature()
        self.duplicate_preserving = False
    
    def get_op_type(self) -> str:
        return "distinct"
    
    def get_required_columns(self) -> Set[str]:
        return {self.column} if self.column and self.column != '*' else set()
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        return input_columns
    
    def execute(self, df: pd.DataFrame, backend_impl: Any) -> pd.DataFrame:
        result = backend_impl.sem_distinct(df, self.column)
        if self.alias and self.column in result.columns:
            result = result.rename(columns={self.column: self.alias})
        return result


@dataclass
class SemOrderBy(SemOp):
    """Semantic ORDER BY operation."""
    
    prompt: str = ""
    column: Optional[str] = None
    
    def __post_init__(self):
        self.kernel = SemanticKernelSignature(prompt_template=self.prompt)
        self.order_preserving = True
        self.blocking = True
    
    def get_op_type(self) -> str:
        return "orderby"
    
    def get_required_columns(self) -> Set[str]:
        import re
        cols = set(re.findall(r'\{([^}]+)\}', self.prompt))
        if self.column:
            cols.add(self.column)
        return cols
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        return input_columns
    
    def execute(self, df: pd.DataFrame, backend_impl: Any) -> pd.DataFrame:
        return backend_impl.sem_order_by(df, self.prompt, self.column)


@dataclass
class SemIntersect(SemOp):
    """Semantic INTERSECT operation."""
    
    is_all: bool = False  # True for INTERSECT ALL (bag semantics)
    
    def __post_init__(self):
        self.determinism = DeterminismClass.DETERMINISTIC
        self.duplicate_preserving = self.is_all
    
    def get_op_type(self) -> str:
        return "intersect_all" if self.is_all else "intersect"
    
    def get_required_columns(self) -> Set[str]:
        return set()  # Operates on whole rows
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        return input_columns
    
    def execute(self, df1: pd.DataFrame, df2: pd.DataFrame, rm: Any, semantic_ops: Any) -> pd.DataFrame:
        return semantic_ops.intersect_operation(df1, df2, rm, is_set=not self.is_all)


@dataclass
class SemExcept(SemOp):
    """Semantic EXCEPT operation."""
    
    is_all: bool = False  # True for EXCEPT ALL (bag semantics)
    
    def __post_init__(self):
        self.determinism = DeterminismClass.DETERMINISTIC
        self.duplicate_preserving = self.is_all
    
    def get_op_type(self) -> str:
        return "except_all" if self.is_all else "except"
    
    def get_required_columns(self) -> Set[str]:
        return set()  # Operates on whole rows
    
    def get_output_columns(self, input_columns: List[str]) -> List[str]:
        return input_columns
    
    def execute(self, df1: pd.DataFrame, df2: pd.DataFrame, rm: Any, semantic_ops: Any) -> pd.DataFrame:
        return semantic_ops.except_operation(df1, df2, rm, is_set=not self.is_all)
