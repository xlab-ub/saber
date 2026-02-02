"""
Logical Plan representation for hybrid relational-semantic queries.

The logical plan is a DAG where:
- RelationalNode represents standard SQL operations (can execute in DuckDB)
- SemanticNode represents semantic operations (requires pandas + SDPS backend)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Union
from enum import Enum
from .operators import SemOp


class NodeType(Enum):
    """Type of plan node."""
    RELATIONAL = "relational"
    SEMANTIC = "semantic"
    SCAN = "scan"
    BRIDGE = "bridge"


@dataclass
class PlanNode:
    """Base class for logical plan nodes."""
    
    node_id: int = 0
    node_type: NodeType = NodeType.RELATIONAL
    children: List['PlanNode'] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)
    
    # Statistics for optimization
    estimated_rows: float = 1000.0
    estimated_cost: float = 1.0


@dataclass
class ScanNode(PlanNode):
    """Scan a base table."""
    
    table_name: str = ""
    alias: Optional[str] = None
    
    def __post_init__(self):
        self.node_type = NodeType.SCAN


@dataclass
class RelationalNode(PlanNode):
    """Standard relational operation (Filter, Project, Join, Agg, etc.)."""
    
    op_type: str = ""  # 'filter', 'project', 'join', 'aggregate', 'sort', 'limit', 'distinct'
    predicate: Optional[str] = None  # For filter
    projections: List[str] = field(default_factory=list)  # For project
    join_type: str = "inner"  # For join
    join_condition: Optional[str] = None
    group_by: List[str] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    aggregations: Dict[str, str] = field(default_factory=dict)  # alias -> agg_expr
    
    def __post_init__(self):
        self.node_type = NodeType.RELATIONAL


@dataclass
class SemanticNode(PlanNode):
    """Semantic operation node."""
    
    sem_op: Optional[SemOp] = None
    backend: str = "lotus"
    
    # For binary semantic ops (join, intersect, except)
    right_child: Optional['PlanNode'] = None
    
    # Limit pushdown hint for early termination
    limit_hint: Optional[int] = None
    
    def __post_init__(self):
        self.node_type = NodeType.SEMANTIC


@dataclass
class BridgeNode(PlanNode):
    """Bridge between relational and semantic execution."""
    
    bridge_type: str = ""  # 'to_pandas' or 'from_pandas'
    
    def __post_init__(self):
        self.node_type = NodeType.BRIDGE


class LogicalPlan:
    """Container for a logical query plan."""
    
    def __init__(self):
        self.root: Optional[PlanNode] = None
        self._node_counter = 0
        self._table_schemas: Dict[str, List[str]] = {}
    
    def set_table_schema(self, table_name: str, columns: List[str]):
        """Register table schema for planning."""
        self._table_schemas[table_name] = columns
    
    def get_table_schema(self, table_name: str) -> List[str]:
        """Get registered table schema."""
        return self._table_schemas.get(table_name, [])
    
    def _next_node_id(self) -> int:
        self._node_counter += 1
        return self._node_counter
    
    def create_scan(self, table_name: str, alias: Optional[str] = None) -> ScanNode:
        """Create a table scan node."""
        node = ScanNode(
            node_id=self._next_node_id(),
            table_name=table_name,
            alias=alias,
            output_columns=self.get_table_schema(table_name)
        )
        return node
    
    def create_relational(self, op_type: str, child: PlanNode, **kwargs) -> RelationalNode:
        """Create a relational operation node."""
        node = RelationalNode(
            node_id=self._next_node_id(),
            op_type=op_type,
            children=[child],
            output_columns=child.output_columns.copy(),
            **kwargs
        )
        return node
    
    def create_semantic(self, sem_op: SemOp, child: PlanNode,
                       right_child: Optional[PlanNode] = None) -> SemanticNode:
        """Create a semantic operation node."""
        output_cols = sem_op.get_output_columns(child.output_columns)
        node = SemanticNode(
            node_id=self._next_node_id(),
            sem_op=sem_op,
            backend=sem_op.backend,
            children=[child],
            right_child=right_child,
            output_columns=output_cols
        )
        return node
    
    def create_bridge(self, bridge_type: str, child: PlanNode) -> BridgeNode:
        """Create a bridge node."""
        node = BridgeNode(
            node_id=self._next_node_id(),
            bridge_type=bridge_type,
            children=[child],
            output_columns=child.output_columns.copy()
        )
        return node
    
    def get_all_nodes(self) -> List[PlanNode]:
        """Return all nodes in the plan via BFS."""
        if not self.root:
            return []
        
        visited = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            visited.append(node)
            queue.extend(node.children)
            if isinstance(node, SemanticNode) and node.right_child:
                queue.append(node.right_child)
        
        return visited
    
    def get_semantic_nodes(self) -> List[SemanticNode]:
        """Return all semantic nodes in the plan."""
        return [n for n in self.get_all_nodes() if isinstance(n, SemanticNode)]
    
    def get_relational_nodes(self) -> List[RelationalNode]:
        """Return all relational nodes in the plan."""
        return [n for n in self.get_all_nodes() if isinstance(n, RelationalNode)]
    
    def has_semantic_ops(self) -> bool:
        """Check if plan contains any semantic operations."""
        return len(self.get_semantic_nodes()) > 0
    
    def explain(self, indent: int = 0) -> str:
        """Return a string representation of the plan."""
        if not self.root:
            return "Empty plan"
        return self._explain_node(self.root, indent)
    
    def _explain_node(self, node: PlanNode, indent: int) -> str:
        prefix = "  " * indent
        
        if isinstance(node, ScanNode):
            desc = f"{prefix}Scan({node.table_name}"
            if node.alias:
                desc += f" AS {node.alias}"
            desc += ")"
        elif isinstance(node, RelationalNode):
            desc = f"{prefix}Relational({node.op_type}"
            if node.predicate:
                desc += f", pred={node.predicate[:30]}..."
            if node.projections:
                desc += f", proj={node.projections}"
            if node.limit:
                desc += f", limit={node.limit}"
            desc += ")"
        elif isinstance(node, SemanticNode):
            op_type = node.sem_op.get_op_type() if node.sem_op else "unknown"
            desc = f"{prefix}Semantic({op_type}, backend={node.backend}"
            # Show limit_hint if pushed down
            if node.limit_hint is not None:
                desc += f", limit_hint={node.limit_hint}"
            desc += ")"
        elif isinstance(node, BridgeNode):
            desc = f"{prefix}Bridge({node.bridge_type})"
        else:
            desc = f"{prefix}Node({node.node_type})"
        
        lines = [desc]
        for child in node.children:
            lines.append(self._explain_node(child, indent + 1))
        if isinstance(node, SemanticNode) and node.right_child:
            lines.append(f"{prefix}  [right]")
            lines.append(self._explain_node(node.right_child, indent + 2))
        
        return "\n".join(lines)
