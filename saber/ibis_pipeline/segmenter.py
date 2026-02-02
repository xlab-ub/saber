"""
Plan Segmenter for Ibis Pipeline.

Partitions the optimized logical plan into segments:
- RelationalSegment: maximal subgraph of relational ops (executes in DuckDB)
- SemanticSegment: semantic ops cluster (executes in pandas + SDPS)

Inserts bridge operators between segments for data conversion.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .logical_plan import LogicalPlan, PlanNode, RelationalNode, SemanticNode, ScanNode, BridgeNode, NodeType
from .operators import SemOp

logger = logging.getLogger(__name__)


class SegmentType(Enum):
    RELATIONAL = "relational"
    SEMANTIC = "semantic"


@dataclass
class Segment:
    """Base class for execution segments."""
    
    segment_id: int = 0
    segment_type: SegmentType = SegmentType.RELATIONAL
    nodes: List[PlanNode] = field(default_factory=list)
    input_columns: List[str] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)
    
    # Execution dependencies
    depends_on: List[int] = field(default_factory=list)  # segment_ids this depends on


@dataclass
class RelationalSegment(Segment):
    """Segment that executes in DuckDB."""
    
    sql: str = ""
    tables_referenced: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.segment_type = SegmentType.RELATIONAL


@dataclass
class SemanticSegment(Segment):
    """Segment that executes in pandas with SDPS backend."""
    
    operations: List[SemOp] = field(default_factory=list)
    backend: str = "lotus"
    
    # For binary operations (JOIN, INTERSECT, EXCEPT)
    right_input_segment: Optional[int] = None
    
    def __post_init__(self):
        self.segment_type = SegmentType.SEMANTIC


class PlanSegmenter:
    """Partitions a logical plan into execution segments."""
    
    def __init__(self):
        self._segment_counter = 0
        self._segments: List[Segment] = []
        self._node_to_segment: Dict[int, int] = {}
    
    def segment(self, plan: LogicalPlan) -> List[Segment]:
        """Partition the plan into segments."""
        self._segment_counter = 0
        self._segments = []
        self._node_to_segment = {}
        
        if not plan.root:
            return []
        
        # Traverse plan bottom-up to identify segment boundaries
        self._build_segments(plan.root, plan)
        
        # Insert bridge operators between segments
        self._insert_bridges()
        
        return self._segments
    
    def _next_segment_id(self) -> int:
        self._segment_counter += 1
        return self._segment_counter
    
    def _build_segments(self, node: PlanNode, plan: LogicalPlan) -> int:
        """Build segments recursively, returning the segment_id containing this node."""
        
        # Process children first
        child_segments = []
        for child in node.children:
            seg_id = self._build_segments(child, plan)
            child_segments.append(seg_id)
        
        # Handle right child for binary ops
        right_seg_id = None
        if isinstance(node, SemanticNode) and node.right_child:
            right_seg_id = self._build_segments(node.right_child, plan)
        
        # Determine if this node starts a new segment
        if isinstance(node, ScanNode):
            # Scans start relational segments
            segment = RelationalSegment(
                segment_id=self._next_segment_id(),
                nodes=[node],
                tables_referenced=[node.table_name],
                output_columns=node.output_columns
            )
            self._segments.append(segment)
            self._node_to_segment[node.node_id] = segment.segment_id
            return segment.segment_id
        
        elif isinstance(node, SemanticNode):
            # Semantic nodes form their own segments
            segment = SemanticSegment(
                segment_id=self._next_segment_id(),
                nodes=[node],
                operations=[node.sem_op] if node.sem_op else [],
                backend=node.backend,
                depends_on=child_segments,
                right_input_segment=right_seg_id,
                output_columns=node.output_columns
            )
            self._segments.append(segment)
            self._node_to_segment[node.node_id] = segment.segment_id
            return segment.segment_id
        
        elif isinstance(node, RelationalNode):
            # Try to merge with child relational segment
            if child_segments:
                parent_seg_id = child_segments[0]
                parent_seg = self._get_segment(parent_seg_id)
                
                if isinstance(parent_seg, RelationalSegment):
                    # Merge into existing relational segment
                    parent_seg.nodes.append(node)
                    parent_seg.output_columns = node.output_columns
                    self._node_to_segment[node.node_id] = parent_seg_id
                    return parent_seg_id
            
            # Create new relational segment
            segment = RelationalSegment(
                segment_id=self._next_segment_id(),
                nodes=[node],
                depends_on=child_segments,
                output_columns=node.output_columns
            )
            self._segments.append(segment)
            self._node_to_segment[node.node_id] = segment.segment_id
            return segment.segment_id
        
        elif isinstance(node, BridgeNode):
            # Bridge nodes are added later, just pass through
            if child_segments:
                return child_segments[0]
        
        # Default: create new segment
        segment = Segment(
            segment_id=self._next_segment_id(),
            nodes=[node],
            depends_on=child_segments
        )
        self._segments.append(segment)
        self._node_to_segment[node.node_id] = segment.segment_id
        return segment.segment_id
    
    def _get_segment(self, segment_id: int) -> Optional[Segment]:
        """Get segment by ID."""
        for seg in self._segments:
            if seg.segment_id == segment_id:
                return seg
        return None
    
    def _insert_bridges(self):
        """Insert bridge operators between different segment types."""
        bridges_needed = []
        
        for seg in self._segments:
            for dep_id in seg.depends_on:
                dep_seg = self._get_segment(dep_id)
                if dep_seg and seg.segment_type != dep_seg.segment_type:
                    # Need a bridge
                    bridges_needed.append((dep_id, seg.segment_id))
        
        # Bridges are implicit in execution - we track them in the dependency graph
        # The executor will handle the actual conversion
    
    def explain(self) -> str:
        """Return a string explanation of the segmentation."""
        lines = ["=== Execution Plan Segments ==="]
        
        for seg in self._segments:
            if isinstance(seg, RelationalSegment):
                lines.append(f"Segment {seg.segment_id} [DuckDB]:")
                lines.append(f"  Tables: {seg.tables_referenced}")
                lines.append(f"  Nodes: {len(seg.nodes)}")
                if seg.sql:
                    lines.append(f"  SQL: {seg.sql[:100]}...")
            elif isinstance(seg, SemanticSegment):
                lines.append(f"Segment {seg.segment_id} [pandas/{seg.backend}]:")
                ops = [op.get_op_type() for op in seg.operations]
                lines.append(f"  Operations: {ops}")
                if seg.right_input_segment:
                    lines.append(f"  Right input: Segment {seg.right_input_segment}")
            else:
                lines.append(f"Segment {seg.segment_id} [{seg.segment_type}]")
            
            if seg.depends_on:
                lines.append(f"  Depends on: {seg.depends_on}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_execution_order(self) -> List[int]:
        """Get topologically sorted order of segment execution."""
        # Build dependency graph
        in_degree = {seg.segment_id: 0 for seg in self._segments}
        
        for seg in self._segments:
            for dep in seg.depends_on:
                in_degree[seg.segment_id] += 1
            if isinstance(seg, SemanticSegment) and seg.right_input_segment:
                in_degree[seg.segment_id] += 1
        
        # Topological sort
        queue = [sid for sid, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            # Reduce in-degree for dependents
            for seg in self._segments:
                if current in seg.depends_on:
                    in_degree[seg.segment_id] -= 1
                    if in_degree[seg.segment_id] == 0:
                        queue.append(seg.segment_id)
                if isinstance(seg, SemanticSegment) and seg.right_input_segment == current:
                    in_degree[seg.segment_id] -= 1
                    if in_degree[seg.segment_id] == 0:
                        queue.append(seg.segment_id)
        
        return order
