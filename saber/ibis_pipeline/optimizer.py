"""
Plan Optimizer for Ibis Pipeline.

Implements semantic-aware optimization rules including:
- Projection pushdown around semantic nodes
- Predicate splitting and pushdown
- Join reordering for semantic cost minimization
- Boundary minimization (reduce DuckDB <-> pandas crossings)
- Limit pushdown for early termination in semantic operations
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Optional, Tuple

from .logical_plan import LogicalPlan, PlanNode, RelationalNode, SemanticNode, ScanNode, BridgeNode

logger = logging.getLogger(__name__)


class OptimizationRule(ABC):
    """Base class for optimization rules."""
    
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply the optimization rule to the plan."""
        pass


class PredicatePushdown(OptimizationRule):
    """Push relational predicates below semantic operations when safe."""
    
    def name(self) -> str:
        return "predicate_pushdown"
    
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        if not plan.root:
            return plan
        
        plan.root = self._push_predicates(plan.root)
        return plan
    
    def _push_predicates(self, node: PlanNode) -> PlanNode:
        """Recursively push predicates down."""
        # Process children first
        for i, child in enumerate(node.children):
            node.children[i] = self._push_predicates(child)
        
        # If this is a filter with a semantic child, try to push below
        if isinstance(node, RelationalNode) and node.op_type == 'filter':
            if node.children and isinstance(node.children[0], SemanticNode):
                sem_node = node.children[0]
                
                # Check if predicate only uses columns not modified by semantic op
                if sem_node.sem_op and self._can_push_past(node, sem_node):
                    # Swap: put filter below semantic node
                    logger.debug(f"Pushing predicate below {sem_node.sem_op.get_op_type()}")
                    
                    if sem_node.children:
                        original_child = sem_node.children[0]
                        node.children = [original_child]
                        sem_node.children = [node]
                        return sem_node
        
        return node
    
    def _can_push_past(self, filter_node: RelationalNode, sem_node: SemanticNode) -> bool:
        """Check if filter can safely be pushed past semantic node.
        
        Key insight: pushing relational filters BEFORE semantic ops reduces
        the number of rows processed by expensive LLM calls.
        """
        if not sem_node.sem_op:
            return False
        
        op_type = sem_node.sem_op.get_op_type()
        
        # SEM_WHERE: does NOT add columns, only filters rows
        # We CAN push relational predicates past it to reduce LLM calls
        if op_type == 'where':
            # The predicate must only reference columns that exist before SEM_WHERE
            # SEM_WHERE doesn't add any columns, so any input column is available
            return self._predicate_uses_input_columns_only(filter_node, sem_node)
        
        # SEM_SELECT: adds a new column (alias)
        # Can push past if predicate doesn't reference the new column
        if op_type == 'select':
            return self._predicate_uses_input_columns_only(filter_node, sem_node)
        
        # SEM_AGG: replaces row content, can't push past
        if op_type == 'agg':
            return False
        
        # SEM_GROUP_BY: modifies structure, can't push past
        if op_type == 'groupby':
            return False
        
        # Can't push past blocking operations in general
        if sem_node.sem_op.blocking:
            return False
        
        # Default: allow pushdown for non-blocking ops
        return self._predicate_uses_input_columns_only(filter_node, sem_node)
    
    def _predicate_uses_input_columns_only(self, filter_node: RelationalNode, 
                                            sem_node: SemanticNode) -> bool:
        """Check if filter predicate only uses columns available at sem_node's input."""
        predicate = filter_node.predicate
        if not predicate:
            return False
        
        # Get columns added by the semantic operation
        added_columns = set()
        if sem_node.sem_op:
            op_type = sem_node.sem_op.get_op_type()
            if op_type == 'select' and hasattr(sem_node.sem_op, 'alias'):
                added_columns.add(sem_node.sem_op.alias)
            elif op_type == 'agg' and hasattr(sem_node.sem_op, 'alias'):
                added_columns.add(sem_node.sem_op.alias)
        
        # Simple heuristic: check if predicate mentions any added column
        # More robust would be to parse the predicate SQL
        predicate_upper = predicate.upper()
        for col in added_columns:
            if col.upper() in predicate_upper:
                logger.debug(f"Predicate '{predicate}' references added column '{col}', cannot push down")
                return False
        
        return True


class JoinPredicatePushdown(OptimizationRule):
    """Push predicates into join inputs to reduce join cardinality.
    
    For queries like:
        SELECT ... FROM SEM_JOIN(t1, t2, ...) WHERE t1.x = 'a' AND t2.y = 'b'
    
    Split the predicate and push:
    - t1.x = 'a' to filter the left input
    - t2.y = 'b' to filter the right input
    
    This dramatically reduces the number of LLM calls in SEM_JOIN since
    we filter down to relevant rows before the expensive semantic comparison.
    """
    
    def name(self) -> str:
        return "join_predicate_pushdown"
    
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        if not plan.root:
            return plan
        
        plan.root = self._push_join_predicates(plan.root)
        return plan
    
    def _push_join_predicates(self, node: PlanNode) -> PlanNode:
        """Recursively look for filter -> sem_join patterns and push predicates."""
        # Process children first (bottom-up)
        for i, child in enumerate(node.children):
            node.children[i] = self._push_join_predicates(child)
        
        # Also process right_child for semantic nodes
        if isinstance(node, SemanticNode) and node.right_child:
            node.right_child = self._push_join_predicates(node.right_child)
        
        # Look for: Filter -> SemJoin pattern
        if isinstance(node, RelationalNode) and node.op_type == 'filter' and node.predicate:
            if node.children and isinstance(node.children[0], SemanticNode):
                sem_node = node.children[0]
                if sem_node.sem_op and sem_node.sem_op.get_op_type() == 'join':
                    # Try to split and push the predicate
                    return self._split_and_push_join_predicate(node, sem_node)
        
        return node
    
    def _split_and_push_join_predicate(self, filter_node: RelationalNode, 
                                        sem_join: SemanticNode) -> PlanNode:
        """Split predicate into left/right parts and push to join inputs.
        
        Args:
            filter_node: The filter node with predicate to push
            sem_join: The SEM_JOIN node
            
        Returns:
            The modified plan tree
        """
        predicate = filter_node.predicate
        if not predicate:
            return filter_node
        
        # Get table aliases from SEM_JOIN
        left_alias = None
        right_alias = None
        
        if sem_join.sem_op and hasattr(sem_join.sem_op, 'left_alias'):
            left_alias = sem_join.sem_op.left_alias
        if sem_join.sem_op and hasattr(sem_join.sem_op, 'right_alias'):
            right_alias = sem_join.sem_op.right_alias
        
        # Fallback: infer from child nodes
        if not left_alias and sem_join.children:
            left_alias = self._get_table_alias(sem_join.children[0])
        if not right_alias and sem_join.right_child:
            right_alias = self._get_table_alias(sem_join.right_child)
        
        if not left_alias or not right_alias:
            logger.debug("Cannot determine table aliases for SEM_JOIN, skipping predicate pushdown")
            return filter_node
        
        # Split predicate by AND
        left_preds, right_preds, remaining_preds = self._split_predicate(
            predicate, left_alias, right_alias
        )
        
        logger.debug(f"Split predicate: left={left_preds}, right={right_preds}, remaining={remaining_preds}")
        
        # Push left predicates to left child
        if left_preds and sem_join.children:
            left_child = sem_join.children[0]
            for pred in left_preds:
                # Convert table.col to just col for the filter
                simple_pred = self._simplify_predicate(pred, left_alias)
                new_filter = RelationalNode(
                    node_id=filter_node.node_id * 100 + 1,  # Generate new ID
                    op_type='filter',
                    predicate=simple_pred,
                    children=[left_child]
                )
                left_child = new_filter
            sem_join.children[0] = left_child
        
        # Push right predicates to right child
        if right_preds and sem_join.right_child:
            right_child = sem_join.right_child
            for pred in right_preds:
                # Convert table.col to just col for the filter
                simple_pred = self._simplify_predicate(pred, right_alias)
                new_filter = RelationalNode(
                    node_id=filter_node.node_id * 100 + 2,  # Generate new ID
                    op_type='filter',
                    predicate=simple_pred,
                    children=[right_child]
                )
                right_child = new_filter
            sem_join.right_child = right_child
        
        # If all predicates were pushed, remove the filter node
        if not remaining_preds:
            return sem_join
        
        # Otherwise, keep filter with remaining predicates
        filter_node.predicate = ' AND '.join(remaining_preds)
        return filter_node
    
    def _get_table_alias(self, node: PlanNode) -> Optional[str]:
        """Get the table alias from a plan node."""
        if isinstance(node, ScanNode):
            return node.alias or node.table_name
        # Traverse down to find scan
        if node.children:
            return self._get_table_alias(node.children[0])
        return None
    
    def _split_predicate(self, predicate: str, left_alias: str, right_alias: str
                         ) -> tuple[list[str], list[str], list[str]]:
        """Split predicate into left, right, and remaining parts.
        
        Returns:
            Tuple of (left_predicates, right_predicates, remaining_predicates)
        """
        import re
        
        left_preds = []
        right_preds = []
        remaining_preds = []
        
        # Split by AND (case-insensitive)
        # Handle nested parentheses properly
        parts = self._split_by_and(predicate)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check which table(s) this predicate references
            refs_left = self._references_table(part, left_alias)
            refs_right = self._references_table(part, right_alias)
            
            if refs_left and not refs_right:
                # Only references left table - can push to left
                left_preds.append(part)
            elif refs_right and not refs_left:
                # Only references right table - can push to right
                right_preds.append(part)
            else:
                # References both or neither - keep in filter
                remaining_preds.append(part)
        
        return left_preds, right_preds, remaining_preds
    
    def _split_by_and(self, predicate: str) -> list[str]:
        """Split predicate by AND, respecting parentheses."""
        import re
        
        # Simple split by AND (handling case)
        # For now, use a simple regex split - can be made more robust
        parts = re.split(r'\s+AND\s+', predicate, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]
    
    def _references_table(self, pred: str, table_alias: str) -> bool:
        """Check if predicate references a specific table."""
        import re
        # Look for table.column pattern
        pattern = rf'\b{re.escape(table_alias)}\.[\w]+\b'
        return bool(re.search(pattern, pred, re.IGNORECASE))
    
    def _simplify_predicate(self, pred: str, table_alias: str) -> str:
        """Remove table prefix from predicate for pushing to single table.
        
        e.g., "reviews.id = 'X'" becomes "id = 'X'" when pushing to reviews table
        """
        import re
        # Replace table.col with col
        pattern = rf'\b{re.escape(table_alias)}\.([\w]+)\b'
        return re.sub(pattern, r'\1', pred, flags=re.IGNORECASE)


class ProjectionPushdown(OptimizationRule):
    """Push projections to limit data passed to semantic operations."""
    
    def name(self) -> str:
        return "projection_pushdown"
    
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        if not plan.root:
            return plan
        
        # Collect columns needed by each node
        needed = self._collect_needed_columns(plan.root)
        
        # Insert projections where beneficial
        plan.root = self._insert_projections(plan.root, needed)
        
        return plan
    
    def _collect_needed_columns(self, node: PlanNode) -> Set[str]:
        """Collect columns needed by this node and its ancestors."""
        needed = set()
        
        if isinstance(node, SemanticNode) and node.sem_op:
            needed.update(node.sem_op.get_required_columns())
        
        if isinstance(node, RelationalNode):
            if node.group_by:
                needed.update(node.group_by)
            if node.order_by:
                needed.update(node.order_by)
        
        # Collect from children
        for child in node.children:
            needed.update(self._collect_needed_columns(child))
        
        return needed
    
    def _insert_projections(self, node: PlanNode, needed: Set[str]) -> PlanNode:
        """Insert projection nodes where they reduce data volume."""
        # For now, just propagate - actual insertion would require more analysis
        for i, child in enumerate(node.children):
            node.children[i] = self._insert_projections(child, needed)
        
        return node


class BoundaryMinimization(OptimizationRule):
    """Minimize transitions between DuckDB and pandas execution."""
    
    def name(self) -> str:
        return "boundary_minimization"
    
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        if not plan.root:
            return plan
        
        # Cluster consecutive semantic operations to reduce boundary crossings
        plan.root = self._cluster_semantic_ops(plan.root)
        
        return plan
    
    def _cluster_semantic_ops(self, node: PlanNode) -> PlanNode:
        """Cluster consecutive semantic operations together."""
        # Process children first
        for i, child in enumerate(node.children):
            node.children[i] = self._cluster_semantic_ops(child)
        
        # Look for patterns like: Relational -> Semantic -> Relational -> Semantic
        # and try to reorder if semantically equivalent
        # For now, just return as-is - full clustering requires more analysis
        
        return node


class SemanticCostReordering(OptimizationRule):
    """Reorder operations to minimize expensive semantic operation costs."""
    
    def name(self) -> str:
        return "semantic_cost_reordering"
    
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        if not plan.root:
            return plan
        
        # Apply filters before expensive semantic ops to reduce input size
        plan.root = self._push_filters_before_semantic(plan.root)
        
        return plan
    
    def _push_filters_before_semantic(self, node: PlanNode) -> PlanNode:
        """Push filters to execute before expensive semantic operations."""
        # Process children first
        for i, child in enumerate(node.children):
            node.children[i] = self._push_filters_before_semantic(child)
        
        # If semantic op with relational filter child, check if we can reorder
        if isinstance(node, SemanticNode) and node.sem_op:
            # Expensive ops: SEM_JOIN, SEM_ORDER_BY
            if node.sem_op.get_op_type() in ('join', 'orderby', 'agg'):
                # Look for filters in children that could be pushed to grandchildren
                pass
        
        return node


class LimitPushdown(OptimizationRule):
    """Push LIMIT hints into semantic operations for early termination.
    
    For queries like:
        SELECT ... FROM t WHERE SEM_WHERE(...) LIMIT N
    
    The LIMIT N can be pushed into the SEM_WHERE as a hint, allowing the 
    semantic filter to stop processing once N qualifying rows are found.
    
    This optimization is valid when:
    - LIMIT directly follows a semantic filter (SEM_WHERE)
    - There's no ORDER BY between the filter and LIMIT (streaming LIMIT)
    - The semantic operation preserves input order (which SEM_WHERE does)
    
    Note: This is different from pushing LIMIT before a filter, which would
    be incorrect. Here we push the LIMIT as a *hint* so the filter can 
    terminate early after finding enough matching rows.
    """
    
    def name(self) -> str:
        return "limit_pushdown"
    
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        if not plan.root:
            return plan
        
        # Find LIMIT nodes and try to push the limit hint down
        plan.root = self._push_limit(plan.root, limit_hint=None)
        return plan
    
    def _push_limit(self, node: PlanNode, limit_hint: Optional[int]) -> PlanNode:
        """Recursively push limit hints down the plan tree.
        
        Args:
            node: Current node being processed
            limit_hint: The limit value to push down, if any
        
        Returns:
            The processed node (potentially modified)
        """
        # If this is a LIMIT node, capture the limit value to push down
        if isinstance(node, RelationalNode) and node.op_type == 'limit' and node.limit:
            new_limit = node.limit
            # If we already have a limit hint from above, take the minimum
            if limit_hint is not None:
                new_limit = min(new_limit, limit_hint)
            
            # Process child with the limit hint
            if node.children:
                node.children[0] = self._push_limit(node.children[0], new_limit)
            return node
        
        # If this is a semantic node, check if we can apply the limit hint
        if isinstance(node, SemanticNode) and node.sem_op and limit_hint is not None:
            op_type = node.sem_op.get_op_type()
            
            if self._can_push_limit_into(node):
                node.limit_hint = limit_hint
                node.sem_op.limit_hint = limit_hint
                logger.debug(f"Pushed LIMIT {limit_hint} into {op_type.upper()} node {node.node_id}")
        
        # Check if we can propagate the limit hint through this node
        propagate_limit = self._can_propagate_limit_through(node, limit_hint)
        
        # Process children
        for i, child in enumerate(node.children):
            child_limit = limit_hint if propagate_limit else None
            node.children[i] = self._push_limit(child, child_limit)
        
        # Handle right child for binary ops
        if isinstance(node, SemanticNode) and node.right_child:
            # Don't propagate limit to right side of binary ops (JOIN, etc.)
            node.right_child = self._push_limit(node.right_child, None)
        
        return node
    
    def _can_push_limit_into(self, node: SemanticNode) -> bool:
        """Check if we can push a limit hint into this semantic node.
        
        Limit can be pushed into:
        - SEM_WHERE: Filter preserves order and can terminate early
        - SEM_JOIN: Can use limit hint for early termination (batched processing)
        
        Limit cannot be pushed into:
        - SEM_ORDER_BY: Needs all data to sort
        - SEM_GROUP_BY: Needs all data to cluster
        - SEM_AGG: Needs all data to aggregate
        - SEM_DISTINCT: Needs to see all data for deduplication
        """
        if not node.sem_op:
            return False
        
        op_type = node.sem_op.get_op_type()
        
        # SEM_WHERE can accept limit hint for early termination
        if op_type == 'where':
            return True
        
        # SEM_JOIN can use limit hint for batched/early termination
        # The join will process in batches and stop once enough matches are found
        if op_type == 'join':
            return True
        
        return False
    
    def _can_propagate_limit_through(self, node: PlanNode, limit_hint: Optional[int]) -> bool:
        """Check if limit hint can be propagated through this node to children.
        
        Limit can propagate through:
        - Project nodes (don't change cardinality)
        - LIMIT nodes (take minimum)
        - Filter nodes (may reduce rows, but still valid to push further)
        
        Limit cannot propagate through:
        - ORDER BY: Needs all data, then limit applies
        - Aggregate: Changes cardinality
        - Blocking semantic ops: Need all data
        """
        if limit_hint is None:
            return False
        
        if isinstance(node, ScanNode):
            # Can't push below scan
            return False
        
        if isinstance(node, RelationalNode):
            # LIMIT: propagate (handled specially above)
            if node.op_type == 'limit':
                return True
            # Project: propagate (doesn't change cardinality)
            if node.op_type == 'project':
                return True
            # Filter: propagate (may reduce, but still valid)
            if node.op_type == 'filter':
                return True
            # Sort/Order By: STOP - need all data first
            if node.op_type in ('sort', 'order'):
                return False
            # Aggregate: STOP - changes cardinality  
            if node.op_type == 'aggregate':
                return False
            # Distinct: STOP - needs all data
            if node.op_type == 'distinct':
                return False
        
        if isinstance(node, SemanticNode) and node.sem_op:
            op_type = node.sem_op.get_op_type()
            # SEM_WHERE: propagate (and also accept the hint)
            if op_type == 'where':
                return True
            # SEM_SELECT: propagate (projection, doesn't change cardinality)
            if op_type == 'select':
                return True
            # Blocking ops: STOP
            if node.sem_op.blocking:
                return False
        
        return False


class PlanOptimizer:
    """Main optimizer that applies a sequence of optimization rules."""
    
    DEFAULT_RULES = [
        JoinPredicatePushdown(),  # Push predicates into join inputs first
        PredicatePushdown(),
        ProjectionPushdown(),
        SemanticCostReordering(),
        LimitPushdown(),
        BoundaryMinimization(),
    ]
    
    def __init__(self, rules: Optional[List[OptimizationRule]] = None):
        self.rules = rules or self.DEFAULT_RULES
    
    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply all optimization rules to the plan."""
        optimized = plan
        
        for rule in self.rules:
            try:
                logger.debug(f"Applying optimization rule: {rule.name()}")
                optimized = rule.apply(optimized)
            except Exception as e:
                logger.warning(f"Optimization rule {rule.name()} failed: {e}")
        
        return optimized
    
    def explain_optimizations(self, original: LogicalPlan, optimized: LogicalPlan) -> str:
        """Explain what optimizations were applied."""
        lines = ["=== Optimization Summary ==="]
        
        orig_sem = len(original.get_semantic_nodes())
        opt_sem = len(optimized.get_semantic_nodes())
        
        lines.append(f"Semantic operations: {orig_sem} -> {opt_sem}")
        
        orig_rel = len(original.get_relational_nodes())
        opt_rel = len(optimized.get_relational_nodes())
        
        lines.append(f"Relational operations: {orig_rel} -> {opt_rel}")
        
        return "\n".join(lines)
