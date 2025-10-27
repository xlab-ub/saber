"""Benchmarking utilities for tracking LLM costs and execution time across backends."""
import time
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkStats:
    """Statistics for query execution tracking.
    
    Tracks five key metrics:
    - total_cost: Total LLM API cost in USD
    - total_execution_time_seconds: Total query execution time (SQL + semantic ops + rewriting)
    - total_semantic_execution_time_seconds: Time spent on semantic operations only
    - total_non_semantic_execution_time_seconds: Time spent on SQL operations only
    - total_rewriting_time_seconds: Time spent on rewriting backend-free queries
    """
    total_cost: float = 0.0
    total_execution_time_seconds: float = 0.0
    total_semantic_execution_time_seconds: float = 0.0
    total_non_semantic_execution_time_seconds: float = 0.0
    total_rewriting_time_seconds: float = 0.0
    
    def __add__(self, other: 'BenchmarkStats') -> 'BenchmarkStats':
        """Add two benchmark stats together."""
        return BenchmarkStats(
            total_cost=self.total_cost + other.total_cost,
            total_execution_time_seconds=self.total_execution_time_seconds + other.total_execution_time_seconds,
            total_semantic_execution_time_seconds=self.total_semantic_execution_time_seconds + other.total_semantic_execution_time_seconds,
            total_non_semantic_execution_time_seconds=self.total_non_semantic_execution_time_seconds + other.total_non_semantic_execution_time_seconds,
            total_rewriting_time_seconds=self.total_rewriting_time_seconds + other.total_rewriting_time_seconds
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/export."""
        return {
            'total_cost': self.total_cost,
            'total_execution_time_seconds': self.total_execution_time_seconds,
            'total_semantic_execution_time_seconds': self.total_semantic_execution_time_seconds,
            'total_non_semantic_execution_time_seconds': self.total_non_semantic_execution_time_seconds,
            'total_rewriting_time_seconds': self.total_rewriting_time_seconds,
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return (f"Cost: ${self.total_cost:.6f} | "
                f"Total Time: {self.total_execution_time_seconds:.2f}s | "
                f"Semantic Time: {self.total_semantic_execution_time_seconds:.2f}s | "
                f"SQL Time: {self.total_non_semantic_execution_time_seconds:.2f}s | "
                f"Rewriting Time: {self.total_rewriting_time_seconds:.2f}s")


class BenchmarkTracker:
    """Tracks benchmarking statistics across operations."""
    
    def __init__(self):
        self.current_stats = BenchmarkStats()
        self._start_time: Optional[float] = None
        
    def start_operation(self):
        """Start timing an operation."""
        self._start_time = time.time()
        
    def end_operation(self, stats: BenchmarkStats):
        """End timing and record stats for current operation."""
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            stats.total_semantic_execution_time_seconds = elapsed
            self._start_time = None
        self.current_stats = self.current_stats + stats
        
    def reset(self):
        """Reset all statistics."""
        self.current_stats = BenchmarkStats()
        self._start_time = None
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all statistics."""
        return self.current_stats.to_dict()


def extract_lotus_stats(lm) -> BenchmarkStats:
    """Extract cost from LOTUS LM object."""
    stats = BenchmarkStats()
    try:
        stats.total_cost = lm.stats.physical_usage.total_cost
        lm.reset_stats()
    except Exception as e:
        logger.warning(f"Failed to extract LOTUS stats: {e}")
    return stats


def extract_docetl_stats(output_text: str) -> BenchmarkStats:
    """Extract statistics from DocETL CLI output text."""
    output_text = output_text.split("Execution Summary")[-1]
    # logger.info(f"DocETL output for stats extraction:\n{output_text}")
    stats = BenchmarkStats()
    try:
        # Extract cost: "Cost: $0.00"
        cost_match = re.search(r'Cost:\s*\$([^\n]+?)(?:\s*\n|\s*│)', output_text)
        # logger.info(f"Extracted cost string: {cost_match.group(1).strip() if cost_match else 'N/A'}")
        if cost_match:
            stats.total_cost = float(cost_match.group(1).strip())
        
        # Extract time: "Time: 0.87s"
        time_match = re.search(r'Time:\s*([^\n]+?)s(?:\s*\n|\s*│)', output_text)
        # logger.info(f"Extracted time string: {time_match.group(1).strip() if time_match else 'N/A'}")
        if time_match:
            stats.total_semantic_execution_time_seconds = float(time_match.group(1).strip())
    except Exception as e:
        logger.warning(f"Failed to extract DocETL stats: {e}")
    return stats


def extract_palimpzest_stats(execution_stats) -> BenchmarkStats:
    """Extract cost and time from Palimpzest execution_stats object."""
    stats = BenchmarkStats()
    try:
        stats.total_cost = execution_stats.total_execution_cost
        stats.total_semantic_execution_time_seconds = getattr(execution_stats, 'total_execution_time', 0.0)
    except Exception as e:
        logger.warning(f"Failed to extract Palimpzest stats: {e}")
    return stats
