"""
Bridge Operators for Ibis Pipeline.

Handle conversion between:
- DuckDB relational results -> pandas DataFrames (ToPandas)
- pandas DataFrames -> DuckDB views/tables (FromPandas)

These are explicit, costed operators in the physical plan.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BridgeCost:
    """Cost estimate for a bridge operation."""
    
    rows: int = 0
    columns: int = 0
    conversion_time_ms: float = 0.0
    memory_bytes: int = 0


class BridgeOperator:
    """Base class for bridge operators."""
    
    def __init__(self):
        self.last_cost: Optional[BridgeCost] = None
    
    def estimate_cost(self, rows: int, columns: int) -> BridgeCost:
        """Estimate the cost of this bridge operation."""
        # Rough estimate: ~0.1ms per 1000 rows per column
        time_ms = (rows * columns) / 10000.0
        memory_bytes = rows * columns * 8  # Assume 8 bytes per cell average
        
        return BridgeCost(
            rows=rows,
            columns=columns,
            conversion_time_ms=time_ms,
            memory_bytes=memory_bytes
        )


class ToPandas(BridgeOperator):
    """Convert DuckDB result to pandas DataFrame."""
    
    def execute(self, duckdb_result: Any, db_adapter: Any = None) -> pd.DataFrame:
        """Convert a DuckDB relation/cursor result to pandas DataFrame."""
        start_time = time.time()
        
        try:
            if hasattr(duckdb_result, 'df'):
                # DuckDB relation
                df = duckdb_result.df()
            elif hasattr(duckdb_result, 'fetchdf'):
                # DuckDB cursor
                df = duckdb_result.fetchdf()
            elif isinstance(duckdb_result, pd.DataFrame):
                # Already a DataFrame
                df = duckdb_result
            else:
                # Try to convert via db_adapter
                if db_adapter:
                    df = db_adapter.fetch_df(duckdb_result)
                else:
                    raise TypeError(f"Cannot convert {type(duckdb_result)} to pandas")
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.last_cost = BridgeCost(
                rows=len(df),
                columns=len(df.columns),
                conversion_time_ms=elapsed_ms,
                memory_bytes=df.memory_usage(deep=True).sum()
            )
            
            logger.debug(f"ToPandas: {self.last_cost.rows} rows x {self.last_cost.columns} cols "
                        f"in {elapsed_ms:.2f}ms")
            
            return df
            
        except Exception as e:
            logger.error(f"ToPandas conversion failed: {e}")
            raise


class FromPandas(BridgeOperator):
    """Convert pandas DataFrame back to DuckDB view/table."""
    
    def execute(self, df: pd.DataFrame, db_adapter: Any, view_name: str) -> str:
        """Register pandas DataFrame as a DuckDB view."""
        start_time = time.time()
        
        try:
            # Register DataFrame as view
            db_adapter.register(view_name, df)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.last_cost = BridgeCost(
                rows=len(df),
                columns=len(df.columns),
                conversion_time_ms=elapsed_ms,
                memory_bytes=df.memory_usage(deep=True).sum()
            )
            
            logger.debug(f"FromPandas: {self.last_cost.rows} rows x {self.last_cost.columns} cols "
                        f"to view '{view_name}' in {elapsed_ms:.2f}ms")
            
            return view_name
            
        except Exception as e:
            logger.error(f"FromPandas conversion failed: {e}")
            raise


class ArrowBridge(BridgeOperator):
    """Optimized bridge using Apache Arrow for zero-copy where possible."""
    
    def to_pandas(self, duckdb_result: Any) -> pd.DataFrame:
        """Convert via Arrow for potentially better performance."""
        start_time = time.time()
        
        try:
            if hasattr(duckdb_result, 'arrow'):
                # DuckDB can produce Arrow directly
                arrow_table = duckdb_result.arrow()
                df = arrow_table.to_pandas()
            elif hasattr(duckdb_result, 'fetchdf'):
                # Fallback to regular conversion
                df = duckdb_result.fetchdf()
            else:
                raise TypeError(f"Cannot convert {type(duckdb_result)} via Arrow")
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.last_cost = BridgeCost(
                rows=len(df),
                columns=len(df.columns),
                conversion_time_ms=elapsed_ms
            )
            
            return df
            
        except Exception as e:
            logger.warning(f"Arrow conversion failed, falling back to standard: {e}")
            return ToPandas().execute(duckdb_result)
    
    def from_pandas(self, df: pd.DataFrame, db_adapter: Any, view_name: str) -> str:
        """Register DataFrame via Arrow for potentially better performance."""
        try:
            import pyarrow as pa
            
            start_time = time.time()
            
            # Convert to Arrow first
            arrow_table = pa.Table.from_pandas(df)
            
            # If db_adapter supports Arrow directly, use that
            if hasattr(db_adapter, 'register_arrow'):
                db_adapter.register_arrow(view_name, arrow_table)
            else:
                # Fallback to pandas registration
                db_adapter.register(view_name, df)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.last_cost = BridgeCost(
                rows=len(df),
                columns=len(df.columns),
                conversion_time_ms=elapsed_ms
            )
            
            return view_name
            
        except ImportError:
            logger.debug("PyArrow not available, using standard bridge")
            return FromPandas().execute(df, db_adapter, view_name)
        except Exception as e:
            logger.warning(f"Arrow registration failed, falling back to standard: {e}")
            return FromPandas().execute(df, db_adapter, view_name)
