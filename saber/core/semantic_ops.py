"""
Backend-agnostic semantic operations for set theory operations.
These operations use embedding-based similarity and delegate to backends for semantic distinct.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Protocol
import logging
from sklearn.metrics.pairwise import cosine_similarity

from ..config import DEFAULT_COSINE_SIMILARITY_THRESHOLD
from ..utils.embedding_utils import encode_dataframe_for_similarity

logger = logging.getLogger(__name__)

class SemanticBackendProtocol(Protocol):
    """Protocol for backends that support semantic distinct operations."""
    
    def sem_distinct(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Perform semantic deduplication on the specified column."""
        ...


class SemanticSetOperations:
    """
    Service class for semantic set operations that need backend support.
    
    This follows the dependency inversion principle by depending on abstractions
    rather than concrete backend implementations.
    """
    
    def __init__(self, backends: dict):
        """
        Initialize with available backends.
        
        Args:
            backends: Dictionary mapping backend names to backend instances
        """
        self.backends = backends
    
    def except_operation(self, df_left: pd.DataFrame, df_right: pd.DataFrame,
                        rm, is_set: bool = True, threshold: float = DEFAULT_COSINE_SIMILARITY_THRESHOLD, 
                        cols: Optional[List[str]] = None, 
                        distinct_backend: str = 'lotus') -> pd.DataFrame:
        """
        Semantic difference between two DataFrames.
        
        Args:
            df_left: Left DataFrame
            df_right: Right DataFrame  
            rm: Retrieval model for embeddings
            is_set: True for set semantics (remove duplicates), False for bag semantics
            threshold: Similarity threshold for matching
            cols: Columns to use for similarity (defaults to all columns)
            distinct_backend: Backend to use for semantic distinct ('lotus' or 'docetl')
        
        Returns:
            DataFrame with semantic difference
        """
        if rm is None:
            raise ValueError("Retrieval model (rm) is required for semantic operations")

        # Use only common columns if cols not specified
        if cols is None:
            cols = list(set(df_left.columns) & set(df_right.columns))
            if not cols:
                raise ValueError("No common columns found between left and right DataFrames")
        
        # Use utility function for encoding
        emb_left = encode_dataframe_for_similarity(df_left, cols, rm)
        emb_right = encode_dataframe_for_similarity(df_right, cols, rm)
        
        sim = cosine_similarity(emb_left, emb_right)

        keep = np.ones(len(df_left), dtype=bool)
        right_available = np.ones(len(df_right), dtype=int)

        for i in range(len(df_left)):
            # find the most similar row in the right DataFrame
            j = sim[i].argmax()
            # if the similarity is above the threshold
            # and the corresponding right row is still available
            if sim[i, j] >= threshold and right_available[j]:
                right_available[j] -= 1     # consume one
                keep[i] = False             # mark left row for removal

        result = df_left[keep].copy()

        if is_set:
            # For set semantics, apply semantic distinct using the specified backend
            result = result.reset_index(drop=True)
            logging.info(f"Semantic difference result (set): {result.head(10)}")
            
            if distinct_backend in self.backends:
                backend = self.backends[distinct_backend]
                # Apply semantic deduplication using the backend
                for col in cols:
                    result = backend.sem_distinct(result, col)
                    logging.info(f"Distinct result for column '{col}': {result.head(10)}")
            else:
                # Fallback to regular deduplication
                logging.warning(f"Backend '{distinct_backend}' not available, using regular deduplication")
                result = result.drop_duplicates(subset=cols).reset_index(drop=True)

        return result

    def intersect_operation(self, df_left: pd.DataFrame, df_right: pd.DataFrame,
                           rm, is_set: bool = True, threshold: float = DEFAULT_COSINE_SIMILARITY_THRESHOLD, 
                           cols: Optional[List[str]] = None,
                           distinct_backend: str = 'lotus') -> pd.DataFrame:
        """
        Semantic intersection between two DataFrames.
        
        Args:
            df_left: Left DataFrame
            df_right: Right DataFrame
            rm: Retrieval model for embeddings
            is_set: True for set semantics (remove duplicates), False for bag semantics
            threshold: Similarity threshold for matching
            cols: Columns to use for similarity (defaults to all columns)
            distinct_backend: Backend to use for semantic distinct ('lotus' or 'docetl')
        
        Returns:
            DataFrame with semantic intersection
        """
        if rm is None:
            raise ValueError("Retrieval model (rm) is required for semantic operations")

        # Use only common columns if cols not specified
        if cols is None:
            cols = list(set(df_left.columns) & set(df_right.columns))
            if not cols:
                raise ValueError("No common columns found between left and right DataFrames")
        
        # Use utility function for encoding
        emb_left = encode_dataframe_for_similarity(df_left, cols, rm)
        emb_right = encode_dataframe_for_similarity(df_right, cols, rm)
        
        sim = cosine_similarity(emb_left, emb_right)

        keep = []
        right_available = np.ones(len(df_right), dtype=int)

        for i in range(len(df_left)):
            # find the most similar row in the right DataFrame
            j = sim[i].argmax()
            # if the similarity is above the threshold
            # and the corresponding right row is still available
            if sim[i, j] >= threshold and right_available[j]:
                right_available[j] -= 1     # consume one
                keep.append(i)              # mark left row for inclusion

        result = df_left.iloc[keep].copy()

        if is_set:
            # For set semantics, apply semantic distinct using the specified backend
            result = result.reset_index(drop=True)
            logging.info(f"Semantic intersection result (set): {result.head(10)}")
            
            if distinct_backend in self.backends:
                backend = self.backends[distinct_backend]
                # Apply semantic deduplication using the backend
                for col in cols:
                    result = backend.sem_distinct(result, col)
                    logging.info(f"Distinct result for column '{col}': {result.head(10)}")
            else:
                # Fallback to regular deduplication
                logging.warning(f"Backend '{distinct_backend}' not available, using regular deduplication")
                result = result.drop_duplicates(subset=cols).reset_index(drop=True)

        return result


# Convenience functions that create a service instance and delegate
def sem_except(df_left: pd.DataFrame, df_right: pd.DataFrame,
               rm, backends: dict, is_set: bool = True, threshold: float = DEFAULT_COSINE_SIMILARITY_THRESHOLD, 
               cols: Optional[List[str]] = None, 
               distinct_backend: str = 'lotus') -> pd.DataFrame:
    """
    Semantic difference between two DataFrames.
    
    This is a convenience function that creates a SemanticSetOperations service
    and delegates to it. This maintains backward compatibility while providing
    a cleaner architecture.
    """
    service = SemanticSetOperations(backends)
    return service.except_operation(df_left, df_right, rm, is_set, threshold, cols, distinct_backend)


def sem_intersect(df_left: pd.DataFrame, df_right: pd.DataFrame,
                  rm, backends: dict, is_set: bool = True, threshold: float = DEFAULT_COSINE_SIMILARITY_THRESHOLD, 
                  cols: Optional[List[str]] = None,
                  distinct_backend: str = 'lotus') -> pd.DataFrame:
    """
    Semantic intersection between two DataFrames.
    
    This is a convenience function that creates a SemanticSetOperations service
    and delegates to it. This maintains backward compatibility while providing  
    a cleaner architecture.
    """
    service = SemanticSetOperations(backends)
    return service.intersect_operation(df_left, df_right, rm, is_set, threshold, cols, distinct_backend)
