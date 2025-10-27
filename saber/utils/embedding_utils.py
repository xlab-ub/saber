"""
Utility functions for embedding operations.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import pandas as pd

def compute_similarity_matches(df_left: pd.DataFrame, df_right: pd.DataFrame, 
                             embeddings_left: np.ndarray, embeddings_right: np.ndarray,
                             threshold: float = 0.85, operation: str = 'except') -> List[int]:
    """
    Compute similarity matches between two sets of embeddings.
    
    Args:
        df_left: Left DataFrame
        df_right: Right DataFrame  
        embeddings_left: Embeddings for left DataFrame
        embeddings_right: Embeddings for right DataFrame
        threshold: Similarity threshold
        operation: 'except' or 'intersect'
        
    Returns:
        List of indices to keep from left DataFrame
    """
    sim = cosine_similarity(embeddings_left, embeddings_right)
    keep = []
    right_available = np.ones(len(df_right), dtype=int)
    
    for i in range(len(df_left)):
        # Find the best match in the right DataFrame
        best_match_idx = np.argmax(sim[i])
        best_similarity = sim[i, best_match_idx]
        
        if operation == 'except':
            # For EXCEPT: keep if no good match found OR if match is already used
            if best_similarity < threshold or right_available[best_match_idx] == 0:
                keep.append(i)
            else:
                # Mark this right row as used
                right_available[best_match_idx] = 0
        elif operation == 'intersect':
            # For INTERSECT: keep if good match found AND match is still available
            if best_similarity >= threshold and right_available[best_match_idx] > 0:
                keep.append(i)
                # Mark this right row as used
                right_available[best_match_idx] = 0
    
    return keep

def encode_dataframe_for_similarity(df: pd.DataFrame, cols: List[str], 
                                   encoder, separator: str = " ␟ ") -> np.ndarray:
    """
    Encode DataFrame rows for similarity computation.
    
    Args:
        df: Input DataFrame
        cols: Columns to include in encoding
        encoder: Encoding function/model
        separator: Separator between column values
        
    Returns:
        Encoded embeddings array
    """
    def _to_sentences(df: pd.DataFrame) -> List[str]:
        return (df[cols].astype(str).agg(separator.join, axis=1).tolist())

    sentences = _to_sentences(df)
    return encoder(sentences)
