"""
LLM Configuration Management for SABER.

This module provides a unified interface for managing LLM configurations
across different providers (OpenAI, local VLLM, etc.) using LiteLLM.
"""
import os
from typing import Optional, Dict, Any
import logging

import litellm

from .config import (
    LOCAL_VLLM_API_BASE,
    LOCAL_VLLM_MODEL,
    OPENAI_DEFAULT_MODEL,
    QUERY_REWRITER_DEFAULT_MODEL,
    QUERY_REWRITER_API_BASE,
    LOTUS_DEFAULT_LM_MODEL,
    DOCETL_DEFAULT_MODEL,
    PALIMPZEST_DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)

class LLMConfig:
    """
    Manages LLM configuration for SABER engine.
    
    Supports:
    - OpenAI models (with API key)
    - Local VLLM models (without API key)
    - Easy switching between providers
    """
    
    def __init__(self, api_key: Optional[str] = None, use_local: bool = False):
        """
        Initialize LLM configuration.
        
        Args:
            api_key: OpenAI API key. If None, will check environment variable.
            use_local: Force use of local VLLM. Default is False (use OpenAI if key available).
        """
        # If not forcing local and no API key provided, try environment variable
        if not use_local and api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        self.api_key = api_key
        # Only use local if explicitly requested OR no API key is available
        self.use_local = use_local or (api_key is None)
        
        # Configure LiteLLM
        self._configure_litellm()
    
    def _configure_litellm(self):
        """Configure LiteLLM settings."""
        # Suppress unnecessary logging
        litellm.suppress_debug_info = True
        
        # Set API key if available
        if self.api_key and not self.use_local:
            os.environ["OPENAI_API_KEY"] = self.api_key
    
    def get_model_config(self, component: str) -> Dict[str, Any]:
        """
        Get model configuration for a specific component.
        
        Args:
            component: One of 'query_rewriter', 'lotus', 'docetl', 'palimpzest'
            
        Returns:
            Dictionary with 'model' and 'api_base' keys
        """
        if self.use_local:
            # Use local VLLM for ALL components when use_local is True
            return {
                'model': LOCAL_VLLM_MODEL,
                'api_base': LOCAL_VLLM_API_BASE,
                'api_key': 'dummy',  # VLLM doesn't need real API key
            }
        else:
            # Use OpenAI models for all components
            if component == 'query_rewriter':
                return {
                    'model': QUERY_REWRITER_DEFAULT_MODEL,
                    'api_base': QUERY_REWRITER_API_BASE,
                    'api_key': self.api_key,
                }
            elif component == 'lotus':
                return {
                    'model': LOTUS_DEFAULT_LM_MODEL,
                    'api_base': None,
                    'api_key': self.api_key,
                }
            elif component == 'docetl':
                return {
                    'model': DOCETL_DEFAULT_MODEL,
                    'api_base': None,
                    'api_key': self.api_key,
                }
            elif component == 'palimpzest':
                return {
                    'model': PALIMPZEST_DEFAULT_MODEL,
                    'api_base': None,
                    'api_key': self.api_key,
                }
            else:
                raise ValueError(f"Unknown component: {component}")
    
    def update_api_key(self, api_key: Optional[str], use_local: bool = False):
        """
        Update API key and reconfigure.
        
        Args:
            api_key: New OpenAI API key
            use_local: Whether to use local VLLM
        """
        self.api_key = api_key
        self.use_local = use_local or (api_key is None)
        self._configure_litellm()
    
    def is_using_local(self) -> bool:
        """Check if currently using local VLLM."""
        return self.use_local
    
    def get_query_rewriter_model(self) -> str:
        """Get the model name for query rewriter."""
        config = self.get_model_config('query_rewriter')
        return config['model']
    
    def get_query_rewriter_api_base(self) -> Optional[str]:
        """Get the API base for query rewriter."""
        config = self.get_model_config('query_rewriter')
        return config['api_base']
    
    @staticmethod
    def test_local_vllm() -> bool:
        """
        Test if local VLLM is available and working.
        
        Returns:
            True if local VLLM is accessible, False otherwise
        """
        try:
            response = litellm.completion(
                model=LOCAL_VLLM_MODEL,
                api_base=LOCAL_VLLM_API_BASE,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=5,
            )
            return True
        except Exception as e:
            logging.error(f"Local VLLM not available: {e}")
            return False
    
    @staticmethod
    def test_openai_key(api_key: str) -> bool:
        """
        Test if OpenAI API key is valid.
        
        Args:
            api_key: OpenAI API key to test
            
        Returns:
            True if key is valid, False otherwise
        """
        try:
            response = litellm.completion(
                model=OPENAI_DEFAULT_MODEL,
                api_key=api_key,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=5,
            )
            return True
        except Exception as e:
            logging.error(f"OpenAI API key test failed: {e}")
            return False


def get_default_llm_config(api_key: Optional[str] = None, use_local: bool = False) -> LLMConfig:
    """
    Get default LLM configuration.
    
    Args:
        api_key: Optional OpenAI API key. If None and use_local=False, will check environment variable.
        use_local: Force use of local VLLM. Default is False.
        
    Returns:
        LLMConfig instance
    """
    return LLMConfig(api_key=api_key, use_local=use_local)
