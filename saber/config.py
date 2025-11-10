"""
Configuration settings for the SABER engine.
"""
import os

# Backend types
SUPPORTED_BACKENDS = ['lotus', 'docetl', 'palimpzest']

# ============================================================================
# LLM Configuration (LiteLLM-based)
# ============================================================================

# Local VLLM Configuration (fallback when no API key provided)
LOCAL_VLLM_API_BASE = "http://localhost:51515/v1"
# LOCAL_VLLM_MODEL = "hosted_vllm/openai/gpt-oss-20b"
# LOCAL_VLLM_MODEL = "hosted_vllm/qwen/Qwen1.5-0.5B-Chat"
# LOCAL_VLLM_MODEL = "hosted_vllm/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
# LOCAL_VLLM_MODEL = "hosted_vllm/Qwen/Qwen2.5-7B-Instruct" 
LOCAL_VLLM_MODEL = "hosted_vllm/Qwen/Qwen3-4B-Instruct-2507"

# Local VLLM Embedding Model Configuration
LOCAL_EMBEDDING_API_BASE = "http://localhost:51516"
LOCAL_EMBEDDING_MODEL = "litellm_proxy/huggingface/Qwen/Qwen3-Embedding-0.6B"
# LOCAL_EMBEDDING_MODEL = "clip-ViT-B-32"
# LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

os.environ["LITELLM_PROXY_API_BASE"] = LOCAL_EMBEDDING_API_BASE
os.environ["LITELLM_PROXY_API_KEY"] = "-"

# Set environment variable for hosted VLLM
os.environ["HOSTED_VLLM_API_BASE"] = LOCAL_VLLM_API_BASE

# OpenAI Configuration (when API key is provided)
OPENAI_DEFAULT_MODEL = "openai/gpt-4o-mini-2024-07-18"
# OPENAI_DEFAULT_MODEL = "openai/gpt-4o-2024-08-06"
# OPENAI_DEFAULT_MODEL = "openai/gpt-4.1-nano-2025-04-14"
# OPENAI_DEFAULT_MODEL = "openai/gpt-4.1-mini-2025-04-14"
# OPENAI_DEFAULT_MODEL = "openai/gpt-4.1-2025-04-14"
# OPENAI_DEFAULT_MODEL = "openai/gpt-5-nano-2025-08-07"

OPENAI_DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Query Rewriter Configuration
# Default to local VLLM for query rewriting (can be overridden)
QUERY_REWRITER_DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
QUERY_REWRITER_API_BASE = None
# QUERY_REWRITER_DEFAULT_MODEL = LOCAL_VLLM_MODEL
# QUERY_REWRITER_API_BASE = LOCAL_VLLM_API_BASE

# Backend-specific LLM Configuration
# These will be used when API key is available
LOTUS_DEFAULT_LM_MODEL = OPENAI_DEFAULT_MODEL
DOCETL_DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
# PALIMPZEST_DEFAULT_MODEL = "gpt-4o-mini-2024-07-18" # old
PALIMPZEST_DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
# LOTUS_DEFAULT_LM_MODEL = LOCAL_VLLM_MODEL
# DOCETL_DEFAULT_MODEL = LOCAL_VLLM_MODEL
# PALIMPZEST_DEFAULT_MODEL = LOCAL_VLLM_MODEL

# Embedding Model Configuration
# LOTUS_DEFAULT_RM_MODEL = "intfloat/e5-base-v2"
LOTUS_DEFAULT_RM_MODEL = OPENAI_DEFAULT_EMBEDDING_MODEL
DOCETL_DEFAULT_EMBEDDING_MODEL = OPENAI_DEFAULT_EMBEDDING_MODEL
PALIMPZEST_DEFAULT_EMBEDDING_MODEL = OPENAI_DEFAULT_EMBEDDING_MODEL
# LOTUS_DEFAULT_RM_MODEL = LOCAL_EMBEDDING_MODEL
# DOCETL_DEFAULT_EMBEDDING_MODEL = LOCAL_EMBEDDING_MODEL
# PALIMPZEST_DEFAULT_EMBEDDING_MODEL = LOCAL_EMBEDDING_MODEL

# Demo Configuration
# Number of free queries allowed without API key
DEMO_FREE_QUERY_LIMIT = 5

# ============================================================================
# Backend-specific Configuration
# ============================================================================

# LOTUS Configuration
LOTUS_DEFAULT_DEDUP_THRESHOLD = 0.815

# DocETL Configuration
DOCETL_OUTPUT_PATH = "docetl_output.json"
DOCETL_DEFAULT_RESOLVE_THRESHOLD = 0.6

# Semantic operation thresholds
DEFAULT_COSINE_SIMILARITY_THRESHOLD = 0.85

# Column name separators
COLUMN_SEPARATOR = " ␟ "  # Use rarely used symbol as separator
