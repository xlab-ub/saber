# SABER: A SQL-Compatible Semantic Document Processing System Based on Extended Relational Algebra

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2509.00277" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
</p>

SABER is a research system that integrates multiple semantic document processing frameworks (LOTUS, DocETL, Palimpzest) with a unified SQL-compatible interface.

## Installation

### Development Installation
```bash
git clone https://github.com/xlab-ub/saber.git
cd saber
pip install -e .[all]
```

<!-- Due to potential dependency conflicts between the integrated frameworks, SABER offers flexible installation options: -->

<!-- ### Option 1: Core Installation (Recommended for most users)
```bash
pip install saber-query
``` -->

<!-- ### Option 1: Install with Specific Frameworks
```bash
# For Lotus AI support
pip install saber-query[lotus]

# For DocETL support  
pip install saber-query[docetl]

# For Palimpzest support
pip install saber-query[palimpzest]

# For Lotus + DocETL
pip install saber-query[lotus-docetl]

# For all frameworks (experimental - may have conflicts)
pip install saber-query[all]
```

### Option 2: Development Installation
```bash
git clone https://github.com/xlab-ub/saber.git
cd saber
pip install -e .[all]
``` -->

### Handling Dependency Conflicts

If you encounter dependency conflicts:

**Use conda environments and force installation scripts**:
```bash
conda create -n saber python=3.12 -y
conda activate saber
git clone https://github.com/xlab-ub/saber.git
cd saber
./scripts/install_all_force.sh
```

<!-- 1. **Use virtual environments** (recommended):
   ```bash
   python -m venv saber_env
   source saber_env/bin/activate
   pip install saber-query[lotus-docetl]
   ```

2. **Use conda environments**:
   ```bash
   conda create -n saber python=3.10
   conda activate saber
   pip install saber-query[all]
   ```

3. **Force install everything (for advanced users)**:
   ```bash
   # Method 1: Use pip's conflict resolution (recommended)
   pip install --upgrade pip  # Ensure latest pip with better resolver
   pip install saber-query[all]
   
   # Method 2: Force install with conflict override
   pip install saber-query[all] --force-reinstall --no-deps
   pip install openai duckdb  # Install core deps separately
   
   # Method 3: Sequential installation
   pip install lotus-ai docetl  # Install compatible pair first
   pip install palimpzest --force-reinstall  # Add conflicting package
   pip install saber-query --no-deps  # Install SABER without deps
   
   # Method 4: Ignore dependency conflicts entirely
   pip install --no-deps lotus-ai docetl palimpzest
   pip install saber-query --no-deps
   pip install openai duckdb  # Install only essential deps
   ```

4. **Use automated force installation scripts**:
   ```bash
   git clone https://github.com/xlab-ub/saber.git
   cd saber
   ./scripts/install_all_force.sh
   ``` -->

## Running Examples

### Semantic Operations Examples

#### Semantic WHERE
Filter rows based on semantic conditions rather than exact matches.
```bash
python examples/semantic_ops_examples/semantic_where.py
```

#### Semantic SELECT
Extract and transform columns using semantic understanding and natural language instructions.
```bash
python examples/semantic_ops_examples/semantic_select.py
```

#### Semantic JOIN
Join tables based on semantic relationships rather than exact key matches.
```bash
python examples/semantic_ops_examples/semantic_join.py
```

#### Semantic GROUP BY
Group records by semantic similarity or conceptual categories.
```bash
python examples/semantic_ops_examples/semantic_group_by.py
```

#### Semantic AGGREGATION
Perform aggregations with semantic understanding of the data.
```bash
python examples/semantic_ops_examples/semantic_aggregation.py
```

#### Semantic ORDER BY
Sort results based on semantic criteria like relevance, similarity, or conceptual ordering.
```bash
python examples/semantic_ops_examples/semantic_order_by.py
```

#### Semantic DISTINCT
Remove duplicates based on semantic similarity rather than exact matches.
```bash
python examples/semantic_ops_examples/semantic_distinct.py
```

#### Semantic INTERSECT (ALL) and EXCEPT (ALL)
Perform semantic (INTERSECT, EXCEPT) operations based on semantic relationships.
```bash
# Semantic INTERSECT - Find semantically overlapping records
python examples/semantic_ops_examples/semantic_intersect.py
python examples/semantic_ops_examples/semantic_intersect_all.py

# Semantic EXCEPT - Find semantically different records  
python examples/semantic_ops_examples/semantic_except.py
python examples/semantic_ops_examples/semantic_except_all.py
```

### Unified Query Examples

#### Backend-Agnostic Semantic Query Rewriting
Demonstrates how SABER automatically rewrites backend-free semantic queries to work with different Semantic Data Processing Systems (LOTUS, DocETL, Palimpzest) without requiring users to modify their code.
```bash
python examples/unified_query_examples/unified_query.py
```

## Citation

If you find this code useful, please consider citing our paper:
```bibtex
@misc{lee2025sabersqlcompatiblesemanticdocument,
      title={SABER: A SQL-Compatible Semantic Document Processing System Based on Extended Relational Algebra}, 
      author={Changjae Lee and Zhuoyue Zhao and Jinjun Xiong},
      year={2025},
      eprint={2509.00277},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2509.00277}, 
}
```
