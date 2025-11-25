"""
Database adapter for multi-database support (DuckDB, SQLite, MySQL)
"""
import re
from typing import Optional, Any
import pandas as pd
import duckdb
import sqlite3
import logging

logger = logging.getLogger(__name__)

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    from sqlalchemy import create_engine, Engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class DatabaseAdapter:
    """Adapter for different database backends (DuckDB, SQLite, MySQL)."""
    
    def __init__(self, db_type: str = 'duckdb', **kwargs):
        self.db_type = db_type.lower()
        self.conn: Optional[Any] = None
        self.mysql_config: dict = kwargs if self.db_type == 'mysql' else {}
        self._sqlalchemy_engine: Optional['Engine'] = None
        self._table_name_mapping: dict = {}  # original_name -> normalized_name
        self._connect(**kwargs)
    
    @staticmethod
    def _fix_apostrophes_in_literals(sql: str, db_type: str) -> str:
        """
        Fix apostrophes/single quotes in string literals for SQL.
        Converts: 'Men\'s walk' -> 'Men''s walk' (SQL standard escaping)
        """
        result = []
        i = 0
        
        while i < len(sql):
            char = sql[i]
            
            # Check if this is the start of a string literal
            if char == "'":
                # Start collecting the string
                string_chars = ["'"]
                i += 1
                
                # Process characters inside the string
                while i < len(sql):
                    char = sql[i]
                    
                    if char == "\\" and i + 1 < len(sql) and sql[i + 1] == "'":
                        # Backslash-escaped quote: \' -> ''
                        string_chars.append("''")
                        i += 2
                        continue
                    elif char == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                        # Already SQL-escaped: '' -> keep as is
                        string_chars.append("''")
                        i += 2
                        continue
                    elif char == "'":
                        # Single quote: could be closing quote or internal apostrophe
                        # Heuristic: if followed by typical SQL delimiters/keywords, it's closing
                        # Otherwise, it's an internal apostrophe that needs escaping
                        next_char = sql[i + 1] if i + 1 < len(sql) else ''
                        
                        # Check if this looks like a closing quote
                        # It's closing if: end of string, followed by whitespace, comma, paren, semicolon
                        # OR followed by SQL keyword (FROM, WHERE, AND, OR, etc.)
                        is_closing = (
                            i + 1 >= len(sql) or
                            next_char in (' ', ',', ')', ';', '\n', '\t', '\r', '(') or
                            (i + 1 < len(sql) and sql[i+1:i+5].upper() == 'FROM') or
                            (i + 1 < len(sql) and sql[i+1:i+6].upper() in ('WHERE', 'ORDER', 'GROUP', 'LIMIT')) or
                            (i + 1 < len(sql) and sql[i+1:i+4].upper() in ('AND', 'OR\n', 'OR ', 'OR,', 'OR)', 'OR;'))
                        )
                        
                        if is_closing:
                            # Closing quote
                            string_chars.append("'")
                            i += 1
                            break
                        else:
                            # Internal apostrophe - escape it by doubling
                            string_chars.append("''")
                            i += 1
                            continue
                    else:
                        string_chars.append(char)
                        i += 1
                
                result.append(''.join(string_chars))
            else:
                result.append(char)
                i += 1
        
        return ''.join(result)
    
    @staticmethod
    def _convert_split_part(sql: str, db_type: str) -> str:
        """
        Convert SPLIT_PART to database-specific string splitting.
        DuckDB: SPLIT_PART(string, delimiter, index) - 1-indexed
        MySQL: SUBSTRING_INDEX - different semantics
        SQLite: SUBSTR/INSTR combination
        """
        if 'SPLIT_PART' not in sql.upper():
            return sql
            
        if db_type == 'mysql':
            def replace_mysql(match):
                string_expr, delimiter, index_str = match.groups()
                try:
                    idx = int(index_str.strip())
                    if idx == 1:
                        return f"SUBSTRING_INDEX({string_expr}, {delimiter}, 1)"
                    else:
                        # Get nth part: nested SUBSTRING_INDEX
                        return f"SUBSTRING_INDEX(SUBSTRING_INDEX({string_expr}, {delimiter}, {idx}), {delimiter}, -1)"
                except ValueError:
                    logger.warning(f"Non-constant SPLIT_PART index in MySQL: {index_str}")
                    return match.group(0)
            
            sql = re.sub(r'SPLIT_PART\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)', 
                        replace_mysql, sql, flags=re.IGNORECASE)
        
        elif db_type == 'sqlite':
            def replace_sqlite(match):
                string_expr, delimiter, index_str = match.groups()
                try:
                    idx = int(index_str.strip())
                    if idx == 1:
                        # First part: substr before first delimiter
                        return f"SUBSTR({string_expr}, 1, INSTR({string_expr}, {delimiter}) - 1)"
                    else:
                        logger.warning(f"SQLite SPLIT_PART with index > 1 not fully supported: {idx}")
                        return string_expr
                except ValueError:
                    return match.group(0)
            
            sql = re.sub(r'SPLIT_PART\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)', 
                        replace_sqlite, sql, flags=re.IGNORECASE)
        
        return sql
    
    @staticmethod
    def _normalize_substring(sql: str, db_type: str) -> str:
        """Normalize SUBSTRING functions."""
        if db_type == 'sqlite' and 'SUBSTRING' in sql.upper():
            sql = re.sub(r'\bSUBSTRING\b', 'SUBSTR', sql, flags=re.IGNORECASE)
        return sql
    
    @staticmethod
    def _normalize_cast(sql: str, db_type: str) -> str:
        """Normalize CAST to database-specific types."""
        if db_type == 'mysql':
            # INT32/INT64 -> SIGNED, FLOAT -> DECIMAL
            sql = re.sub(r'\bCAST\s*\(([^)]+)\s+AS\s+(?:INT32|INT64|int)\s*\)', 
                        r'CAST(\1 AS SIGNED)', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bCAST\s*\(([^)]+)\s+AS\s+FLOAT\s*\)', 
                        r'CAST(\1 AS DECIMAL(10,2))', sql, flags=re.IGNORECASE)
        
        elif db_type == 'sqlite':
            sql = re.sub(r'\bCAST\s*\(([^)]+)\s+AS\s+(?:INT32|INT64)\s*\)', 
                        r'CAST(\1 AS INTEGER)', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bCAST\s*\(([^)]+)\s+AS\s+FLOAT\s*\)', 
                        r'CAST(\1 AS REAL)', sql, flags=re.IGNORECASE)
        
        return sql
    
    def _connect(self, **kwargs):
        if self.db_type == 'duckdb':
            self.conn = duckdb.connect(kwargs.get('database', ':memory:'))
        elif self.db_type == 'sqlite':
            self.conn = sqlite3.connect(kwargs.get('database', ':memory:'))
            self.conn.row_factory = sqlite3.Row
        elif self.db_type == 'mysql':
            # For MySQL, use SQLAlchemy engine, not a direct connection
            # self.conn will be None, but _sqlalchemy_engine will be created on first use
            pass
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def register(self, name: str, df: pd.DataFrame) -> None:
        # Normalize table name: replace spaces with underscores, handle special chars
        # Remove leading/trailing underscores to avoid naming conflicts
        normalized_name = name.replace(' ', '_').replace("'", '').replace('"', '')
        normalized_name = re.sub(r'[^a-zA-Z0-9_]', '_', normalized_name)
        # Remove consecutive underscores and leading/trailing underscores
        normalized_name = re.sub(r'_+', '_', normalized_name).strip('_')
        # Ensure table name doesn't start with number (use 'table_' prefix to match preprocessing)
        if normalized_name and normalized_name[0].isdigit():
            normalized_name = 'table_' + normalized_name
        
        # Store mapping from original to normalized name
        self._table_name_mapping[name] = normalized_name
        
        # Check for empty DataFrame with no columns - occurs when filtering removes all rows
        if df.empty and len(df.columns) == 0:
            # Create a placeholder DataFrame to prevent registration errors
            # This allows queries to continue even when intermediate results are empty
            df = pd.DataFrame({'_placeholder': [None]})
            logger.debug(f"Empty DataFrame with no columns detected for table '{name}'. "
                        f"Registered placeholder table to allow query continuation.")
        
        if self.db_type == 'duckdb':
            self.conn.register(normalized_name, df)
            return
        
        # Drop temp tables before registration to prevent conflicts (except _sem_final which is special)
        if normalized_name.startswith(('temp_', '_sem')) and normalized_name != '_sem_final':
            self._drop_table_if_exists(normalized_name)
        elif normalized_name == '_sem_final':
            # For _sem_final, only drop if it exists AND we have data to replace it with
            if not df.empty:
                self._drop_table_if_exists(normalized_name)
        
        # Register table with transaction error retry
        for attempt in range(3):  # Increased from 2 to 3 attempts
            try:
                df.to_sql(normalized_name, self._get_sql_connection(), if_exists='replace', index=False)
                return
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle "table already exists" errors
                if 'already exists' in error_str and attempt < 2:
                    logger.warning(f"Table '{normalized_name}' already exists (attempt {attempt+1}). Dropping and retrying...")
                    self._drop_table_if_exists(normalized_name)
                    continue
                
                # Handle transaction errors
                if self._is_transaction_error(e) and attempt < 2:
                    logger.warning(f"Transaction error (attempt {attempt+1}). Resetting connection...")
                    self._reset_engine()
                    continue
                
                # Handle duplicate column errors by renaming columns
                if 'duplicate column' in error_str and attempt < 2:
                    logger.warning(f"Duplicate column detected. Renaming columns...")
                    # Rename duplicate columns
                    df = self._rename_duplicate_columns(df)
                    continue
                
                # Final attempt failed
                raise
    
    def _rename_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename duplicate columns by appending numeric suffixes."""
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            # Find all occurrences of the duplicate column
            dup_indices = cols[cols == dup].index.tolist()
            # Rename each occurrence with a numeric suffix
            for i, idx in enumerate(dup_indices[1:], start=2):
                cols[idx] = f"{dup}_{i}"
        df.columns = cols
        return df
    
    def _is_transaction_error(self, error: Exception) -> bool:
        """Check if error is transaction-related."""
        error_msg = str(error).lower()
        return 'transaction' in error_msg or 'rollback' in error_msg
    
    def _reset_engine(self) -> None:
        """Reset SQLAlchemy engine on transaction errors."""
        if self.db_type == 'mysql' and self._sqlalchemy_engine:
            self._sqlalchemy_engine.dispose()
            self._sqlalchemy_engine = None
    
    def _drop_table_if_exists(self, name: str) -> None:
        """Drop table if it exists (cleanup for temp tables)."""
        try:
            if self.db_type == 'mysql':
                # Use SQLAlchemy engine to avoid connection conflicts
                engine = self._get_sql_connection()
                with engine.begin() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {name}"))
            elif self.db_type == 'sqlite':
                self.conn.execute(f"DROP TABLE IF EXISTS {name}")
                self.conn.commit()
        except Exception:
            pass
    
    def _normalize_table_names_in_sql(self, sql: str) -> str:
        """
        Replace original table names with normalized names in SQL query.
        Uses the stored mapping to ensure queries work with normalized table names.
        """
        if not self._table_name_mapping:
            return sql
        
        # Sort by length (descending) to handle cases where one name is a substring of another
        sorted_names = sorted(self._table_name_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        
        normalized_sql = sql
        for original_name, normalized_name in sorted_names:
            if original_name == normalized_name:
                continue  # Skip if no normalization needed
            
            # Replace table names in FROM, JOIN, and table references
            # Handle quoted identifiers (backticks, double quotes, single quotes)
            patterns = [
                # Quoted with backticks: `table name` -> table_name
                (r'`' + re.escape(original_name) + r'`', normalized_name),
                # Quoted with double quotes: "table name" -> table_name
                (r'"' + re.escape(original_name) + r'"', normalized_name),
                # Quoted with single quotes (only for table names, not string literals)
                # This pattern is more careful - only after FROM/JOIN keywords
                (r'\b(FROM|JOIN)\s+\'' + re.escape(original_name) + r'\'', r'\1 ' + normalized_name),
                # Unquoted table names with word boundaries
                (r'\b(FROM|JOIN)\s+' + re.escape(original_name) + r'\b', r'\1 ' + normalized_name),
            ]
            
            for pattern, replacement in patterns:
                normalized_sql = re.sub(pattern, replacement, normalized_sql, flags=re.IGNORECASE)
        
        return normalized_sql
    
    def _get_sql_connection(self) -> Optional[Any]:
        if self.db_type == 'sqlite':
            return self.conn
        
        if self.db_type == 'mysql':
            if self._sqlalchemy_engine is None:
                if not SQLALCHEMY_AVAILABLE:
                    raise ImportError("SQLAlchemy required for MySQL. Install: pip install sqlalchemy")
                if not MYSQL_AVAILABLE:
                    raise ImportError("mysql-connector-python required for MySQL. Install: pip install mysql-connector-python")
                
                from urllib.parse import quote_plus
                
                config = self.mysql_config
                user = config.get('user', 'root')
                password = config.get('password', '')
                host = config.get('host', 'localhost')
                port = config.get('port', 3307)
                database = config.get('database', 'saber_db')
                
                password_part = f":{quote_plus(password)}" if password else ''
                connection_string = f"mysql+mysqlconnector://{user}{password_part}@{host}:{port}/{database}"
                
                self._sqlalchemy_engine = create_engine(
                    connection_string,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    pool_size=5,
                    max_overflow=10,
                    connect_args={'sql_mode': 'STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'}
                )
            
            return self._sqlalchemy_engine
        
        return None

    @staticmethod
    def normalize_sql_for_db(sql: str, db_type: str) -> str:
        """
        Normalize SQL syntax for specific database types.
        Handles database-specific function mappings and syntax differences.
        
        Args:
            sql: SQL query to normalize
            db_type: Database type ('mysql', 'sqlite', 'duckdb')
        
        Returns:
            Normalized SQL query
        """
        original_sql = sql
        
        # Fix apostrophes in string literals
        sql = DatabaseAdapter._fix_apostrophes_in_literals(sql, db_type)
        
        # Convert SPLIT_PART to database-specific functions
        sql = DatabaseAdapter._convert_split_part(sql, db_type)
        
        # Normalize SUBSTRING functions
        sql = DatabaseAdapter._normalize_substring(sql, db_type)
        
        # Normalize CAST operations
        sql = DatabaseAdapter._normalize_cast(sql, db_type)
        
        if db_type == 'mysql':
            # Fix UBIGINT typo -> UNSIGNED (common LLM mistake)
            if 'UBIGINT' in sql.upper():
                sql = re.sub(r'\bUBIGINT\b', 'UNSIGNED', sql, flags=re.IGNORECASE)
                logger.debug("Normalized UBIGINT typo to UNSIGNED for MySQL")
            
            # Fix column names with special characters (°, etc.) - escape them with backticks
            # Pattern: column_name containing special chars
            special_char_pattern = r'(\w*[°℃℉±×÷≤≥≠]+\w*)'
            if re.search(special_char_pattern, sql):
                # Find all column names with special characters
                matches = re.findall(r'\b(\w*[°℃℉±×÷≤≥≠]+[\w\s()]*)\b', sql)
                for match in set(matches):
                    if match and not match.startswith('`'):
                        # Only quote if it looks like a column name (appears after SELECT, FROM, WHERE, etc.)
                        sql = re.sub(
                            r'\b(SELECT|FROM|WHERE|ORDER BY|GROUP BY|,)\s+(' + re.escape(match) + r')\b',
                            r'\1 `\2`',
                            sql
                        )
                logger.debug(f"Escaped column names with special characters: {matches}")
            
            # Fix nested SUBSTRING_INDEX - warn about potential syntax errors
            if 'SUBSTRING_INDEX(SUBSTRING_INDEX' in sql or 'SUBSTRING_INDEX(SUBSTR' in sql:
                logger.warning("Detected nested SUBSTRING_INDEX which may cause syntax errors. Consider simplifying query.")
            
            # Fix MAX/MIN with multiple arguments (MySQL only accepts 1 argument)
            # Pattern: MAX(arg1, arg2, arg3, ...) -> GREATEST(arg1, arg2, arg3, ...)
            # Pattern: MIN(arg1, arg2, arg3, ...) -> LEAST(arg1, arg2, arg3, ...)
            def fix_max_min_multi_args(match):
                func_name = match.group(1).upper()
                args = match.group(2)
                
                # Count commas to determine if multiple arguments (exclude commas in nested functions)
                paren_depth = 0
                comma_count = 0
                for char in args:
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                    elif char == ',' and paren_depth == 0:
                        comma_count += 1
                
                # If multiple arguments, convert to GREATEST/LEAST
                if comma_count > 0:
                    replacement = 'GREATEST' if func_name == 'MAX' else 'LEAST'
                    logger.debug(f"Converted {func_name} with {comma_count + 1} args to {replacement}")
                    return f"{replacement}({args})"
                else:
                    # Single argument, keep as is
                    return match.group(0)
            
            sql = re.sub(r'\b(MAX|MIN)\s*\(([^)]+)\)', fix_max_min_multi_args, sql, flags=re.IGNORECASE)
            
            # MySQL reserved keywords need backtick quoting when used as column names
            # Quote common reserved keywords ONLY when they appear as column names, not as SQL keywords
            reserved_keywords = ['From', 'To', 'When', 'Date', 'Rank', 'Index', 'Key', 'Action', 'Status', 'Duration', 'Type', 'Group']
            for keyword in reserved_keywords:
                # Skip if already quoted
                if f'`{keyword}`' in sql or f'"{keyword}"' in sql:
                    continue
                
                # Pattern 1: After SELECT/FROM/WHERE/ORDER BY/GROUP BY, before comma or AS
                # This catches: SELECT From, To FROM table or SELECT From AS alias
                sql = re.sub(
                    r'\b(SELECT|,)\s+(' + keyword + r')\b(?=\s*(?:,|AS\s|\s+FROM))',
                    r'\1 `\2`',
                    sql,
                    flags=re.IGNORECASE
                )
                # Pattern 2: In WHERE/ON/CASE clauses - column = value
                sql = re.sub(
                    r'\b(WHERE|AND|OR|ON|WHEN|THEN|ELSE)\s+(' + keyword + r')\b(?=\s*[=<>!])',
                    r'\1 `\2`',
                    sql,
                    flags=re.IGNORECASE
                )
                # Pattern 3: ORDER BY/GROUP BY column names
                sql = re.sub(
                    r'\b(ORDER\s+BY|GROUP\s+BY)\s+(' + keyword + r')\b(?=\s*(?:ASC|DESC|,|$))',
                    r'\1 `\2`',
                    sql,
                    flags=re.IGNORECASE
                )
                # Pattern 4: In SUBSTRING_INDEX, STR_TO_DATE, and other functions
                # SUBSTRING_INDEX(From, ...) -> SUBSTRING_INDEX(`From`, ...)
                sql = re.sub(
                    r'(SUBSTRING_INDEX|STR_TO_DATE|DATE_FORMAT)\s*\(\s*(' + keyword + r')\s*,',
                    r'\1(`\2`, ',
                    sql,
                    flags=re.IGNORECASE
                )
                # Pattern 5: In CAST expressions - CAST(From AS ...)
                sql = re.sub(
                    r'\bCAST\s*\(\s*(' + keyword + r')\s+AS\s+',
                    r'CAST(`\1` AS ',
                    sql,
                    flags=re.IGNORECASE
                )
            
            # MySQL doesn't support STR_POSITION - use LOCATE instead
            # STR_POSITION(substring IN string) -> LOCATE(substring, string)
            if 'STR_POSITION' in sql.upper():
                sql = re.sub(r'STR_POSITION\s*\(\s*([^\s]+)\s+IN\s+([^)]+)\)', r'LOCATE(\1, \2)', sql, flags=re.IGNORECASE)
                if sql != original_sql:
                    logger.debug("Normalized STR_POSITION to LOCATE for MySQL")
            
            # MySQL doesn't support 'INT' or 'INTEGER' in CAST - use SIGNED/UNSIGNED instead
            if 'CAST' in sql.upper() and ('AS INT' in sql.upper() or 'AS INTEGER' in sql.upper()):
                original_cast = sql
                sql = re.sub(r'\bCAST\s*\(([^)]+)\s+AS\s+INT(EGER)?\s*\)', r'CAST(\1 AS SIGNED)', sql, flags=re.IGNORECASE)
                if sql != original_cast:
                    logger.debug("Normalized CAST AS INT/INTEGER to CAST AS SIGNED for MySQL")
            
            # MySQL doesn't support NULLS FIRST/LAST
            if 'NULLS' in sql.upper():
                sql = re.sub(r'\s+NULLS\s+(FIRST|LAST)', '', sql, flags=re.IGNORECASE)
            
            # Fix DATEDIFF - MySQL requires exactly 2 arguments
            # Replace 3-argument DATEDIFF(date1, date2, unit) with 2-argument DATEDIFF(date1, date2)
            if 'DATEDIFF' in sql.upper():
                original_datediff = sql
                sql = re.sub(r'DATEDIFF\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*[^)]+\)', r'DATEDIFF(\1, \2)', sql, flags=re.IGNORECASE)
                if sql != original_datediff:
                    logger.debug("Normalized 3-argument DATEDIFF to 2-argument for MySQL")
            
            # MySQL GROUP BY strict mode - IMPORTANT: Fix ANY_VALUE misuse
            # ANY_VALUE should ONLY wrap column references, not literal values or function arguments
            # Pattern:  ANY_VALUE('literal') or ANY_VALUE(123) -> should be 'literal' or 123
            def fix_anyvalue_literals(match):
                """Remove ANY_VALUE() wrapper from literal values and constants."""
                content = match.group(1).strip()
                
                # Check if it's a literal string (starts with quote)
                if content.startswith(("'", '"')):
                    return content
                
                # Check if it's a numeric literal
                try:
                    float(content.replace(',', ''))
                    return content
                except ValueError:
                    pass
                
                # Check if it's a negative number
                if content.startswith('-') and content[1:].replace('.', '').replace(',', '').isdigit():
                    return content
                
                # Keep ANY_VALUE for actual column references
                return match.group(0)
            
            # Fix ANY_VALUE wrapping literal values
            if 'ANY_VALUE' in sql:
                original_anyvalue = sql
                sql = re.sub(r'ANY_VALUE\s*\(\s*([^)]+)\s*\)', fix_anyvalue_literals, sql, flags=re.IGNORECASE)
                if sql != original_anyvalue:
                    logger.debug("Fixed ANY_VALUE misuse - removed from literal values")
            
        elif db_type == 'sqlite':
            # SQLite uses INTEGER instead of INT/SIGNED
            sql = re.sub(r'\bCAST\s*\(([^)]+)\s+AS\s+(INT|SIGNED|UNSIGNED)\)', r'CAST(\1 AS INTEGER)', sql, flags=re.IGNORECASE)
            
            # SQLite doesn't have ANY_VALUE - use MIN as a deterministic replacement
            # ANY_VALUE(column) -> MIN(column)
            if 'ANY_VALUE' in sql.upper():
                original_any = sql
                sql = re.sub(r'\bANY_VALUE\s*\(', r'MIN(', sql, flags=re.IGNORECASE)
                if sql != original_any:
                    logger.debug("Normalized ANY_VALUE to MIN for SQLite")
            
            # SQLite doesn't have SUBSTRING_INDEX - would need custom function
            # For now, just log a warning if detected
            if 'SUBSTRING_INDEX' in sql.upper():
                logger.warning("SUBSTRING_INDEX detected in SQLite query - this function may not be supported")
            
            # SQLite uses GROUP_CONCAT instead of STRING_AGG
            # STRING_AGG(column, delimiter) -> GROUP_CONCAT(column, delimiter)
            if 'STRING_AGG' in sql.upper():
                sql = re.sub(r'\bSTRING_AGG\s*\(', r'GROUP_CONCAT(', sql, flags=re.IGNORECASE)
                logger.debug("Normalized STRING_AGG to GROUP_CONCAT for SQLite")
            
            # SQLite uses || for string concatenation, not CONCAT function
            # CONCAT(str1, str2, str3) -> (str1 || str2 || str3)
            def replace_concat(match):
                args = match.group(1)
                # Split by commas not inside parentheses
                parts = []
                current = []
                depth = 0
                for char in args:
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                    elif char == ',' and depth == 0:
                        parts.append(''.join(current).strip())
                        current = []
                        continue
                    current.append(char)
                if current:
                    parts.append(''.join(current).strip())
                
                if len(parts) > 1:
                    result = '(' + ' || '.join(parts) + ')'
                    logger.debug(f"Converted CONCAT to || operator for SQLite")
                    return result
                else:
                    return parts[0] if parts else ''
            
            if 'CONCAT(' in sql.upper():
                sql = re.sub(r'\bCONCAT\s*\(([^)]+)\)', replace_concat, sql, flags=re.IGNORECASE)
                
        elif db_type == 'duckdb':
            # DuckDB uses INTEGER
            sql = re.sub(r'\bCAST\s*\(([^)]+)\s+AS\s+INT\)', r'CAST(\1 AS INTEGER)', sql, flags=re.IGNORECASE)
        
        return sql
    
    def execute(self, query: str) -> Any:
        # Normalize table names in query before execution
        query = self._normalize_table_names_in_sql(query)
        
        if self.db_type == 'duckdb':
            return self.conn.execute(query)
        
        if self.db_type == 'mysql':
            # Use SQLAlchemy for MySQL
            engine = self._get_sql_connection()
            with engine.connect() as conn:
                result = conn.execute(text(query))
                if not query.strip().upper().startswith('SELECT'):
                    conn.commit()
                return result
        
        # SQLite
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        if not query.strip().upper().startswith('SELECT'):
            self.conn.commit()
        
        return cursor
    
    def fetch_df(self, cursor: Optional[Any] = None) -> pd.DataFrame:
        if cursor is None:
            return pd.DataFrame()
        
        if self.db_type == 'duckdb':
            return cursor.fetch_df()
        
        if self.db_type == 'mysql':
            # SQLAlchemy CursorResult
            columns = list(cursor.keys())
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        
        # SQLite - regular cursor
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
    
    def sql(self, query: str) -> Any:
        # Normalize table names in query before execution
        query = self._normalize_table_names_in_sql(query)
        
        if self.db_type == 'duckdb':
            return self.conn.sql(query)
        
        cursor = self.execute(query)
        adapter = self
        return type('Result', (), {'df': lambda self: adapter.fetch_df(cursor)})()
    
    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
        
        if self._sqlalchemy_engine:
            self._sqlalchemy_engine.dispose()
            self._sqlalchemy_engine = None
