"""
rag_chatbot.py
Database RAG Chatbot that uses pre-computed schema index and embeddings.
Make sure to run schema_indexer.py first to create the index.
"""

import json
import pickle
import requests
from typing import List, Dict, Tuple
import pyodbc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


class DatabaseRAGChatbot:
    def __init__(
        self,
        index_dir: str = './schema_index',
        ollama_url: str = "http://localhost:11434",
        model: str = "mistral"
    ):
        """
        Initialize the chatbot using pre-computed index
        
        Args:
            index_dir: Directory containing saved index and embeddings
            ollama_url: URL to Ollama instance
            model: Ollama model name
        """
        self.ollama_url = ollama_url
        self.model = model
        
        # Load index and embeddings
        self._load_index(index_dir)
        
        print(f"\n✓ Chatbot initialized with {len(self.table_index)} tables")
    
    def _load_index(self, index_dir: str) -> None:
        """Load pre-computed index and embeddings from disk"""
        print(f"\nLoading schema index from: {index_dir}")
        
        # Load table index
        index_file = os.path.join(index_dir, 'table_index.json')
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        with open(index_file, 'r', encoding='utf-8') as f:
            self.table_index = json.load(f)
        print(f"  ✓ Loaded table index: {len(self.table_index)} tables")
        
        # Load embeddings
        embeddings_file = os.path.join(index_dir, 'embeddings.pkl')
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        with open(embeddings_file, 'rb') as f:
            embeddings_data = pickle.load(f)
            self.model_name = embeddings_data['model_name']
            self.table_embeddings = embeddings_data['embeddings']
        print(f"  ✓ Loaded embeddings: {len(self.table_embeddings)} tables")
        
        # Load metadata
        metadata_file = os.path.join(index_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"  ✓ Loaded metadata")
        
        # Initialize embedding model for query encoding
        print(f"  Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
    
    def find_relevant_tables(self, user_query: str, top_k: int = 5) -> List[str]:
        """
        Find relevant tables using semantic similarity
        
        Args:
            user_query: The user's natural language query
            top_k: Number of tables to return
            
        Returns:
            List of relevant table names (Schema.TableName format)
        """
        # Encode the user query
        query_embedding = self.embedding_model.encode(user_query)
        
        # Calculate similarity with all tables
        scores = {}
        for table_name, table_embedding in self.table_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding],
                [table_embedding]
            )[0][0]
            scores[table_name] = similarity
        
        # Get top-k tables
        relevant = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [table_name for table_name, _ in relevant]
    
    def get_schema_for_tables(self, table_names: List[str]) -> str:
        """
        Build schema context string for the specified tables
        
        Args:
            table_names: List of table names to include
            
        Returns:
            Formatted schema string for LLM context
        """
        schema_text = "Database Schema:\n\n"
        
        for full_table_name in table_names:
            if full_table_name not in self.table_index:
                continue
            
            table_info = self.table_index[full_table_name]
            schema_name = table_info['schema']
            table_name = table_info['table']
            description = table_info['description']
            
            schema_text += f"Table: {schema_name}.{table_name}\n"
            
            if description:
                schema_text += f"Description: {description}\n"
            
            schema_text += "Columns:\n"
            for col in table_info['columns']:
                col_def = f"  - {col['name']} ({col['type']}"
                if not col.get('nullable', True):
                    col_def += " NOT NULL"
                col_def += ")"
                schema_text += col_def + "\n"
            
            if table_info['primary_keys']:
                schema_text += f"Primary Key: {', '.join(table_info['primary_keys'])}\n"
            
            if table_info['foreign_keys']:
                schema_text += "Foreign Keys:\n"
                for fk in table_info['foreign_keys']:
                    schema_text += f"  - {fk['column']} references {fk['references']}\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def generate_sql(self, user_query: str, relevant_tables: List[str]) -> str:
        """
        Generate SQL query using Ollama
        
        Args:
            user_query: Natural language query from user
            relevant_tables: List of relevant table names
            
        Returns:
            Generated SQL query
        """
        schema_context = self.get_schema_for_tables(relevant_tables)
        
        prompt = f"""You are a MSSQL expert. Generate a valid MSSQL query to answer the user's question.

{schema_context}

User Query: {user_query}

Important Instructions:
- Use exact table names with schema prefix (e.g., Sales.ShoppingCartItem, Production.Product)
- Only use columns that exist in the schema
- Return ONLY the SQL query, nothing else
- Do not include markdown formatting, backticks, or any explanation
- The query must be valid MSSQL T-SQL syntax
- Use appropriate JOINs if multiple tables are needed
- Add WHERE clauses if the user specifies filtering conditions

SQL Query:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40
                },
                timeout=120
            )
            
            response.raise_for_status()
            sql_query = response.json()['response'].strip()
            
            # Clean up response (remove markdown if present)
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            return sql_query
        
        except requests.exceptions.ConnectionError:
            return f"Error: Cannot connect to Ollama at {self.ollama_url}. Make sure Ollama is running."
        except Exception as e:
            return f"Error generating SQL: {str(e)}"
    
    def validate_sql(self, sql_query: str) -> Tuple[bool, str]:
        """
        Validate SQL query for safety
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        dangerous_keywords = ['DELETE', 'DROP', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        
        sql_upper = sql_query.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Query contains dangerous keyword: {keyword}"
        
        if not sql_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        return True, "Valid"
    
    def execute_query(
        self,
        sql_query: str,
        connection_string: str,
        max_rows: int = 100
    ) -> List[Dict]:
        """
        Execute SQL query against MSSQL database
        
        Args:
            sql_query: SQL query to execute
            connection_string: MSSQL connection string
            max_rows: Maximum rows to return
            
        Returns:
            List of result rows as dictionaries
        """
        try:
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            columns = [col[0] for col in cursor.description]
            
            results = []
            for i, row in enumerate(cursor.fetchall()):
                if i >= max_rows:
                    break
                
                row_dict = {}
                for col, val in zip(columns, row):
                    # Handle datetime serialization
                    if hasattr(val, 'isoformat'):
                        row_dict[col] = val.isoformat()
                    elif val is None:
                        row_dict[col] = None
                    else:
                        row_dict[col] = str(val) if not isinstance(val, (int, float, bool)) else val
                
                results.append(row_dict)
            
            cursor.close()
            conn.close()
            
            return results
        
        except pyodbc.Error as e:
            return [{"error": f"Database error: {str(e)}"}]
        except Exception as e:
            return [{"error": f"Execution error: {str(e)}"}]
    
    def chat(
        self,
        user_query: str,
        connection_string: str,
        top_k_tables: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: find tables -> generate SQL -> validate -> execute
        
        Args:
            user_query: User's natural language query
            connection_string: MSSQL connection string
            top_k_tables: Number of relevant tables to consider
            verbose: Print detailed output
            
        Returns:
            Dictionary with results and metadata
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Query: {user_query}")
            print('='*70)
        
        # Step 1: Find relevant tables
        if verbose:
            print("\n[Step 1] Finding relevant tables...")
        relevant_tables = self.find_relevant_tables(user_query, top_k=top_k_tables)
        
        if verbose:
            for i, table in enumerate(relevant_tables, 1):
                desc = self.table_index[table]['description']
                print(f"  {i}. {table}")
                if desc:
                    print(f"     {desc}")
        
        # Step 2: Generate SQL
        if verbose:
            print("\n[Step 2] Generating SQL query...")
        sql_query = self.generate_sql(user_query, relevant_tables)
        
        if verbose:
            print(f"\nGenerated SQL:\n{sql_query}\n")
        
        # Step 3: Validate SQL
        if verbose:
            print("[Step 3] Validating SQL...")
        is_valid, validation_msg = self.validate_sql(sql_query)
        
        if not is_valid:
            if verbose:
                print(f"✗ Validation failed: {validation_msg}")
            return {
                "user_query": user_query,
                "relevant_tables": relevant_tables,
                "generated_sql": sql_query,
                "error": validation_msg,
                "success": False
            }
        
        if verbose:
            print(f"✓ Validation passed")
        
        # Step 4: Execute query
        if verbose:
            print("\n[Step 4] Executing query...")
        results = self.execute_query(sql_query, connection_string)
        
        if verbose:
            print(f"✓ Query executed successfully")
            print(f"✓ Retrieved {len(results)} rows")
        
        return {
            "user_query": user_query,
            "relevant_tables": relevant_tables,
            "generated_sql": sql_query,
            "results": results,
            "result_count": len(results),
            "success": True
        }


# Main execution
if __name__ == "__main__":
    # Configuration
    INDEX_DIR = "C:/AdventureWorksRAG/schema_index"  
    OLLAMA_URL = "http://127.0.0.1:11434"
    OLLAMA_MODEL = "mistral:latest"
    
    # MSSQL Connection String
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=LAPTOP-RDP9UNKG\\SQLEXPRESS;"
        "DATABASE=AdventureWorks2014;"
        "Trusted_Connection=yes;"
    )
    
    try:
        # Initialize chatbot
        print("Initializing RAG Chatbot...")
        chatbot = DatabaseRAGChatbot(
            index_dir=INDEX_DIR,
            ollama_url=OLLAMA_URL,
            model=OLLAMA_MODEL
        )
        
        # Example queries
        test_queries = [
            "Show number of employees per department.",
            "Find average list price per product category.",
            "List employees hired in the last 5 years."
    ]
        
        # Run queries
        for query in test_queries:
            response = chatbot.chat(
                user_query=query,
                connection_string=connection_string,
                top_k_tables=5,
                verbose=True
            )
            
            if response['success']:
                print("\nResults:")
                for row in response['results'][:5]:
                    print(f"  {row}")
                if response['result_count'] > 5:
                    print(f"  ... and {response['result_count'] - 5} more rows")
            else:
                print(f"\nError: {response['error']}")
            
            print()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()