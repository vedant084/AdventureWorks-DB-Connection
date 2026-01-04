import json
import requests
from typing import List, Dict, Tuple
import pyodbc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DatabaseRAGChatbot:
    def __init__(self, schema_file: str, ollama_url: str = "http://127.0.0.1:11434", model: str = "mistralmistral:latest"):
        """
        Initialize the chatbot with schema file and Ollama connection
        
        Args:
            schema_file: Path to JSON file with database schema (keys are "Schema.TableName")
            ollama_url: URL to Ollama instance
            model: Ollama model name (default: mistral)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load schema
        with open(schema_file, 'r') as f:
            self.full_schema = json.load(f)
        
        print(f"Loaded {len(self.full_schema)} tables from schema")
        
        # Build table index and embeddings
        self.table_index = self._build_table_index()
        self.table_embeddings = self._embed_table_descriptions()
    
    def _build_table_index(self) -> Dict:
        """Create a lightweight index of tables with descriptions"""
        index = {}
        
        for full_table_name, table_info in self.full_schema.items():
            # Parse schema and table name
            schema_name, table_name = full_table_name.split('.') if '.' in full_table_name else ('dbo', full_table_name)
            
            # Build description from column names
            column_names = [col['name'] for col in table_info['columns']]
            description = f"{table_name}: {', '.join(column_names[:5])}"
            if len(column_names) > 5:
                description += f" and {len(column_names) - 5} more columns"
            
            index[full_table_name] = {
                'schema': schema_name,
                'table': table_name,
                'description': description,
                'columns': table_info['columns'],
                'primary_keys': table_info.get('primary_keys', []),
                'foreign_keys': table_info.get('foreign_keys', [])
            }
        
        return index
    
    def _embed_table_descriptions(self) -> Dict:
        """Create embeddings for semantic search"""
        embeddings = {}
        
        for full_table_name, info in self.table_index.items():
            # Create rich description for embedding
            schema, table = info['schema'], info['table']
            col_names = ' '.join([col['name'] for col in info['columns']])
            
            desc = f"{schema}.{table} {col_names}"
            embedding = self.embedding_model.encode(desc)
            embeddings[full_table_name] = embedding
        
        return embeddings
    
    def find_relevant_tables(self, user_query: str, top_k: int = 5) -> List[str]:
        """
        Use semantic search to find relevant tables for the user query
        Returns list of "Schema.TableName" strings
        """
        query_embedding = self.embedding_model.encode(user_query)
        
        # Calculate similarity scores
        scores = {}
        for table_name, table_embedding in self.table_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding],
                [table_embedding]
            )[0][0]
            scores[table_name] = similarity
        
        # Return top-k relevant tables sorted by score
        relevant = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        print(f"\nTable Relevance Scores:")
        for table_name, score in relevant:
            print(f"  {table_name}: {score:.4f}")
        
        return [table_name for table_name, _ in relevant]
    
    def get_schema_for_tables(self, table_names: List[str]) -> str:
        """
        Get full schema details only for specified tables
        """
        schema_text = "Database Schema:\n\n"
        
        for full_table_name in table_names:
            if full_table_name in self.table_index:
                info = self.table_index[full_table_name]
                schema_name = info['schema']
                table_name = info['table']
                
                schema_text += f"Table: {schema_name}.{table_name}\n"
                schema_text += "Columns:\n"
                
                for col in info['columns']:
                    col_def = f"  {col['name']} {col['type']}"
                    if col.get('nullable') == False:
                        col_def += " NOT NULL"
                    schema_text += col_def + "\n"
                
                if info['primary_keys']:
                    schema_text += f"Primary Key: {', '.join(info['primary_keys'])}\n"
                
                if info['foreign_keys']:
                    schema_text += "Foreign Keys:\n"
                    for fk in info['foreign_keys']:
                        schema_text += f"  {fk}\n"
                
                schema_text += "\n"
        
        return schema_text
    
    def generate_sql(self, user_query: str, relevant_tables: List[str]) -> str:
        """
        Use Ollama to generate SQL based on user query and relevant schema
        """
        schema_context = self.get_schema_for_tables(relevant_tables)
        
        prompt = f"""You are a MSSQL expert. Generate a valid MSSQL query to answer the user's question.

Database Schema:
{schema_context}

User Query: {user_query}

Important:
- Use the exact table names with schema prefix (e.g., Production.ScrapReason, HumanResources.Shift)
- Only use columns that exist in the schema
- Return ONLY the SQL query, no explanations or markdown
- Make sure the query is valid MSSQL syntax

SQL Query:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,  # Low temperature for deterministic SQL
                    "top_p": 0.9,
                    "top_k": 40
                },
                timeout=60
            )
            
            sql_query = response.json()['response'].strip()
            
            # Clean up the response (remove markdown formatting if present)
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            return sql_query
        
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None
    
    def validate_sql(self, sql_query: str) -> Tuple[bool, str]:
        """
        Basic SQL validation to prevent dangerous operations
        """
        dangerous_keywords = ['DELETE', 'DROP', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        
        sql_upper = sql_query.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Query contains dangerous keyword: {keyword}"
        
        if not sql_upper.startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        return True, "OK"
    
    def execute_query(self, sql_query: str, connection_string: str, max_rows: int = 100) -> List[Dict]:
        """
        Execute the generated SQL against MSSQL database
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
                # Convert to serializable format
                row_dict = {}
                for col, val in zip(columns, row):
                    if hasattr(val, 'isoformat'):  # datetime objects
                        row_dict[col] = val.isoformat()
                    else:
                        row_dict[col] = val
                results.append(row_dict)
            
            cursor.close()
            conn.close()
            
            return results
        
        except Exception as e:
            return [{"error": str(e)}]
    
    def chat(self, user_query: str, connection_string: str, top_k_tables: int = 5) -> Dict:
        """
        Complete RAG pipeline: query -> find tables -> generate SQL -> execute
        """
        print(f"\n{'='*70}")
        print(f"User Query: {user_query}")
        print('='*70)
        
        # Step 1: Find relevant tables
        print("\n[Step 1] Finding relevant tables...")
        relevant_tables = self.find_relevant_tables(user_query, top_k=top_k_tables)
        
        # Step 2: Generate SQL
        print("\n[Step 2] Generating SQL query...")
        sql_query = self.generate_sql(user_query, relevant_tables)
        
        if sql_query is None:
            return {
                "user_query": user_query,
                "error": "Failed to generate SQL",
                "relevant_tables": relevant_tables
            }
        
        print(f"Generated SQL:\n{sql_query}")
        
        # Step 3: Validate SQL
        print("\n[Step 3] Validating SQL...")
        is_valid, validation_msg = self.validate_sql(sql_query)
        
        if not is_valid:
            return {
                "user_query": user_query,
                "relevant_tables": relevant_tables,
                "generated_sql": sql_query,
                "error": validation_msg
            }
        
        # Step 4: Execute query
        print("\n[Step 4] Executing query...")
        results = self.execute_query(sql_query, connection_string)
        
        return {
            "user_query": user_query,
            "relevant_tables": relevant_tables,
            "generated_sql": sql_query,
            "results": results,
            "result_count": len(results)
        }
