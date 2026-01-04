import json
import pickle
import os
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import numpy as np


class SchemaIndexer:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.model_name = embedding_model_name
        self.schema = None
        self.table_index = {}
        self.embeddings = {}
    
    def load_schema(self, schema_file: str) -> None:
        print(f"\nLoading schema from: {schema_file}")
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
            print(f"✓ Loaded {len(self.schema)} tables")
        except FileNotFoundError:
            print(f"✗ Error: Schema file '{schema_file}' not found")
            raise
        except json.JSONDecodeError:
            print(f"✗ Error: Invalid JSON in schema file")
            raise
    
    def build_table_index(self) -> None:
        print("\nBuilding table index...")
        
        for full_table_name, table_info in self.schema.items():
            # Parse schema and table name
            if '.' in full_table_name:
                schema_name, table_name = full_table_name.split('.', 1)
            else:
                schema_name, table_name = 'dbo', full_table_name
            
            # Extract column information
            columns_info = []
            for col in table_info['columns']:
                columns_info.append({
                    'name': col['name'],
                    'type': col['type'],
                    'nullable': col.get('nullable', True)
                })
            
            # Build index entry
            self.table_index[full_table_name] = {
                'schema': schema_name,
                'table': table_name,
                'description': table_info.get('description', ''),
                'columns': columns_info,
                'primary_keys': table_info.get('primary_keys', []),
                'foreign_keys': table_info.get('foreign_keys', [])
            }
        
        print(f"✓ Built index for {len(self.table_index)} tables")
    
    def create_embeddings(self) -> None:
        print("\nCreating vector embeddings...")
        
        for full_table_name, table_info in self.table_index.items():
            schema = table_info['schema']
            table = table_info['table']
            description = table_info['description']
            
            col_names = ' '.join([col['name'] for col in table_info['columns']])
            
            # Create comprehensive embedding text
            embedding_text = (
                f"Table {schema}.{table}: {description} "
                f"Columns: {col_names}"
            )
            
            # Generate embedding
            embedding = self.embedding_model.encode(embedding_text)
            self.embeddings[full_table_name] = embedding
        
        print(f"✓ Created embeddings for {len(self.embeddings)} tables")
    
    def save_index(self, output_dir: str = './schema_index') -> None:
        print(f"\nSaving index to: {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save table index as JSON (human readable)
        index_file = os.path.join(output_dir, 'table_index.json')
        with open(index_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_index = {}
            for table_name, info in self.table_index.items():
                json_index[table_name] = {
                    'schema': info['schema'],
                    'table': info['table'],
                    'description': info['description'],
                    'columns': info['columns'],
                    'primary_keys': info['primary_keys'],
                    'foreign_keys': info['foreign_keys']
                }
            json.dump(json_index, f, indent=2, ensure_ascii=False)
        print(f" Saved table index: {index_file}")
        
        # Save embeddings as pickle (binary, efficient)
        embeddings_file = os.path.join(output_dir, 'embeddings.pkl')
        embeddings_data = {
            'model_name': self.model_name,
            'embeddings': self.embeddings
        }
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
        print(f"Saved embeddings: {embeddings_file}")
        
        # Save metadata
        metadata_file = os.path.join(output_dir, 'metadata.json')
        metadata = {
            'embedding_model': self.model_name,
            'embedding_dimension': len(list(self.embeddings.values())[0]) if self.embeddings else 0,
            'total_tables': len(self.table_index),
            'created_from': 'schema_indexer.py'
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_file}")
        
        print(f"\nIndex saved successfully!")
    
    def process_schema(self, schema_file: str, output_dir: str = './schema_index') -> None:
        """
        Complete pipeline: load schema -> build index -> create embeddings -> save
        """
        print("="*70)
        print("SCHEMA INDEXER - Creating embeddings and index")
        print("="*70)
        
        self.load_schema(schema_file)
        self.build_table_index()
        self.create_embeddings()
        self.save_index(output_dir)
        
        print("\n" + "="*70)
        print("Schema indexing complete!")
        print("="*70)
        print(f"Pass '{output_dir}' as index_dir when initializing the chatbot.\n")



if __name__ == "__main__":
    SCHEMA_FILE = "C:/AdventureWorksRAG/database_schema.json"  
    OUTPUT_DIR = "./schema_index"          
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  
    
    # Create and process schema
    indexer = SchemaIndexer(embedding_model_name=EMBEDDING_MODEL)
    
    try:
        indexer.process_schema(
            schema_file=SCHEMA_FILE,
            output_dir=OUTPUT_DIR
        )
    except Exception as e:
        print(f"\nError during indexing: {e}")
        import traceback
        traceback.print_exc()