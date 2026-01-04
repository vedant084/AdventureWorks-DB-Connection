import json
import requests
from typing import List, Dict, Tuple
import pyodbc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sql_generater_functions import DatabaseRAGChatbot
try:
    chatbot = DatabaseRAGChatbot(
        schema_file="C:/AdventureWorksRAG/adventureworks_schema.json",  
        ollama_url="http://127.0.0.1:11434",
        model="mistral:latest"  
    )
        
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=LAPTOP-RDP9UNKG\\SQLEXPRESS;"
        "DATABASE=AdventureWorks2014;"
        "Trusted_Connection=yes;"
    )

    queries = [
        "Show number of employees per department.",
        "Find average list price per product category.",
        "List employees hired in the last 5 years."
    ]
        
    for query in queries:
        response = chatbot.chat(query, connection_string, top_k_tables=5)
            
        if "error" in response:
            print(f"\nError: {response['error']}")
        else:
            print(f"\nResults ({response['result_count']} rows):")
            for result in response['results'][:10]:
                print(f"  {result}")
        
        
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()