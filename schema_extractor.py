import pyodbc
import json
from collections import defaultdict

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=LAPTOP-RDP9UNKG\\SQLEXPRESS;"
    "DATABASE=AdventureWorks2014;"
    "Trusted_Connection=yes;"
)

cursor = conn.cursor()

schema = defaultdict(lambda: {
    "columns": [],
    "primary_keys": [],
    "foreign_keys": []
})

# Columns
cursor.execute("""
SELECT
    s.name, t.name, c.name, ty.name, c.is_nullable
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
JOIN sys.columns c ON t.object_id = c.object_id
JOIN sys.types ty ON c.user_type_id = ty.user_type_id
""")

for s, t, c, ty, nullable in cursor.fetchall():
    schema[f"{s}.{t}"]["columns"].append({
        "name": c,
        "type": ty,
        "nullable": bool(nullable)
    })

# Primary Keys
cursor.execute("""
SELECT
    s.name, t.name, c.name
FROM sys.indexes i
JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
JOIN sys.tables t ON i.object_id = t.object_id
JOIN sys.schemas s ON t.schema_id = s.schema_id
WHERE i.is_primary_key = 1
""")

for s, t, c in cursor.fetchall():
    schema[f"{s}.{t}"]["primary_keys"].append(c)

# Foreign Keys
cursor.execute("""
SELECT
    sch1.name, tab1.name, col1.name,
    sch2.name, tab2.name, col2.name
FROM sys.foreign_key_columns fkc
JOIN sys.tables tab1 ON fkc.parent_object_id = tab1.object_id
JOIN sys.columns col1 ON fkc.parent_object_id = col1.object_id
    AND fkc.parent_column_id = col1.column_id
JOIN sys.tables tab2 ON fkc.referenced_object_id = tab2.object_id
JOIN sys.columns col2 ON fkc.referenced_object_id = col2.object_id
    AND fkc.referenced_column_id = col2.column_id
JOIN sys.schemas sch1 ON tab1.schema_id = sch1.schema_id
JOIN sys.schemas sch2 ON tab2.schema_id = sch2.schema_id
""")

for s1, t1, c1, s2, t2, c2 in cursor.fetchall():
    schema[f"{s1}.{t1}"]["foreign_keys"].append({
        "column": c1,
        "references": f"{s2}.{t2}({c2})"
    })

conn.close()

with open("adventureworks_schema.json", "w") as f:
    json.dump(schema, f, indent=2)
