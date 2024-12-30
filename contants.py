import os
from chromadb.config import Settings
 
#defining chromadb settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl = "duckdb+parquet",
    persist_directory = "db",
    anonymized_telemetry=False
)