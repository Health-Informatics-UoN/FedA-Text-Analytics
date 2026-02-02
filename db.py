"""
DB client for OMOP NOTE -> NOTE_NLP via CogStack ModelServe
"""

# -------------------------
# db_client module
# -------------------------

import os
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from typing import List, Dict, Any, Generator

load_dotenv("config.env")

# DB config from env
TRE_DB = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)) if os.getenv("DB_PORT") else 5432,
    "options": f"-c search_path={os.getenv('DB_SCHEMA', 'omop_cdm')}"
}

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 500))

print("DB Config:", TRE_DB)

def get_db_conn():
    return psycopg.connect(**TRE_DB, row_factory=dict_row)


def fetch_notes_by_ids(conn, note_ids: List[int]) -> List[Dict[str, Any]]:
    if not note_ids:
        return []
    q = """
        SELECT note_id, person_id, note_date, note_datetime, note_type_concept_id,
               note_class_concept_id, note_text
        FROM note
        WHERE note_id = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(q, (note_ids,))
        return cur.fetchall()
    
def fetch_all_notes(conn) -> Generator[Dict[str, Any], None, None]:
    with conn.cursor(name="tes_server_cursor") as cur:
        cur.execute("SELECT note_id, person_id, note_date, note_datetime, note_type_concept_id, note_class_concept_id, note_text FROM note")
        while True:
            batch = cur.fetchmany(BATCH_SIZE)
            if not batch:
                break
            for row in batch:
                yield row


def fetch_notes_by_query(conn, query: str) -> Generator[Dict[str, Any], None, None]:
    with conn.cursor(name="tes_server_cursor") as cur:
        cur.execute(query)
        while True:
            batch = cur.fetchmany(BATCH_SIZE)
            if not batch:
                break
            for row in batch:
                yield row


def insert_note_nlp_rows(conn, rows: List[Dict[str, Any]]):
    if not rows:
        return

    insert_cols = [
        "note_id",
        "lexical_variant",
        "nlp_system",
        "nlp_date",
        "nlp_date_time",
        "term_exists",
        "term_temporal",
        "term_modifiers",
        "snippet",
        "offset",
        "note_nlp_concept_id",
        "note_nlp_source_concept_id",
    ]

    insert_q = f"INSERT INTO note_nlp ({', '.join(insert_cols)}) VALUES ({', '.join(['%s'] * len(insert_cols))})"

    with conn.cursor() as cur:
        cur.executemany(insert_q, [tuple(r.get(c) for c in insert_cols) for r in rows])
    conn.commit()