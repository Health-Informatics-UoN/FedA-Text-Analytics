import sys
import traceback
import psycopg
import csv
from psycopg import sql
from psycopg.rows import dict_row
from pathlib import Path
from typing import Any
import pandas as pd
from datetime import datetime

def load_vocab_csvs(conn: psycopg.Connection, schema_name: str, vocab_dir: Path):
    """
    Load all OMOP vocab CSVs into the target schema using client-side COPY with psycopg3.
    
    Args:
        conn: psycopg3 connection object
        schema_name: target schema
        vocab_dir: path to directory containing vocab CSVs
    """
    # Map CSV filenames to target table names
    vocab_files = {
        "DRUG_STRENGTH.csv": "drug_strength",
        "CONCEPT.csv": "concept",
        "CONCEPT_RELATIONSHIP.csv": "concept_relationship",
        "CONCEPT_ANCESTOR.csv": "concept_ancestor",
        "CONCEPT_SYNONYM.csv": "concept_synonym",
        "VOCABULARY.csv": "vocabulary",
        "RELATIONSHIP.csv": "relationship",
        "CONCEPT_CLASS.csv": "concept_class",
        "DOMAIN.csv": "domain"
    }

    with conn.cursor() as cur:
        # Ensure we are in the correct schema
        cur.execute(sql.SQL("SET search_path TO {};").format(sql.Identifier(schema_name)))
        conn.commit()

        for csv_file, table_name in vocab_files.items():
            file_path = vocab_dir / csv_file
            if not file_path.exists():
                print(f"Skipping {csv_file} (file not found).")
                continue

            print(f"Loading {csv_file} into {schema_name}.{table_name}...")

            with open(file_path, "r", encoding="utf-8") as f:
                # Read header row
                headers = f.readline().strip().split("\t")  # OMOP CSVs are tab-delimited

                # Build COPY statement
                copy_stmt = sql.SQL("""
                    COPY {} ({}) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\t', HEADER FALSE, QUOTE E'\b')
                """).format(
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(map(sql.Identifier, headers))
                )

                # Use psycopg3 copy context
                with cur.copy(copy_stmt) as copy:
                    for line in f:
                        copy.write(line)

            conn.commit()
            print(f"Loaded {csv_file}")

def create_person_stubs(mimic_version, csv_path, conn, chunksize=10000):
    """
    Generate minimal PERSON table entries for every unique subject_id in the notes CSV.
    """

    print("Generating PERSON stubs...")
    subject_id_col = 'subject_id' if mimic_version == 4 else 'SUBJECT_ID'

    # Connect to DB
    with conn.cursor(row_factory=dict_row) as cur:
        
        seen_person_ids = set()
        for chunk in pd.read_csv(csv_path, usecols=[subject_id_col], chunksize=chunksize):
            # Extract unique subject_ids in this chunk
            unique_ids = set(chunk[subject_id_col].dropna().astype(int).unique())
            
            # Skip IDs we've already inserted
            new_ids = unique_ids - seen_person_ids
            seen_person_ids.update(new_ids)
            
            if not new_ids:
                continue
            
            # Insert PERSON stubs
            for pid in new_ids:
                cur.execute("""
                    INSERT INTO person (
                        person_id,
                        gender_concept_id,
                        year_of_birth,
                        month_of_birth,
                        day_of_birth,
                        race_concept_id,
                        ethnicity_concept_id,
                        location_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (pid, 0, 1900, 1, 1, 0, 0, None))
        
        print(f"PERSON stubs created for {len(seen_person_ids)} unique subject_ids.")

def load_mimic_3_notes_to_omop(csv_path: str | Path, conn: psycopg.Connection, note_type_map=None, chunksize=15000):
    """
    Load MIMIC-III noteevents into OMOP CDM v5.4 NOTE table using psycopg.

    Parameters
    ----------
    csv_path : str
        Path to MIMIC-III noteevents CSV file.
    conn : psycopg.Connection
        Active database connection.
    note_type_map : dict, optional
        Mapping from MIMIC note_type to OMOP note_type_concept_id.
    chunksize : int
        Number of rows per batch insert.
    """
    # create dummy PERSON records for all subject_ids in the CSV
    create_person_stubs(3, csv_path, conn, chunksize=chunksize)

    print(f"Starting import from {csv_path}")
    print(f"Using chunks of {chunksize} rows...")

    total_inserted = 0
    batch_number = 0
    note_id = 1  # start note_id from 1. DB does not auto-increment it.
    
    # Connect once; we'll reuse it for all batches
    with conn.cursor() as cur:
        try:
            for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                batch_number += 1
                print(f"\nProcessing batch {batch_number} (rows {total_inserted + 1}–{total_inserted + len(chunk)})")

                # --- Transform & map fields ---
                chunk['person_id'] = chunk['SUBJECT_ID']
                chunk['visit_occurrence_id'] = chunk['HADM_ID']

                if note_type_map is None:
                    # User didn't provide a map — default to 0 quietly
                    chunk['note_type_concept_id'] = 0
                elif isinstance(note_type_map, dict):
                    # Proper mapping dict provided
                    chunk['note_type_concept_id'] = (
                        chunk['note_type'].map(note_type_map).fillna(0).astype(int)
                    )
                else:
                    # Weird type provided — warn user
                    print(f"note_type_map should be a dict or None, got {type(note_type_map)}. Defaulting to 0s.")
                    chunk['note_type_concept_id'] = 0
                    

                # Prepare rows for insertion
                rows = []
                
                for _, row in chunk.iterrows():

                    storetime = pd.to_datetime("1970-01-01", errors='coerce')  # MIMIC-III does not have storetime or charttime
                    charttime = pd.to_datetime("1970-01-01", errors='coerce')

                    if pd.notnull(storetime):
                        note_date = storetime.date()
                    elif pd.notnull(charttime):
                        note_date = charttime.date()
                    else:
                        note_date = datetime.today().date()  # fallback, ensures NOT NULL

                    rows.append((
                        note_id, # leave note_id to auto-generate since there may be duplicates
                        int(row['person_id']),
                        0,  # note_event_field_concept_id
                        note_date,
                        charttime,
                        int(row['note_type_concept_id']),
                        0,  # note_class_concept_id
                        None,
                        row['TEXT'],
                        0,  # encoding_concept_id
                        4180186,  # English
                        None,  # provider_id
                        None,  # visit_occurrence_id
                        None,  # visit_detail_id
                        "0"
                    ))
                    note_id += 1

                # --- Build the insert SQL ---
                insert_query = sql.SQL("""
                    INSERT INTO note (
                        note_id, person_id, note_event_field_concept_id, note_date, note_datetime,
                        note_type_concept_id, note_class_concept_id, note_title, note_text,
                        encoding_concept_id, language_concept_id, provider_id,
                        visit_occurrence_id, visit_detail_id, note_source_value
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """)

                # --- Insert within transaction ---
                try:
                    cur.executemany(insert_query, rows)
                    conn.commit()
                    total_inserted += len(rows)
                    print(f"Inserted batch {batch_number} ({len(rows)} rows)")
                except Exception as e:
                    conn.rollback()
                    print(f"ERROR in batch {batch_number}: {e}")
                    print(traceback.format_exc())
                    print("Rolled back this batch. Stopping import.")
                    return  # stop processing on first error

            print(f"\nImport complete. Total rows inserted: {total_inserted}")

        except Exception as e:
            print(f"Fatal error reading CSV or processing data: {e}")
            print(traceback.format_exc())
            conn.rollback()

def load_mimic_4_notes_to_omop(csv_path: str | Path, conn: psycopg.Connection, note_type_map=None, chunksize=15000):
    """
    Load MIMIC-IV discharge notes into OMOP CDM v5.4 NOTE table using psycopg.

    Parameters
    ----------
    csv_path : str
        Path to MIMIC-IV discharge summary CSV file.
    conn : psycopg.Connection
        Active database connection.
    note_type_map : dict, optional
        Mapping from MIMIC note_type to OMOP note_type_concept_id.
    chunksize : int
        Number of rows per batch insert.
    """

    # create dummy PERSON records for all subject_ids in the CSV
    create_person_stubs(4, csv_path, conn, chunksize=chunksize)

    print(f"Starting import from {csv_path}")
    print(f"Using chunks of {chunksize} rows...")

    total_inserted = 0
    batch_number = 0
    note_id = 1  # start note_id from 1. DB does not auto-increment it.
    
    # Connect once; we'll reuse it for all batches
    with conn.cursor() as cur:
        try:
            for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                batch_number += 1
                print(f"\nProcessing batch {batch_number} (rows {total_inserted + 1}–{total_inserted + len(chunk)})")

                # --- Transform & map fields ---
                chunk['person_id'] = chunk['subject_id']
                chunk['visit_occurrence_id'] = chunk['hadm_id']

                if note_type_map is None:
                    # User didn't provide a map — default to 0 quietly
                    chunk['note_type_concept_id'] = 0
                elif isinstance(note_type_map, dict):
                    # Proper mapping dict provided
                    chunk['note_type_concept_id'] = (
                        chunk['note_type'].map(note_type_map).fillna(0).astype(int)
                    )
                else:
                    # Weird type provided — warn user
                    print(f"note_type_map should be a dict or None, got {type(note_type_map)}. Defaulting to 0s.")
                    chunk['note_type_concept_id'] = 0
                    

                # Prepare rows for insertion
                rows = []
                
                for _, row in chunk.iterrows():

                    storetime = pd.to_datetime(row['storetime'], errors='coerce')
                    charttime = pd.to_datetime(row['charttime'], errors='coerce')

                    if pd.notnull(storetime):
                        note_date = storetime.date()
                    elif pd.notnull(charttime):
                        note_date = charttime.date()
                    else:
                        note_date = datetime.today().date()  # fallback, ensures NOT NULL

                    rows.append((
                        note_id, # leave note_id to auto-generate since there may be duplicates
                        int(row['person_id']),
                        0,  # note_event_field_concept_id
                        note_date,
                        charttime,
                        int(row['note_type_concept_id']),
                        0,  # note_class_concept_id
                        row['note_type'],
                        row['text'],
                        0,  # encoding_concept_id
                        4180186,  # English
                        None,  # provider_id
                        None,  # visit_occurrence_id
                        None,  # visit_detail_id
                        str(row['note_seq'])
                    ))
                    note_id += 1

                # --- Build the insert SQL ---
                insert_query = sql.SQL("""
                    INSERT INTO note (
                        note_id, person_id, note_event_field_concept_id, note_date, note_datetime,
                        note_type_concept_id, note_class_concept_id, note_title, note_text,
                        encoding_concept_id, language_concept_id, provider_id,
                        visit_occurrence_id, visit_detail_id, note_source_value
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """)

                # --- Insert within transaction ---
                try:
                    cur.executemany(insert_query, rows)
                    conn.commit()
                    total_inserted += len(rows)
                    print(f"Inserted batch {batch_number} ({len(rows)} rows)")
                except Exception as e:
                    conn.rollback()
                    print(f"ERROR in batch {batch_number}: {e}")
                    print(traceback.format_exc())
                    print("Rolled back this batch. Stopping import.")
                    return  # stop processing on first error

            print(f"\nImport complete. Total rows inserted: {total_inserted}")

        except Exception as e:
            print(f"Fatal error reading CSV or processing data: {e}")
            print(traceback.format_exc())
            conn.rollback()



def setup_omop_database(
    mimic_version: int,
    db_config: dict[str, Any],
    schema_sql_dir: Path,
    schema_name: str = "omop_cdm"
):
    """
    Create OMOP CDM schema, load data, and apply constraints (Postgres + psycopg3).

    Args:
        db_config (dict): { 'host', 'dbname', 'user', 'password', 'port' }
        schema_sql_dir (str): Directory with OMOP CDM v5.4 SQL files.
        populate_function (callable): Function to load data after schema creation.
        schema_name (str): Name of the target schema.
    """

    conn_str = (
        f"host={db_config['host']} "
        f"dbname={db_config['dbname']} "
        f"user={db_config['user']} "
        f"password={db_config['password']} "
        f"port={db_config['port']}"
    )

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            try:
                print(f"Creating schema '{schema_name}' if not exists...")
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(schema_name)))
                conn.commit()

                # Create base tables
                base_sql_path = schema_sql_dir / "OMOPCDM_postgresql_5.4_ddl.sql"
                if not base_sql_path.exists():
                    raise FileNotFoundError(f"Base SQL file not found: {base_sql_path}")

                print("Running base OMOP CDM schema creation script...")
                with open(base_sql_path, "r", encoding="utf-8") as f:
                    base_sql: str = f.read()

                # Replace schema name placeholder
                base_sql = base_sql.replace("@cdmDatabaseSchema", schema_name)
                cur.execute(base_sql)
                conn.commit()

                # populate vocabs
                print("Running vocabulary loading script...")
                load_vocab_csvs(conn, schema_name, Path(__file__).parent.parent / "OMOP_Vocabs")
                

                # Populate MIMIC discharge note data
                print("Populating MIMIC data...")
                if mimic_version == 3:
                    load_mimic_3_notes_to_omop(Path(__file__).parent.parent / "mimic_3_data" / "NOTEEVENTS.csv", conn)
                else:
                    load_mimic_4_notes_to_omop(Path(__file__).parent.parent / "mimic_4_data" / "discharge.csv", conn)
                conn.commit()

                # Apply keys, indices and constraints IN THIS ORDER
                ddl_files = [
                    "OMOPCDM_postgresql_5.4_primary_keys.sql",
                    "OMOPCDM_postgresql_5.4_indices.sql",
                    "OMOPCDM_postgresql_5.4_constraints.sql"
                ]

                for ddl_file in ddl_files:
                    path = schema_sql_dir / ddl_file
                    if path.exists():
                        print(f"Running {ddl_file}...")
                        with open(path, "r", encoding="utf-8") as f:
                            ddl_sql = f.read().replace("@cdmDatabaseSchema", schema_name)
                        cur.execute(ddl_sql)
                        conn.commit()
                    else:
                        print(f"Skipping {ddl_file} (not found).")

                print("OMOP CDM setup completed successfully.")

            except Exception as e:
                conn.rollback()
                print(f"Error: {e}")
                raise


if __name__ == "__main__":
    db_config = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": 5432
    }
    if sys.argv[1:]:
        mimic_version = int(sys.argv[1])
    else:
        mimic_version = 4
    setup_omop_database(mimic_version, db_config=db_config, schema_sql_dir=Path(__file__).parent.parent / "OMOP_5_4_Postgres", schema_name="omop_cdm")