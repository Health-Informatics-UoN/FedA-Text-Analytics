import os
import json
import csv
import argparse
import requests
from dotenv import load_dotenv
from db import get_db_conn, fetch_notes_by_query, fetch_all_notes

load_dotenv("config.env")

MODEL_SERVE_URL = "http://" + os.getenv("MODEL_SERVE_URL", "127.0.0.1:8000")
MODEL_SERVE_TIMEOUT = int(os.getenv("MODEL_SERVE_TIMEOUT", 60))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_model(
    model: str,
    query: str | None,
    batch_size: int,
    output_csv: str,
):
    model_url = f"{MODEL_SERVE_URL.rstrip('/')}/process_bulk"

    def modelserve_custom(texts):
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        response = requests.post(
            model_url,
            headers=headers,
            json=texts,
            timeout=MODEL_SERVE_TIMEOUT
        )
        response.raise_for_status()
        return response.json()

    results = []
    processed = 0

    with get_db_conn() as conn:
        notes = fetch_notes_by_query(conn, query) if query else fetch_all_notes(conn)
        notes = list(notes)
        total = len(notes)
        print(f"Running model '{model}' on {total} notes...")

        for batch in batched(notes, batch_size):
            texts = [n.get("note_text", "") or "" for n in batch]
            note_ids = [n.get("note_id") for n in batch]

            try:
                outputs = modelserve_custom(texts)
                for note_id, output in zip(note_ids, outputs):
                    results.append({
                        "note_id": note_id,
                        "modelserve_output": json.dumps(output),
                        "error": ""
                    })
            except Exception as e:
                for note_id in note_ids:
                    results.append({
                        "note_id": note_id,
                        "modelserve_output": "",
                        "error": str(e)
                    })

            processed += len(batch)
            print(f"Processed {processed}/{total} notes")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["note_id", "modelserve_output", "error"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Completed successfully. Results written to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run NLP model over OMOP notes and write results to CSV"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--query")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    run_model(
        model=args.model,
        query=args.query,
        batch_size=args.batch_size,
        output_csv=args.output
    )


if __name__ == "__main__":
    main()
