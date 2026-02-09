# tre_modelserve_tes_bridge.py
import os
import json
import csv
from datetime import datetime
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from db import get_db_conn, fetch_notes_by_query, fetch_all_notes
from uuid import uuid4
from datetime import datetime, timezone

load_dotenv("config.env")

MODEL_SERVE_URL = "http://" + os.getenv("MODEL_SERVE_URL", "127.0.0.1:8000")
MODEL_SERVE_TIMEOUT = int(os.getenv("MODEL_SERVE_TIMEOUT", 60))
SERVER_BIND = os.getenv("SERVER_BIND", "0.0.0.0:8080")
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))

app = Flask(__name__)

# -------------------------------
# Utility functions
# -------------------------------

def call_modelserve(texts):
    """Send a batch of texts to CogStack ModelServe and return parsed JSON."""
    if not MODEL_SERVE_URL:
        raise RuntimeError("MODEL_SERVE_URL not set.")
    resp = requests.post(MODEL_SERVE_URL, json={"texts": texts}, timeout=MODEL_SERVE_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def batched(iterable, batch_size):
    """Yield successive batches from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_omop_notes(query=None, batch_size=DEFAULT_BATCH_SIZE, output_csv=None):
    """Core OMOP -> ModelServe pipeline."""
    results = []
    processed = 0

    with get_db_conn() as conn:
        notes = fetch_notes_by_query(conn, query) if query else fetch_all_notes(conn)
        notes = list(notes)
        total_notes = len(notes)
        print(f"Processing {total_notes} notes in batches of {batch_size}...")

        for batch in batched(notes, batch_size):
            texts = [n.get("note_text", "") or "" for n in batch]
            note_ids = [n.get("note_id") for n in batch]
            try:
                batch_output = call_modelserve(texts)
                for note_id, output in zip(note_ids, batch_output):
                    results.append({
                        "note_id": note_id,
                        "modelserve_output": json.dumps(output)
                    })
            except Exception as e:
                for note_id in note_ids:
                    results.append({
                        "note_id": note_id,
                        "error": str(e)
                    })
            processed += len(batch)
            print(f"Processed {processed}/{total_notes} notes")

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["note_id", "modelserve_output", "error"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    return {"processed": processed, "results": results, "output_csv": output_csv}

# -------------------------------
# REST endpoints
# -------------------------------

@app.route("/process", methods=["POST"])
def process_notes():
    """
    Local endpoint for manual use.
    {
        "query": "...",
        "batch_size": 100,
        "output_csv": "results.csv"
    }
    """
    body = request.get_json(force=True)
    query = body.get("query")
    batch_size = int(body.get("batch_size", DEFAULT_BATCH_SIZE))
    output_csv = body.get("output_csv")

    result = process_omop_notes(query, batch_size, output_csv)

    if output_csv:
        return jsonify({
            "status": "complete",
            "processed": result["processed"],
            "output_file": result["output_csv"]
        }), 200
    else:
        return jsonify(result), 200

@app.route("/run_model", methods=["POST"])
def run_model():
    """
    Run a specific NLP model on OMOP notes and save results locally.

    Example request:
    {
        "model": "medcat_snomed",
        "query": "SELECT * FROM note LIMIT 50",
        "batch_size": 50,
        "output_csv": "/mnt/results/ner_output.csv"
    }
    """
    body = request.get_json(force=True)
    model = body.get("model")
    query = body.get("query")
    batch_size = int(body.get("batch_size", DEFAULT_BATCH_SIZE))
    output_csv = body.get("output_csv")

    if not model:
        return jsonify({"error": "Missing required parameter: 'model'"}), 400
    if not output_csv:
        return jsonify({"error": "Missing required parameter: 'output_csv'"}), 400

    # Build the specific model endpoint
    model_url = f"{MODEL_SERVE_URL.rstrip('/')}/process_bulk"



    def modelserve_custom(texts):
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        response = requests.post(model_url, headers=headers, json=texts)
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
                print(f"Error processing batch: {e}")
                for note_id in note_ids:
                    results.append({
                        "note_id": note_id,
                        "error": str(e)
                    })
            processed += len(batch)
            print(f"Processed {processed}/{total} notes")

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["note_id", "modelserve_output", "error"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    return jsonify({
        "status": "complete",
        "model": model,
        "processed": processed,
        "output_csv": output_csv
    }), 200


if __name__ == "__main__":
    host, port = SERVER_BIND.split(":")
    app.run(host=host, port=int(port), threaded=False)
