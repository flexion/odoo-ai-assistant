"""
AWS Lambda function for automated data pipeline.

Triggered by S3 upload of sources.json:
1. Download sources.json from S3
2. Run ingest (scrape docs)
3. Generate QA pairs
4. Build FAISS index
5. Upload artifacts back to S3

Environment Variables:
- S3_BUCKET: Bucket name for data storage
- S3_PREFIX: Prefix for data files (default: data/)
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import boto3

s3 = boto3.client("s3")


def lambda_handler(event, context):
    """
    S3 event handler for sources.json upload.

    Event structure:
    {
        "Records": [{
            "s3": {
                "bucket": {"name": "odoo-rag-data-2026-cohort"},
                "object": {"key": "input/sources.json"}
            }
        }]
    }
    """
    # Extract S3 details from event
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    print(f"Processing S3 event: s3://{bucket}/{key}")

    # Only process sources.json uploads
    if not key.endswith("sources.json"):
        print(f"Ignoring non-sources.json file: {key}")
        return {"statusCode": 200, "body": "Skipped"}

    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        data_dir = work_dir / "data"
        data_dir.mkdir()

        # Download sources.json
        sources_path = data_dir / "sources.json"
        print(f"Downloading {key} to {sources_path}")
        s3.download_file(bucket, key, str(sources_path))

        # Verify sources.json is valid
        with open(sources_path) as f:
            sources = json.load(f)
            print(f"Found {len(sources.get('sources', []))} sources")

        # Run data pipeline
        try:
            # Step 1: Ingest documentation
            print("Step 1: Ingesting documentation...")
            subprocess.run(
                ["python", "-m", "odoo_rag.ingest"],
                cwd=work_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
            )

            # Step 2: Generate QA pairs (DISABLED - not needed for RAG approach)
            # Generates synthetic Q&A pairs for fine-tuning custom models.
            # This project uses pre-trained Bedrock models with RAG, so this step is disabled.
            # Enable this step only when fine-tuning a custom model on the Odoo domain.
            # print("Step 2: Generating QA pairs...")
            # subprocess.run(
            #     ['python', '-m', 'odoo_rag.generate_qa'],
            #     cwd=work_dir,
            #     check=True,
            #     capture_output=True,
            #     text=True,
            #     timeout=300
            # )

            # Step 3: Build FAISS index
            print("Step 2: Building FAISS index...")
            subprocess.run(
                ["python", "-m", "odoo_rag.indexer"],
                cwd=work_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=840,  # 14 min timeout (Lambda has 15 min max)
            )

            # Upload artifacts back to S3 output/ folder
            output_prefix = os.environ.get("S3_OUTPUT_PREFIX", "output/")
            artifacts = [
                ("data/faiss_index/index.faiss", f"{output_prefix}index.faiss"),
                ("data/faiss_index/corpus.json", f"{output_prefix}corpus.json"),
            ]

            # Enable this line if QA generation (Step 2 above) is enabled:
            # artifacts.append(('data/qa_pairs.jsonl', f'{output_prefix}qa_pairs.jsonl'))

            for local_path, s3_key in artifacts:
                artifact_path = work_dir / local_path
                if artifact_path.exists():
                    print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
                    s3.upload_file(str(artifact_path), bucket, s3_key)
                else:
                    print(f"Warning: {local_path} not found, skipping upload")

            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": "Pipeline completed successfully",
                        "artifacts": [s3_key for _, s3_key in artifacts],
                    }
                ),
            }

        except subprocess.TimeoutExpired as e:
            print(f"Pipeline step timed out: {e}")
            return {"statusCode": 500, "body": json.dumps({"error": f"Timeout: {e}"})}
        except subprocess.CalledProcessError as e:
            print(f"Pipeline step failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "error": f"Pipeline failed: {e}",
                        "stdout": e.stdout,
                        "stderr": e.stderr,
                    }
                ),
            }
