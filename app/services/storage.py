import os
import tempfile
import uuid

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
    return _s3_client


def is_s3_configured() -> bool:
    return bool(settings.s3_bucket_name and settings.aws_access_key_id and settings.aws_secret_access_key)


def new_s3_key(filename: str | None) -> str:
    ext = os.path.splitext(filename or "")[1] or ".pdf"
    return f"documents/{uuid.uuid4()}{ext}"


def upload_bytes_to_s3(data: bytes, key: str) -> str:
    """Upload raw document bytes to S3 and return the object key."""
    client = _get_s3_client()
    client.put_object(Bucket=settings.s3_bucket_name, Key=key, Body=data)
    return key


def download_document(source_url: str | None, s3_key: str | None, suffix: str) -> str:
    """Materialize a document to a local temp file, preferring S3 (when configured
    and a key is on hand) and falling back to a direct URL download otherwise.
    Returns the local path; the caller is responsible for cleaning it up."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)

    if s3_key and is_s3_configured():
        client = _get_s3_client()
        try:
            client.download_file(settings.s3_bucket_name, s3_key, tmp_path)
            return tmp_path
        except (BotoCoreError, ClientError) as exc:
            os.remove(tmp_path)
            raise RuntimeError(f"Failed to download {s3_key} from S3: {exc}") from exc

    if not source_url:
        os.remove(tmp_path)
        raise RuntimeError("No source_url or S3 key available for document download")

    response = requests.get(source_url, timeout=30)
    if response.status_code != 200:
        os.remove(tmp_path)
        raise RuntimeError(f"Failed to download document. Status: {response.status_code}")

    with open(tmp_path, "wb") as f:
        f.write(response.content)
    return tmp_path
