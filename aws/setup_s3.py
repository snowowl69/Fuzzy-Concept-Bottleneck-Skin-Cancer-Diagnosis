"""
AWS S3 Bucket Setup Script
===========================
Creates the required folder structure, encryption, lifecycle policies,
CORS configuration, and versioning for the skin-cancer-diagnosis-bucket.

Folder structure:
    uploads/          - Raw dermoscopic images uploaded by users
    masks/            - Segmented lesion masks (U-Net output)
    counterfactuals/  - Generated counterfactual images
    reports/          - Diagnostic JSON reports
    models/           - Trained model weights (.pth, .pt)
    logs/             - Application log exports
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_s3_client():
    """Create an S3 client using environment credentials."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def ensure_bucket_exists(s3_client, bucket_name, region):
    """Verify the S3 bucket exists; create it if missing."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"[✓] S3 Bucket '{bucket_name}' already exists.")
        return True
    except ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            print(f"[+] Creating S3 Bucket '{bucket_name}'...")
            try:
                if region == "us-east-1":
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": region},
                    )
                print(f"[✓] Created S3 Bucket: {bucket_name}")
                return True
            except ClientError as create_err:
                print(f"[!] Failed to create bucket: {create_err}")
                return False
        else:
            print(f"[!] Cannot access bucket: {e}")
            return False


def create_folder_structure(s3_client, bucket_name):
    """Create the required folder prefixes in S3 using empty marker objects."""
    folders = [
        "uploads/",
        "masks/",
        "counterfactuals/",
        "reports/",
        "models/",
        "logs/",
    ]

    print("\n[+] Creating S3 folder structure...")
    for folder in folders:
        try:
            # Check if any objects already exist with this prefix
            response = s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=folder, MaxKeys=1
            )
            if response.get("KeyCount", 0) > 0:
                print(f"    [✓] Folder '{folder}' already exists.")
            else:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=folder,
                    Body=b"",
                    ServerSideEncryption="AES256",
                )
                print(f"    [✓] Created folder '{folder}'")
        except ClientError as e:
            print(f"    [!] Error creating folder '{folder}': {e}")


def configure_encryption(s3_client, bucket_name):
    """Enable default AES-256 server-side encryption on the bucket."""
    print("\n[+] Configuring server-side encryption (AES-256)...")
    try:
        s3_client.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        },
                        "BucketKeyEnabled": True,
                    }
                ]
            },
        )
        print("[✓] Server-side encryption enabled.")
    except ClientError as e:
        print(f"[!] Encryption configuration failed: {e}")


def configure_versioning(s3_client, bucket_name):
    """Enable bucket versioning for data protection."""
    print("\n[+] Enabling bucket versioning...")
    try:
        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={"Status": "Enabled"},
        )
        print("[✓] Bucket versioning enabled.")
    except ClientError as e:
        print(f"[!] Versioning configuration failed: {e}")


def configure_lifecycle(s3_client, bucket_name):
    """Set lifecycle rules: transition to Glacier after 90 days, delete after 365."""
    print("\n[+] Configuring lifecycle rules...")
    try:
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration={
                "Rules": [
                    {
                        "ID": "ArchiveOldUploads",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "uploads/"},
                        "Transitions": [
                            {"Days": 90, "StorageClass": "GLACIER"}
                        ],
                        "Expiration": {"Days": 365},
                    },
                    {
                        "ID": "ArchiveOldMasks",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "masks/"},
                        "Transitions": [
                            {"Days": 90, "StorageClass": "GLACIER"}
                        ],
                        "Expiration": {"Days": 365},
                    },
                    {
                        "ID": "ArchiveOldCounterfactuals",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "counterfactuals/"},
                        "Transitions": [
                            {"Days": 90, "StorageClass": "GLACIER"}
                        ],
                        "Expiration": {"Days": 365},
                    },
                    {
                        "ID": "CleanupOldLogs",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "logs/"},
                        "Expiration": {"Days": 90},
                    },
                    {
                        "ID": "CleanupOldReports",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "reports/"},
                        "Transitions": [
                            {"Days": 90, "StorageClass": "GLACIER"}
                        ],
                        "Expiration": {"Days": 365},
                    },
                ]
            },
        )
        print("[✓] Lifecycle rules configured:")
        print("    - uploads/, masks/, counterfactuals/, reports/ → Glacier at 90d, delete at 365d")
        print("    - logs/ → delete at 90d")
    except ClientError as e:
        print(f"[!] Lifecycle configuration failed: {e}")


def configure_cors(s3_client, bucket_name):
    """Set CORS rules to allow the web application to access S3."""
    print("\n[+] Configuring CORS for web application access...")
    try:
        s3_client.put_bucket_cors(
            Bucket=bucket_name,
            CORSConfiguration={
                "CORSRules": [
                    {
                        "AllowedHeaders": ["*"],
                        "AllowedMethods": ["GET", "PUT", "POST"],
                        "AllowedOrigins": ["*"],
                        "ExposeHeaders": ["ETag"],
                        "MaxAgeSeconds": 3600,
                    }
                ]
            },
        )
        print("[✓] CORS configuration applied.")
    except ClientError as e:
        print(f"[!] CORS configuration failed: {e}")


def configure_public_access_block(s3_client, bucket_name):
    """Ensure public access is fully blocked."""
    print("\n[+] Ensuring public access is blocked...")
    try:
        s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        print("[✓] Public access fully blocked.")
    except ClientError as e:
        print(f"[!] Public access block failed: {e}")


def upload_model_weights(s3_client, bucket_name):
    """Upload the existing trained model weights to S3 models/ prefix."""
    model_path = PROJECT_ROOT / "models" / "concept_bottleneck.pth"
    if not model_path.exists():
        print("\n[~] Model file 'concept_bottleneck.pth' not found locally. Skipping upload.")
        return

    print(f"\n[+] Uploading model weights ({model_path.stat().st_size} bytes)...")
    try:
        s3_client.upload_file(
            str(model_path),
            bucket_name,
            "models/concept_bottleneck.pth",
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        print("[✓] Uploaded 'models/concept_bottleneck.pth' to S3.")
    except ClientError as e:
        print(f"[!] Model upload failed: {e}")


def setup_s3():
    """Main entry point: set up the entire S3 bucket configuration."""
    print("=" * 60)
    print("AWS S3 BUCKET SETUP")
    print("=" * 60)

    bucket_name = os.getenv("S3_BUCKET_NAME", "skin-cancer-diagnosis-bucket")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("[!] AWS credentials not found in .env file. Aborting.")
        return False

    s3_client = get_s3_client()

    if not ensure_bucket_exists(s3_client, bucket_name, region):
        return False

    configure_public_access_block(s3_client, bucket_name)
    configure_encryption(s3_client, bucket_name)
    configure_versioning(s3_client, bucket_name)
    create_folder_structure(s3_client, bucket_name)
    configure_lifecycle(s3_client, bucket_name)
    configure_cors(s3_client, bucket_name)
    upload_model_weights(s3_client, bucket_name)

    print("\n" + "=" * 60)
    print("S3 SETUP COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = setup_s3()
    sys.exit(0 if success else 1)
