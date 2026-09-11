import os
import sys
import boto3
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path.home() / ".env")

def test_aws_connection():
    print("=" * 60)
    print("TESTING AWS CONNECTION & PROVISIONING")
    print("=" * 60)

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    s3_bucket = os.getenv("S3_BUCKET_NAME", "skin-cancer-diagnosis-bucket")

    if not aws_access_key or not aws_secret_key:
        print("[!] AWS credentials missing from environment / .env file.")
        print("Please configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        return False

    try:
        sts_client = boto3.client(
            "sts",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        identity = sts_client.get_caller_identity()
        print(f"[✓] AWS Connection Successful!")
        print(f"    Account: {identity.get('Account')}")
        print(f"    User ARN: {identity.get('Arn')}")
        print(f"    Region: {aws_region}")
    except Exception as e:
        print(f"[!] AWS Authentication Failed: {e}")
        return False

    # Test S3 Bucket Access / Creation
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        print(f"\n[+] Checking S3 Bucket: {s3_bucket}...")
        buckets = [b['Name'] for b in s3_client.list_buckets().get('Buckets', [])]
        if s3_bucket in buckets:
            print(f"[✓] S3 Bucket '{s3_bucket}' already exists and is accessible.")
        else:
            print(f"[+] Creating S3 Bucket '{s3_bucket}'...")
            if aws_region == "us-east-1":
                s3_client.create_bucket(Bucket=s3_bucket)
            else:
                s3_client.create_bucket(
                    Bucket=s3_bucket,
                    CreateBucketConfiguration={'LocationConstraint': aws_region}
                )
            print(f"[✓] Created S3 Bucket: {s3_bucket}")
    except Exception as e:
        print(f"[!] S3 Setup Warning: {e}")

    # Test CloudWatch Logs
    try:
        logs_client = boto3.client(
            "logs",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        log_group = "SkinCancerApp/InferenceLogs"
        print(f"\n[+] Checking CloudWatch Log Group: {log_group}...")
        groups = [g['logGroupName'] for g in logs_client.describe_log_groups().get('logGroups', [])]
        if log_group not in groups:
            logs_client.create_log_group(logGroupName=log_group)
            print(f"[✓] Created CloudWatch Log Group: {log_group}")
        else:
            print(f"[✓] CloudWatch Log Group '{log_group}' is active.")
    except Exception as e:
        print(f"[!] CloudWatch Setup Warning: {e}")

    # Test API Gateway Client
    try:
        apigw_client = boto3.client(
            "apigateway",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        print("\n[+] Checking API Gateway REST APIs...")
        apis = apigw_client.get_rest_apis().get('items', [])
        print(f"[✓] API Gateway Service active. Found {len(apis)} existing REST APIs.")
    except Exception as e:
        print(f"[!] API Gateway Check Warning: {e}")

    # Test Cognito Identity Provider
    try:
        cognito_client = boto3.client(
            "cognito-idp",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        print("\n[+] Checking Cognito User Pools...")
        pools = cognito_client.list_user_pools(MaxResults=10).get('UserPools', [])
        print(f"[✓] Cognito Service active. Found {len(pools)} User Pools.")
    except Exception as e:
        print(f"[!] Cognito Check Warning: {e}")

    print("\n" + "=" * 60)
    print("AWS CONNECTION TEST COMPLETED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_aws_connection()
