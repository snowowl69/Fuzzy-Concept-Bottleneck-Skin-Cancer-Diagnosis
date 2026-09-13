"""
AWS Master Setup Script — Complete 8-Service Architecture
=========================================================
Orchestrates the complete AWS infrastructure setup for the
Fuzzy Concept Bottleneck Skin Cancer Diagnosis system.

Services:
    1. IAM Roles & Instance Profile (SkinCancerEC2Role, etc.)
    2. AWS Secrets Manager (skincancer/rds/credentials)
    3. Amazon RDS (skincancer-db PostgreSQL)
    4. Amazon S3 (skin-cancer-diagnosis-bucket structure & models)
    5. Amazon EC2 (SkinCancer-AIPipeline-Server)
    6. Amazon Cognito (SkinCancerClinicians User Pool & Client)
    7. Amazon API Gateway (SkinCancerDiagnosisAPI REST API)
    8. Amazon CloudWatch (Log groups, alarms, dashboard in us-east-1 and ap-south-1)

Usage:
    python aws/setup_all_aws.py
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Import individual setup modules
sys.path.insert(0, str(PROJECT_ROOT))
from aws.setup_iam_roles import setup_iam_roles
from aws.setup_secrets_manager import setup_secrets_manager
from aws.setup_rds import setup_rds
from aws.setup_s3 import setup_s3
from aws.setup_ec2 import setup_ec2
from aws.setup_cognito import setup_cognito
from aws.setup_apigateway import setup_apigateway
from aws.setup_cloudwatch import setup_cloudwatch


def print_banner():
    """Print the setup banner."""
    print()
    print("+" + "=" * 66 + "+")
    print("|  FUZZY CONCEPT BOTTLENECK -- FULL AWS ARCHITECTURE SETUP        |")
    print("|  Skin Cancer Diagnosis Cloud Deployment (All 8 Core Services)    |")
    print("+" + "=" * 66 + "+")
    print()


def verify_credentials():
    """Verify AWS credentials are available before starting."""
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    bucket = os.getenv("S3_BUCKET_NAME", "skin-cancer-diagnosis-bucket")

    if not aws_key or not aws_secret:
        print("[!] ERROR: AWS credentials not found.")
        print("    Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        return False

    print(f"[✓] AWS Access Key: {aws_key[:8]}...{aws_key[-4:]}")
    print(f"[✓] Region: {region}")
    print(f"[✓] S3 Bucket: {bucket}")
    print()
    return True


def main():
    """Run all AWS setup steps in order."""
    print_banner()

    if not verify_credentials():
        return 1

    results = {}
    steps = [
        ("1. IAM Roles & Instance Profiles", setup_iam_roles),
        ("2. AWS Secrets Manager Credentials", setup_secrets_manager),
        ("3. Amazon RDS PostgreSQL Database", setup_rds),
        ("4. Amazon S3 Storage & Model Weights", setup_s3),
        ("5. Amazon EC2 AI Pipeline Server", setup_ec2),
        ("6. Amazon Cognito User Authentication", setup_cognito),
        ("7. Amazon API Gateway REST API", setup_apigateway),
        ("8. Amazon CloudWatch Monitoring", setup_cloudwatch),
    ]

    for step_name, step_func in steps:
        print(f"\n{'-' * 68}")
        print(f"  STEP: {step_name}")
        print(f"{'-' * 68}\n")
        try:
            res = step_func()
            results[step_name] = "[OK] Success" if res is not False else "[FAIL] Failed"
        except Exception as e:
            print(f"\n[!] Error in {step_name}: {e}")
            results[step_name] = f"[FAIL] Error: {e}"
        time.sleep(1)

    # Summary Report
    print("\n")
    print("+" + "=" * 66 + "+")
    print("|                    SETUP SUMMARY REPORT                          |")
    print("+" + "=" * 66 + "+")
    for step_name, result in results.items():
        line = f"  {step_name}: {result}"
        print(f"| {line:<65}|")
    print("+" + "=" * 66 + "+")

    all_passed = all("[OK]" in r for r in results.values())
    print(f"\nResult: {'ALL SERVICES PROVISIONED SUCCESSFULLY' if all_passed else 'SOME SERVICES FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
