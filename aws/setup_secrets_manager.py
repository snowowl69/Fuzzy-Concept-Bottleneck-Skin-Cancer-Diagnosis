"""
AWS Secrets Manager Setup Script — Skin Cancer Diagnosis Project
Creates and manages secrets for RDS database credentials and application keys.
"""

import os
import json
import secrets
import string
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
SECRET_NAME = "skincancer/rds/credentials"

def generate_secure_password(length=16):
    """Generate a database-compliant alphanumeric password without problematic characters."""
    alphabet = string.ascii_letters + string.digits + "!#%^*"
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in pwd)
            and any(c.isupper() for c in pwd)
            and any(c.isdigit() for c in pwd)
            and any(c in "!#%^*" for c in pwd)):
            return pwd

def setup_secrets_manager():
    print(f"Connecting to AWS Secrets Manager in {REGION}...")
    client = boto3.client("secretsmanager", region_name=REGION)

    db_password = generate_secure_password()
    jwt_secret = secrets.token_hex(32)

    secret_payload = {
        "username": "skincanceradmin",
        "password": db_password,
        "engine": "postgres",
        "port": 5432,
        "dbname": "skincancerdb",
        "host": "skincancer-db.placeholder.internal",
        "jwt_secret": jwt_secret
    }

    try:
        response = client.describe_secret(SecretId=SECRET_NAME)
        print(f"[✓] Secret '{SECRET_NAME}' already exists (ARN: {response['ARN']}).")
        val_response = client.get_secret_value(SecretId=SECRET_NAME)
        curr_val = json.loads(val_response["SecretString"])
        print(f"[✓] Existing secret retrieved. DB Username: {curr_val.get('username')}")
        return response["ARN"], curr_val

    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            print(f"Creating new secret '{SECRET_NAME}'...")
            create_resp = client.create_secret(
                Name=SECRET_NAME,
                Description="RDS PostgreSQL and App credentials for Fuzzy Concept Bottleneck Diagnosis",
                SecretString=json.dumps(secret_payload),
                Tags=[
                    {"Key": "Project", "Value": "FuzzyConceptBottleneck"},
                    {"Key": "ManagedBy", "Value": "Boto3Setup"}
                ]
            )
            print(f"[✓] Successfully created secret '{SECRET_NAME}'")
            print(f"    ARN: {create_resp['ARN']}")
            return create_resp["ARN"], secret_payload
        else:
            print(f"[!] Error accessing Secrets Manager: {e}")
            raise

if __name__ == "__main__":
    setup_secrets_manager()
