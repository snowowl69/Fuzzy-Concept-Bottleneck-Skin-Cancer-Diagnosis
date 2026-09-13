"""
Amazon Cognito Setup Script — Skin Cancer Diagnosis Project
Creates User Pool, App Client, and Groups for Clinicians & Administrators.
"""

import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
USER_POOL_NAME = "SkinCancerClinicians"
CLIENT_NAME = "SkinCancerWebClient"

def setup_cognito():
    print(f"Connecting to Amazon Cognito in {REGION}...")
    cognito = boto3.client("cognito-idp", region_name=REGION)

    # 1. Check if User Pool already exists
    user_pools = cognito.list_user_pools(MaxResults=60)["UserPools"]
    pool_id = None
    for p in user_pools:
        if p["Name"] == USER_POOL_NAME:
            pool_id = p["Id"]
            print(f"[✓] User Pool '{USER_POOL_NAME}' already exists (ID: {pool_id}).")
            break

    if not pool_id:
        print(f"Creating User Pool '{USER_POOL_NAME}'...")
        resp = cognito.create_user_pool(
            PoolName=USER_POOL_NAME,
            Policies={
                "PasswordPolicy": {
                    "MinimumLength": 8,
                    "RequireUppercase": True,
                    "RequireLowercase": True,
                    "RequireNumbers": True,
                    "RequireSymbols": True
                }
            },
            AutoVerifiedAttributes=["email"],
            UsernameAttributes=["email"],
            UserPoolTags={
                "Project": "FuzzyConceptBottleneck",
                "ManagedBy": "Boto3Setup"
            }
        )
        pool_id = resp["UserPool"]["Id"]
        print(f"[✓] Successfully created User Pool: {pool_id}")

    # 2. Check or create User Pool Client
    clients = cognito.list_user_pool_clients(UserPoolId=pool_id, MaxResults=60)["UserPoolClients"]
    client_id = None
    for c in clients:
        if c["ClientName"] == CLIENT_NAME:
            client_id = c["ClientId"]
            print(f"[✓] App Client '{CLIENT_NAME}' already exists (ID: {client_id}).")
            break

    if not client_id:
        print(f"Creating App Client '{CLIENT_NAME}'...")
        client_resp = cognito.create_user_pool_client(
            UserPoolId=pool_id,
            ClientName=CLIENT_NAME,
            GenerateSecret=False,
            ExplicitAuthFlows=[
                "ALLOW_USER_SRP_AUTH",
                "ALLOW_USER_PASSWORD_AUTH",
                "ALLOW_REFRESH_TOKEN_AUTH"
            ],
            PreventUserExistenceErrors="ENABLED"
        )
        client_id = client_resp["UserPoolClient"]["ClientId"]
        print(f"[✓] Successfully created App Client: {client_id}")

    # 3. Create User Groups
    groups = [
        {"name": "Clinicians", "desc": "Dermatologists and diagnostic clinicians"},
        {"name": "Administrators", "desc": "System and clinical administrators"}
    ]
    for g in groups:
        try:
            cognito.get_group(GroupName=g["name"], UserPoolId=pool_id)
            print(f"[✓] Group '{g['name']}' exists.")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                cognito.create_group(
                    GroupName=g["name"],
                    UserPoolId=pool_id,
                    Description=g["desc"]
                )
                print(f"[✓] Created Group '{g['name']}'.")
            else:
                raise

    print("\n" + "="*50)
    print("           COGNITO SETUP SUMMARY")
    print("="*50)
    print(f"  User Pool Name: {USER_POOL_NAME}")
    print(f"  User Pool ID:   {pool_id}")
    print(f"  App Client ID:  {client_id}")
    print("="*50)

    return pool_id, client_id

if __name__ == "__main__":
    setup_cognito()
