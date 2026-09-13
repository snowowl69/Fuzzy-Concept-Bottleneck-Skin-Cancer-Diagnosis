"""
AWS IAM Roles Setup Script
============================
Creates the required IAM roles, policies, and instance profiles
for the Fuzzy Concept Bottleneck Skin Cancer Diagnosis system.

Roles:
    SkinCancerEC2Role           - EC2 instance role with S3, CloudWatch, Secrets Manager access
    SkinCancerAPIGatewayRole    - API Gateway role with CloudWatch Logs write access
    SkinCancerCloudWatchRole    - CloudWatch role with SNS publish for alarm actions

Instance Profile:
    SkinCancerEC2InstanceProfile - Attaches SkinCancerEC2Role to EC2 instances
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_iam_client():
    """Create an IAM client using environment credentials."""
    return boto3.client(
        "iam",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def get_account_id():
    """Get the current AWS account ID."""
    sts = boto3.client(
        "sts",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    return sts.get_caller_identity()["Account"]


def create_role(iam_client, role_name, trust_policy, description, tags):
    """Create an IAM role if it doesn't already exist."""
    try:
        response = iam_client.get_role(RoleName=role_name)
        print(f"    [✓] Role '{role_name}' already exists.")
        return response["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            try:
                response = iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description=description,
                    Tags=tags,
                )
                print(f"    [✓] Created role '{role_name}'")
                return response["Role"]["Arn"]
            except ClientError as create_err:
                print(f"    [!] Failed to create role '{role_name}': {create_err}")
                return None
        else:
            print(f"    [!] Error checking role '{role_name}': {e}")
            return None


def attach_inline_policy(iam_client, role_name, policy_name, policy_document):
    """Attach an inline policy to an IAM role."""
    try:
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document),
        )
        print(f"        [✓] Attached policy '{policy_name}' to '{role_name}'")
    except ClientError as e:
        print(f"        [!] Failed to attach policy '{policy_name}': {e}")


def create_ec2_role(iam_client, account_id, region):
    """Create the EC2 instance role with S3, CloudWatch, Secrets Manager, and RDS access."""
    print("\n[+] Setting up SkinCancerEC2Role...")

    bucket_name = os.getenv("S3_BUCKET_NAME", "skin-cancer-diagnosis-bucket")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    role_arn = create_role(
        iam_client,
        "SkinCancerEC2Role",
        trust_policy,
        "EC2 role for Skin Cancer Diagnosis app - S3, CloudWatch, Secrets Manager access",
        [
            {"Key": "Project", "Value": "FuzzyConceptBottleneck"},
            {"Key": "Environment", "Value": "Production"},
        ],
    )

    if not role_arn:
        return None

    # S3 Access Policy
    s3_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3ReadWriteAccess",
                "Effect": "Allow",
                "Action": [
                    "s3:PutObject",
                    "s3:GetObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*",
                ],
            }
        ],
    }
    attach_inline_policy(iam_client, "SkinCancerEC2Role", "S3AccessPolicy", s3_policy)

    # CloudWatch Logs Policy
    cw_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CloudWatchLogsAccess",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams",
                ],
                "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:SkinCancerApp/*",
            },
            {
                "Sid": "CloudWatchMetricsAccess",
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:PutMetricData",
                ],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {"cloudwatch:namespace": "SkinCancerApp"}
                },
            },
        ],
    }
    attach_inline_policy(iam_client, "SkinCancerEC2Role", "CloudWatchPolicy", cw_policy)

    # Secrets Manager Policy
    sm_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "SecretsManagerRead",
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue"],
                "Resource": f"arn:aws:secretsmanager:{region}:{account_id}:secret:skincancer/*",
            }
        ],
    }
    attach_inline_policy(iam_client, "SkinCancerEC2Role", "SecretsManagerPolicy", sm_policy)

    # RDS Connect Policy
    rds_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "RDSConnect",
                "Effect": "Allow",
                "Action": [
                    "rds-db:connect",
                ],
                "Resource": f"arn:aws:rds-db:{region}:{account_id}:dbuser:*/skincancerdb",
            }
        ],
    }
    attach_inline_policy(iam_client, "SkinCancerEC2Role", "RDSConnectPolicy", rds_policy)

    return role_arn


def create_api_gateway_role(iam_client, account_id, region):
    """Create the API Gateway role with CloudWatch Logs write access."""
    print("\n[+] Setting up SkinCancerAPIGatewayRole...")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "apigateway.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    role_arn = create_role(
        iam_client,
        "SkinCancerAPIGatewayRole",
        trust_policy,
        "API Gateway role for CloudWatch access logging",
        [
            {"Key": "Project", "Value": "FuzzyConceptBottleneck"},
            {"Key": "Environment", "Value": "Production"},
        ],
    )

    if not role_arn:
        return None

    # CloudWatch Logs write policy for API Gateway access logging
    apigw_cw_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "APIGatewayCloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams",
                    "logs:PutLogEvents",
                    "logs:GetLogEvents",
                    "logs:FilterLogEvents",
                ],
                "Resource": "*",
            }
        ],
    }
    attach_inline_policy(
        iam_client, "SkinCancerAPIGatewayRole", "APIGatewayCloudWatchPolicy", apigw_cw_policy
    )

    # Also attach the AWS managed policy for API Gateway → CloudWatch
    try:
        iam_client.attach_role_policy(
            RoleName="SkinCancerAPIGatewayRole",
            PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs",
        )
        print("        [✓] Attached managed policy 'AmazonAPIGatewayPushToCloudWatchLogs'")
    except ClientError as e:
        print(f"        [~] Managed policy attachment note: {e}")

    return role_arn


def create_cloudwatch_role(iam_client, account_id, region):
    """Create the CloudWatch role with SNS publish for alarm notifications."""
    print("\n[+] Setting up SkinCancerCloudWatchRole...")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "cloudwatch.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    role_arn = create_role(
        iam_client,
        "SkinCancerCloudWatchRole",
        trust_policy,
        "CloudWatch role for SNS alarm notifications",
        [
            {"Key": "Project", "Value": "FuzzyConceptBottleneck"},
            {"Key": "Environment", "Value": "Production"},
        ],
    )

    if not role_arn:
        return None

    # SNS publish policy for alarm actions
    sns_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "SNSPublishAlarms",
                "Effect": "Allow",
                "Action": ["sns:Publish"],
                "Resource": f"arn:aws:sns:{region}:{account_id}:SkinCancerAlerts",
            }
        ],
    }
    attach_inline_policy(iam_client, "SkinCancerCloudWatchRole", "SNSPublishPolicy", sns_policy)

    return role_arn


def create_instance_profile(iam_client):
    """Create an EC2 Instance Profile and attach the EC2 role."""
    print("\n[+] Setting up EC2 Instance Profile...")

    profile_name = "SkinCancerEC2InstanceProfile"

    try:
        iam_client.get_instance_profile(InstanceProfileName=profile_name)
        print(f"    [✓] Instance profile '{profile_name}' already exists.")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            print(f"    [!] Error checking instance profile: {e}")
            return

    try:
        iam_client.create_instance_profile(
            InstanceProfileName=profile_name,
            Tags=[
                {"Key": "Project", "Value": "FuzzyConceptBottleneck"},
            ],
        )
        print(f"    [✓] Created instance profile '{profile_name}'")

        # Wait briefly for profile to propagate
        time.sleep(2)

        iam_client.add_role_to_instance_profile(
            InstanceProfileName=profile_name,
            RoleName="SkinCancerEC2Role",
        )
        print(f"    [✓] Attached 'SkinCancerEC2Role' to instance profile")
    except ClientError as e:
        if "LimitExceeded" in str(e) or "EntityAlreadyExists" in str(e):
            print(f"    [✓] Instance profile already configured.")
        else:
            print(f"    [!] Instance profile setup failed: {e}")


def setup_iam_roles():
    """Main entry point: create all required IAM roles and instance profiles."""
    print("=" * 60)
    print("AWS IAM ROLES SETUP")
    print("=" * 60)

    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("[!] AWS credentials not found in .env file. Aborting.")
        return False

    iam_client = get_iam_client()
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    try:
        account_id = get_account_id()
        print(f"[✓] AWS Account: {account_id}")
        print(f"[✓] Region: {region}")
    except Exception as e:
        print(f"[!] Failed to get account ID: {e}")
        return False

    ec2_arn = create_ec2_role(iam_client, account_id, region)
    apigw_arn = create_api_gateway_role(iam_client, account_id, region)
    cw_arn = create_cloudwatch_role(iam_client, account_id, region)
    create_instance_profile(iam_client)

    print("\n" + "-" * 60)
    print("IAM ROLES SUMMARY")
    print("-" * 60)
    print(f"  EC2 Role:         {'✓ ' + ec2_arn if ec2_arn else '✗ Failed'}")
    print(f"  API Gateway Role: {'✓ ' + apigw_arn if apigw_arn else '✗ Failed'}")
    print(f"  CloudWatch Role:  {'✓ ' + cw_arn if cw_arn else '✗ Failed'}")

    print("\n" + "=" * 60)
    print("IAM ROLES SETUP COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = setup_iam_roles()
    sys.exit(0 if success else 1)
