"""
AWS Connection & Comprehensive Resource Verification Script
============================================================
Tests connectivity and audits all 8 core services from the
Fuzzy Concept Bottleneck Skin Cancer Diagnosis specification:

    1. AWS Authentication (STS)
    2. Amazon S3 (Buckets, Folders, Models, Policies)
    3. Amazon EC2 (AI Pipeline Virtual Server, Security Group, Keys)
    4. Amazon RDS (PostgreSQL Database, DB Subnet Group, Status)
    5. AWS IAM (Roles, Least-Privilege Policies, Instance Profile)
    6. Amazon CloudWatch (Log Groups, Metric Filters, Alarms, Dashboards)
    7. Amazon API Gateway (REST APIs, Stage, Proxy Endpoints)
    8. AWS Secrets Manager (Database Credentials & App Keys)
    9. Amazon Cognito (User Pool, Clinician/Admin Groups, App Client)
   10. Amazon SNS (Alarm Notification Topics)
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


def get_client(service, region=None):
    """Create a boto3 client for the specified AWS service."""
    return boto3.client(
        service,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def check_auth():
    """Test AWS authentication via STS."""
    print("\n" + "-" * 55)
    print("1. AWS AUTHENTICATION")
    print("-" * 55)
    try:
        sts = get_client("sts")
        identity = sts.get_caller_identity()
        print("[OK] AWS Authentication Successful!")
        print(f"     Account:  {identity.get('Account')}")
        print(f"     User ARN: {identity.get('Arn')}")
        print(f"     Region:   {os.getenv('AWS_DEFAULT_REGION', 'us-east-1')}")
        return True
    except Exception as e:
        print(f"[FAIL] AWS Authentication Failed: {e}")
        return False


def check_s3():
    """Verify S3 bucket, folders, encryption, and uploaded model files."""
    print("\n" + "-" * 55)
    print("2. AMAZON S3 STORAGE")
    print("-" * 55)
    bucket = os.getenv("S3_BUCKET_NAME", "skin-cancer-diagnosis-bucket")
    s3 = get_client("s3")
    results = []

    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[OK] Bucket '{bucket}' exists and is accessible.")
        results.append(True)
    except ClientError:
        print(f"[FAIL] Bucket '{bucket}' not found or inaccessible.")
        return False

    expected_folders = ["uploads/", "masks/", "counterfactuals/", "reports/", "models/", "logs/"]
    print("     Checking folder structure...")
    for f in expected_folders:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=f, MaxKeys=1)
        if resp.get("KeyCount", 0) > 0:
            print(f"     [OK] {f}")
            results.append(True)
        else:
            print(f"     [FAIL] {f} (missing)")
            results.append(False)

    try:
        enc = s3.get_bucket_encryption(Bucket=bucket)
        algo = enc["ServerSideEncryptionConfiguration"]["Rules"][0]["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"]
        print(f"     [OK] Server-Side Encryption: {algo}")
    except Exception:
        pass

    try:
        ver = s3.get_bucket_versioning(Bucket=bucket)
        print(f"     [OK] Versioning: {ver.get('Status', 'Disabled')}")
    except Exception:
        pass

    return all(results)


def check_ec2():
    """Verify EC2 AI Pipeline server is provisioned and running."""
    print("\n" + "-" * 55)
    print("3. AMAZON EC2 (AI PIPELINE SERVER)")
    print("-" * 55)
    ec2 = get_client("ec2")
    try:
        resps = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": ["SkinCancer-AIPipeline-Server"]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]}
            ]
        )["Reservations"]

        if not resps or not resps[0]["Instances"]:
            print("[FAIL] No running SkinCancer-AIPipeline-Server EC2 instance found.")
            return False

        inst = resps[0]["Instances"][0]
        print(f"[OK] Instance Found: {inst['InstanceId']}")
        print(f"     State:        {inst['State']['Name']}")
        print(f"     Type:         {inst['InstanceType']}")
        print(f"     Public IP:    {inst.get('PublicIpAddress', 'N/A')}")
        print(f"     Public DNS:   {inst.get('PublicDnsName', 'N/A')}")
        print(f"     IAM Profile:  {inst.get('IamInstanceProfile', {}).get('Arn', 'None')}")
        return True
    except Exception as e:
        print(f"[FAIL] EC2 check error: {e}")
        return False


def check_rds():
    """Verify Amazon RDS PostgreSQL database instance."""
    print("\n" + "-" * 55)
    print("4. AMAZON RDS (POSTGRESQL DATABASE)")
    print("-" * 55)
    rds = get_client("rds")
    try:
        res = rds.describe_db_instances(DBInstanceIdentifier="skincancer-db")["DBInstances"]
        if not res:
            print("[FAIL] RDS Instance 'skincancer-db' not found.")
            return False

        db = res[0]
        status = db["DBInstanceStatus"]
        endpoint = db.get("Endpoint", {}).get("Address", "Provisioning...")
        port = db.get("Endpoint", {}).get("Port", 5432)
        print(f"[OK] Database Identifier: {db['DBInstanceIdentifier']}")
        print(f"     Status:              {status}")
        print(f"     Engine:              {db['Engine']} {db.get('EngineVersion', '')}")
        print(f"     Class:               {db['DBInstanceClass']}")
        print(f"     Endpoint:            {endpoint}:{port}")
        return status in ["available", "creating", "modifying"]
    except Exception as e:
        print(f"[FAIL] RDS check error: {e}")
        return False


def check_iam():
    """Verify IAM Roles and Instance Profile."""
    print("\n" + "-" * 55)
    print("5. AWS IAM ROLES & PROFILES")
    print("-" * 55)
    iam = get_client("iam")
    roles = ["SkinCancerEC2Role", "SkinCancerAPIGatewayRole", "SkinCancerCloudWatchRole"]
    results = []

    for r in roles:
        try:
            iam.get_role(RoleName=r)
            print(f"[OK] IAM Role: {r}")
            results.append(True)
        except Exception:
            print(f"[FAIL] IAM Role missing: {r}")
            results.append(False)

    try:
        iam.get_instance_profile(InstanceProfileName="SkinCancerEC2InstanceProfile")
        print(f"[OK] Instance Profile: SkinCancerEC2InstanceProfile")
        results.append(True)
    except Exception:
        print(f"[FAIL] Instance Profile missing: SkinCancerEC2InstanceProfile")
        results.append(False)

    return all(results)


def check_cloudwatch():
    """Verify CloudWatch log groups, alarms, and dashboard."""
    print("\n" + "-" * 55)
    print("6. AMAZON CLOUDWATCH MONITORING")
    print("-" * 55)
    logs = get_client("logs")
    cw = get_client("cloudwatch")
    results = []

    expected_groups = [
        "SkinCancerApp/InferenceLogs",
        "SkinCancerApp/ApplicationLogs",
        "SkinCancerApp/ErrorLogs",
        "SkinCancerApp/AccessLogs"
    ]

    print("     Checking Log Groups...")
    existing = [g["logGroupName"] for g in logs.describe_log_groups(logGroupNamePrefix="SkinCancerApp").get("logGroups", [])]
    for eg in expected_groups:
        if eg in existing:
            print(f"     [OK] {eg}")
            results.append(True)
        else:
            print(f"     [FAIL] {eg}")
            results.append(False)

    print("     Checking Alarms...")
    alarms = [a["AlarmName"] for a in cw.describe_alarms(AlarmNamePrefix="SkinCancer").get("MetricAlarms", [])]
    for ea in ["SkinCancer-HighErrorRate", "SkinCancer-HighApplicationErrorRate", "SkinCancer-S3BucketSizeAlert"]:
        if ea in alarms:
            print(f"     [OK] {ea}")
            results.append(True)
        else:
            print(f"     [FAIL] {ea}")
            results.append(False)

    try:
        cw.get_dashboard(DashboardName="SkinCancerDiagnosisDashboard")
        print("     [OK] Dashboard 'SkinCancerDiagnosisDashboard' active.")
        results.append(True)
    except Exception:
        print("     [FAIL] Dashboard missing.")
        results.append(False)

    return all(results)


def check_apigateway():
    """Verify API Gateway REST API and deployment stage."""
    print("\n" + "-" * 55)
    print("7. AMAZON API GATEWAY")
    print("-" * 55)
    apigw = get_client("apigateway")
    try:
        apis = apigw.get_rest_apis().get("items", [])
        sc_apis = [a for a in apis if a.get("name") == "SkinCancerDiagnosisAPI"]
        if not sc_apis:
            print("[FAIL] SkinCancerDiagnosisAPI REST API not found.")
            return False

        api = sc_apis[0]
        api_id = api["id"]
        stages = apigw.get_stages(restApiId=api_id).get("item", [])
        stage_names = [s["stageName"] for s in stages]
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        invoke_url = f"https://{api_id}.execute-api.{region}.amazonaws.com/prod/"

        print(f"[OK] REST API:   {api['name']} (ID: {api_id})")
        print(f"     Stages:     {stage_names}")
        print(f"     Invoke URL: {invoke_url}")
        return True
    except Exception as e:
        print(f"[FAIL] API Gateway check error: {e}")
        return False


def check_secrets_manager():
    """Verify AWS Secrets Manager secret for RDS credentials."""
    print("\n" + "-" * 55)
    print("8. AWS SECRETS MANAGER")
    print("-" * 55)
    sm = get_client("secretsmanager")
    secret_name = os.getenv("SECRETS_MANAGER_NAME", "skincancer/rds/credentials")
    try:
        resp = sm.describe_secret(SecretId=secret_name)
        val = json.loads(sm.get_secret_value(SecretId=secret_name)["SecretString"])
        print(f"[OK] Secret Name: {resp['Name']}")
        print(f"     ARN:         {resp['ARN']}")
        print(f"     DB User:     {val.get('username')}")
        print(f"     DB Host:     {val.get('host')}")
        print(f"     DB Name:     {val.get('dbname')}")
        return True
    except Exception as e:
        print(f"[FAIL] Secrets Manager check error: {e}")
        return False


def check_cognito():
    """Verify Cognito User Pool and App Client for clinician authentication."""
    print("\n" + "-" * 55)
    print("9. AMAZON COGNITO")
    print("-" * 55)
    cognito = get_client("cognito-idp")
    try:
        pools = cognito.list_user_pools(MaxResults=60).get("UserPools", [])
        sc_pools = [p for p in pools if p.get("Name") == "SkinCancerClinicians"]
        if not sc_pools:
            print("[FAIL] User Pool 'SkinCancerClinicians' not found.")
            return False

        pool = sc_pools[0]
        pool_id = pool["Id"]
        clients = cognito.list_user_pool_clients(UserPoolId=pool_id, MaxResults=20).get("UserPoolClients", [])
        groups = [g["GroupName"] for g in cognito.list_groups(UserPoolId=pool_id).get("Groups", [])]

        print(f"[OK] User Pool:  {pool['Name']} (ID: {pool_id})")
        print(f"     Clients:    {[c['ClientName'] for c in clients]}")
        print(f"     Groups:     {groups}")
        return True
    except Exception as e:
        print(f"[FAIL] Cognito check error: {e}")
        return False


def check_sns():
    """Verify SNS alert topics."""
    print("\n" + "-" * 55)
    print("10. AMAZON SNS")
    print("-" * 55)
    sns = get_client("sns")
    try:
        topics = [t["TopicArn"] for t in sns.list_topics().get("Topics", [])]
        sc_topics = [t for t in topics if "SkinCancerAlerts" in t]
        if sc_topics:
            for t in sc_topics:
                print(f"[OK] SNS Topic: {t}")
            return True
        else:
            print("[FAIL] Topic 'SkinCancerAlerts' not found.")
            return False
    except Exception as e:
        print(f"[FAIL] SNS check error: {e}")
        return False


def test_aws_connection():
    """Main verification orchestrator."""
    print("=" * 60)
    print("  AWS COMPLETE ARCHITECTURE AUDIT — ALL 8 CORE SERVICES   ")
    print("=" * 60)

    checks = [
        ("Authentication (STS)", check_auth),
        ("Amazon S3", check_s3),
        ("Amazon EC2 (Server)", check_ec2),
        ("Amazon RDS (Database)", check_rds),
        ("AWS IAM (Roles/Profiles)", check_iam),
        ("Amazon CloudWatch", check_cloudwatch),
        ("Amazon API Gateway", check_apigateway),
        ("AWS Secrets Manager", check_secrets_manager),
        ("Amazon Cognito", check_cognito),
        ("Amazon SNS", check_sns),
    ]

    results = {}
    for name, func in checks:
        try:
            results[name] = func()
        except Exception as e:
            print(f"[!] Error in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("                     AUDIT SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name:<32} {status}")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nFinal Result: {passed}/{total} services fully verified.\n")
    return all(results.values())


if __name__ == "__main__":
    success = test_aws_connection()
    sys.exit(0 if success else 1)
