"""
Amazon RDS PostgreSQL Setup Script — Skin Cancer Diagnosis Project
Provisions DB Subnet Group, Security Group, and PostgreSQL Database Instance.
"""

import os
import json
import time
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DB_IDENTIFIER = "skincancer-db"
DB_NAME = "skincancerdb"
DB_SUBNET_GROUP = "skincancer-dbsubnetgroup"
RDS_SG_NAME = "SkinCancer-RDS-SG"
SECRET_NAME = "skincancer/rds/credentials"

def get_db_credentials(sm_client):
    """Retrieve database password and user from Secrets Manager."""
    try:
        resp = sm_client.get_secret_value(SecretId=SECRET_NAME)
        data = json.loads(resp["SecretString"])
        return data["username"], data["password"]
    except Exception as e:
        print(f"[!] Could not retrieve secret '{SECRET_NAME}': {e}")
        raise

def setup_rds_security_group(ec2_client, vpc_id):
    """Create or retrieve Security Group for RDS permitting PostgreSQL (port 5432)."""
    try:
        sgs = ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [RDS_SG_NAME]}]
        )["SecurityGroups"]
        if sgs:
            sg_id = sgs[0]["GroupId"]
            print(f"[✓] RDS Security Group '{RDS_SG_NAME}' exists: {sg_id}")
            return sg_id
    except ClientError as e:
        pass

    print(f"Creating Security Group '{RDS_SG_NAME}' in VPC {vpc_id}...")
    sg_resp = ec2_client.create_security_group(
        GroupName=RDS_SG_NAME,
        Description="Security group for Skin Cancer PostgreSQL Database",
        VpcId=vpc_id,
        TagSpecifications=[{
            "ResourceType": "security-group",
            "Tags": [
                {"Key": "Name", "Value": RDS_SG_NAME},
                {"Key": "Project", "Value": "FuzzyConceptBottleneck"}
            ]
        }]
    )
    sg_id = sg_resp["GroupId"]

    # Allow port 5432 from anywhere (for development access)
    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 5432,
                "ToPort": 5432,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "PostgreSQL access"}]
            }
        ]
    )
    print(f"[✓] Created RDS Security Group: {sg_id}")
    return sg_id

def setup_db_subnet_group(rds_client, ec2_client, vpc_id):
    """Create or retrieve RDS DB Subnet Group across multiple AZs."""
    try:
        resp = rds_client.describe_db_subnet_groups(DBSubnetGroupName=DB_SUBNET_GROUP)
        print(f"[✓] DB Subnet Group '{DB_SUBNET_GROUP}' exists.")
        return DB_SUBNET_GROUP
    except ClientError as e:
        if e.response["Error"]["Code"] != "DBSubnetGroupNotFoundFault":
            raise

    # Query subnets in default VPC
    subnets = ec2_client.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
    )["Subnets"]
    subnet_ids = [s["SubnetId"] for s in subnets]

    print(f"Creating DB Subnet Group '{DB_SUBNET_GROUP}' with {len(subnet_ids)} subnets...")
    rds_client.create_db_subnet_group(
        DBSubnetGroupName=DB_SUBNET_GROUP,
        DBSubnetGroupDescription="Subnet group for Skin Cancer Diagnosis PostgreSQL DB",
        SubnetIds=subnet_ids,
        Tags=[
            {"Key": "Name", "Value": DB_SUBNET_GROUP},
            {"Key": "Project", "Value": "FuzzyConceptBottleneck"}
        ]
    )
    print(f"[✓] Created DB Subnet Group: {DB_SUBNET_GROUP}")
    return DB_SUBNET_GROUP

def setup_rds():
    print(f"Connecting to AWS RDS & EC2 in {REGION}...")
    rds_client = boto3.client("rds", region_name=REGION)
    ec2_client = boto3.client("ec2", region_name=REGION)
    sm_client = boto3.client("secretsmanager", region_name=REGION)

    # Get default VPC
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        vpcs = ec2_client.describe_vpcs()["Vpcs"]
    vpc_id = vpcs[0]["VpcId"]
    print(f"[✓] Using VPC: {vpc_id}")

    # Set up Subnet Group & Security Group
    subnet_group = setup_db_subnet_group(rds_client, ec2_client, vpc_id)
    sg_id = setup_rds_security_group(ec2_client, vpc_id)

    # Check if DB instance already exists
    try:
        resp = rds_client.describe_db_instances(DBInstanceIdentifier=DB_IDENTIFIER)
        instance = resp["DBInstances"][0]
        status = instance["DBInstanceStatus"]
        endpoint = instance.get("Endpoint", {}).get("Address", "Provisioning...")
        print(f"[✓] RDS DB Instance '{DB_IDENTIFIER}' already exists (Status: {status}).")
        print(f"    Endpoint: {endpoint}")
        return instance
    except ClientError as e:
        if e.response["Error"]["Code"] != "DBInstanceNotFound":
            raise

    # Get credentials from Secrets Manager
    username, password = get_db_credentials(sm_client)

    print(f"Creating PostgreSQL DB Instance '{DB_IDENTIFIER}' (db.t3.micro)...")
    create_resp = rds_client.create_db_instance(
        DBInstanceIdentifier=DB_IDENTIFIER,
        AllocatedStorage=20,
        DBInstanceClass="db.t3.micro",
        Engine="postgres",
        MasterUsername=username,
        MasterUserPassword=password,
        DBName=DB_NAME,
        VpcSecurityGroupIds=[sg_id],
        DBSubnetGroupName=subnet_group,
        PubliclyAccessible=True,
        StorageType="gp2",
        BackupRetentionPeriod=1,
        EnablePerformanceInsights=False,
        MultiAZ=False,
        Tags=[
            {"Key": "Name", "Value": DB_IDENTIFIER},
            {"Key": "Project", "Value": "FuzzyConceptBottleneck"}
        ]
    )
    instance = create_resp["DBInstance"]
    print(f"[✓] RDS DB Instance creation initiated!")
    print(f"    Identifier: {instance['DBInstanceIdentifier']}")
    print(f"    Status:     {instance['DBInstanceStatus']}")
    print(f"    Engine:     {instance['Engine']} {instance.get('EngineVersion', '')}")
    return instance

if __name__ == "__main__":
    setup_rds()
