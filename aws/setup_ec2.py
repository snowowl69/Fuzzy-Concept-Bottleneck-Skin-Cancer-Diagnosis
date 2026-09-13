"""
Amazon EC2 Setup Script — Skin Cancer Diagnosis Project
Creates Security Group, Key Pair, and launches EC2 instance hosting the AI pipeline.
"""

import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
INSTANCE_NAME = "SkinCancer-AIPipeline-Server"
KEY_PAIR_NAME = "skincancer-keypair"
EC2_SG_NAME = "SkinCancer-EC2-SG"
INSTANCE_PROFILE_NAME = "SkinCancerEC2InstanceProfile"
INSTANCE_TYPE = "t3.small"

USER_DATA_SCRIPT = """#!/bin/bash
set -e

# Update and install system dependencies
echo "[1/4] Updating package lists..."
apt-get update -y
apt-get install -y python3-pip python3-venv git nginx libgl1-mesa-glx awscli

# Setup application directory
echo "[2/4] Setting up application directories..."
mkdir -p /home/ubuntu/skin_cancer_app
chown -R ubuntu:ubuntu /home/ubuntu/skin_cancer_app

# Set up systemd service for Gunicorn
echo "[3/4] Configuring Gunicorn systemd service..."
cat << 'EOF' > /etc/systemd/system/skincancer.service
[Unit]
Description=Skin Cancer Diagnosis Gunicorn Application Server
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/skin_cancer_app
Environment="PATH=/home/ubuntu/skin_cancer_app/venv/bin"
Environment="AWS_DEFAULT_REGION=us-east-1"
ExecStart=/usr/local/bin/gunicorn --workers 3 --bind 0.0.0.0:5000 web.app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Setup Nginx reverse proxy
echo "[4/4] Configuring Nginx reverse proxy..."
cat << 'EOF' > /etc/nginx/sites-available/skincancer
server {
    listen 80;
    server_name _;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/skincancer /etc/nginx/sites-enabled/default
systemctl restart nginx

echo "Bootstrap complete!"
"""

def get_ubuntu_ami(ec2_client):
    """Find the latest Ubuntu 22.04 LTS AMI for us-east-1."""
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        param = ssm.get_parameter(
            Name="/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id"
        )
        ami_id = param["Parameter"]["Value"]
        print(f"[✓] Retrieved Ubuntu 22.04 LTS AMI via SSM: {ami_id}")
        return ami_id
    except Exception:
        # Fallback to image query
        images = ec2_client.describe_images(
            Owners=["099720109477"],  # Canonical
            Filters=[
                {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
                {"Name": "state", "Values": ["available"]}
            ]
        )["Images"]
        images.sort(key=lambda x: x["CreationDate"], reverse=True)
        ami_id = images[0]["ImageId"]
        print(f"[✓] Retrieved Ubuntu 22.04 LTS AMI via search: {ami_id}")
        return ami_id

def setup_key_pair(ec2_client):
    """Create or verify SSH key pair."""
    try:
        resp = ec2_client.describe_key_pairs(KeyNames=[KEY_PAIR_NAME])
        print(f"[✓] Key Pair '{KEY_PAIR_NAME}' exists.")
        return KEY_PAIR_NAME
    except ClientError as e:
        if "InvalidKeyPair.NotFound" not in str(e):
            raise

    print(f"Creating Key Pair '{KEY_PAIR_NAME}'...")
    key = ec2_client.create_key_pair(KeyName=KEY_PAIR_NAME)
    key_file = os.path.join(os.path.dirname(__file__), f"{KEY_PAIR_NAME}.pem")
    with open(key_file, "w") as f:
        f.write(key["KeyMaterial"])
    print(f"[✓] Created Key Pair and saved private key to: {key_file}")
    return KEY_PAIR_NAME

def setup_ec2_security_group(ec2_client, vpc_id):
    """Create or verify EC2 security group allowing ports 22, 80, 443, 5000."""
    try:
        sgs = ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [EC2_SG_NAME]}]
        )["SecurityGroups"]
        if sgs:
            sg_id = sgs[0]["GroupId"]
            print(f"[✓] EC2 Security Group '{EC2_SG_NAME}' exists: {sg_id}")
            return sg_id
    except ClientError:
        pass

    print(f"Creating Security Group '{EC2_SG_NAME}' in VPC {vpc_id}...")
    sg_resp = ec2_client.create_security_group(
        GroupName=EC2_SG_NAME,
        Description="Security group for Skin Cancer EC2 AI Server",
        VpcId=vpc_id,
        TagSpecifications=[{
            "ResourceType": "security-group",
            "Tags": [
                {"Key": "Name", "Value": EC2_SG_NAME},
                {"Key": "Project", "Value": "FuzzyConceptBottleneck"}
            ]
        }]
    )
    sg_id = sg_resp["GroupId"]

    rules = [
        {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}]},
        {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTP"}]},
        {"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTPS"}]},
        {"IpProtocol": "tcp", "FromPort": 5000, "ToPort": 5000, "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Flask API"}]}
    ]
    ec2_client.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=rules)
    print(f"[✓] Created EC2 Security Group: {sg_id} (Ports: 22, 80, 443, 5000)")
    return sg_id

def setup_ec2():
    print(f"Connecting to AWS EC2 in {REGION}...")
    ec2_client = boto3.client("ec2", region_name=REGION)

    # Check for existing running instance
    instances = ec2_client.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [INSTANCE_NAME]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}
        ]
    )
    for res in instances.get("Reservations", []):
        for inst in res.get("Instances", []):
            print(f"[✓] EC2 Instance '{INSTANCE_NAME}' already exists:")
            print(f"    ID:        {inst['InstanceId']}")
            print(f"    State:     {inst['State']['Name']}")
            print(f"    Public IP: {inst.get('PublicIpAddress', 'N/A')}")
            print(f"    DNS:       {inst.get('PublicDnsName', 'N/A')}")
            return inst

    # Get default VPC
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    vpc_id = vpcs[0]["VpcId"]

    # Security Group, Key Pair, and AMI
    sg_id = setup_ec2_security_group(ec2_client, vpc_id)
    key_name = setup_key_pair(ec2_client)
    ami_id = get_ubuntu_ami(ec2_client)

    print(f"Launching EC2 Instance '{INSTANCE_NAME}' ({INSTANCE_TYPE})...")
    resp = ec2_client.run_instances(
        ImageId=ami_id,
        InstanceType=INSTANCE_TYPE,
        KeyName=key_name,
        SecurityGroupIds=[sg_id],
        IamInstanceProfile={"Name": INSTANCE_PROFILE_NAME},
        UserData=USER_DATA_SCRIPT,
        MinCount=1,
        MaxCount=1,
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": 30,
                "VolumeType": "gp3",
                "DeleteOnTermination": True
            }
        }],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": INSTANCE_NAME},
                {"Key": "Project", "Value": "FuzzyConceptBottleneck"},
                {"Key": "Role", "Value": "AIPipelineHost"}
            ]
        }]
    )

    inst = resp["Instances"][0]
    inst_id = inst["InstanceId"]
    print(f"[✓] EC2 Instance launched successfully!")
    print(f"    Instance ID: {inst_id}")
    print(f"    Type:        {INSTANCE_TYPE}")
    print(f"    State:       {inst['State']['Name']}")
    return inst

if __name__ == "__main__":
    setup_ec2()
