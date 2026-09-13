"""
Amazon API Gateway Setup Script — Skin Cancer Diagnosis Project
Creates REST API, Cognito Authorizer, Proxy Resources, and Deploys 'prod' stage.
"""

import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
API_NAME = "SkinCancerDiagnosisAPI"
STAGE_NAME = "prod"

def get_ec2_dns(ec2_client):
    """Get the public DNS or IP of the running AI Pipeline EC2 instance."""
    instances = ec2_client.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": ["SkinCancer-AIPipeline-Server"]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]}
        ]
    )
    for res in instances.get("Reservations", []):
        for inst in res.get("Instances", []):
            dns = inst.get("PublicDnsName") or inst.get("PublicIpAddress")
            if dns:
                return dns
    return "localhost"

def get_cognito_pool_arn(cognito_client):
    """Retrieve ARN of SkinCancerClinicians User Pool."""
    pools = cognito_client.list_user_pools(MaxResults=60)["UserPools"]
    for p in pools:
        if p["Name"] == "SkinCancerClinicians":
            pool_id = p["Id"]
            desc = cognito_client.describe_user_pool(UserPoolId=pool_id)
            return desc["UserPool"]["Arn"]
    return None

def setup_apigateway():
    print(f"Connecting to Amazon API Gateway in {REGION}...")
    apigw = boto3.client("apigateway", region_name=REGION)
    ec2_client = boto3.client("ec2", region_name=REGION)
    cognito_client = boto3.client("cognito-idp", region_name=REGION)

    ec2_dns = get_ec2_dns(ec2_client)
    cognito_arn = get_cognito_pool_arn(cognito_client)
    backend_url = f"http://{ec2_dns}"
    print(f"[✓] Backend EC2 Endpoint: {backend_url}")

    # 1. Check if REST API already exists
    apis = apigw.get_rest_apis().get("items", [])
    api_id = None
    for item in apis:
        if item["name"] == API_NAME:
            api_id = item["id"]
            print(f"[✓] REST API '{API_NAME}' already exists (ID: {api_id}).")
            break

    if not api_id:
        print(f"Creating REST API '{API_NAME}'...")
        resp = apigw.create_rest_api(
            name=API_NAME,
            description="API Gateway routing for Fuzzy Concept Bottleneck Diagnosis Backend",
            endpointConfiguration={"types": ["REGIONAL"]},
            tags={"Project": "FuzzyConceptBottleneck"}
        )
        api_id = resp["id"]
        print(f"[✓] Created REST API: {api_id}")

    # 2. Get Root Resource ID
    resources = apigw.get_resources(restApiId=api_id).get("items", [])
    root_id = None
    proxy_id = None
    for r in resources:
        if r["path"] == "/":
            root_id = r["id"]
        elif r.get("pathPart") == "{proxy+}":
            proxy_id = r["id"]

    # 3. Create Cognito Authorizer if available
    if cognito_arn:
        authorizers = apigw.get_authorizers(restApiId=api_id).get("items", [])
        auth_exists = any(a["name"] == "CognitoUserAuthorizer" for a in authorizers)
        if not auth_exists:
            print("Creating Cognito Authorizer...")
            apigw.create_authorizer(
                restApiId=api_id,
                name="CognitoUserAuthorizer",
                type="COGNITO_USER_POOLS",
                providerARNs=[cognito_arn],
                identitySource="method.request.header.Authorization"
            )
            print("[✓] Created Cognito Authorizer.")

    # 4. Create root ANY method (HTTP_PROXY)
    try:
        apigw.put_method(
            restApiId=api_id,
            resourceId=root_id,
            httpMethod="ANY",
            authorizationType="NONE"
        )
        apigw.put_integration(
            restApiId=api_id,
            resourceId=root_id,
            httpMethod="ANY",
            type="HTTP_PROXY",
            integrationHttpMethod="ANY",
            uri=f"{backend_url}/"
        )
        print("[✓] Configured Root (/) Proxy Method.")
    except ClientError:
        pass

    # 5. Create {proxy+} resource and ANY method
    if not proxy_id:
        print("Creating {proxy+} resource...")
        res = apigw.create_resource(
            restApiId=api_id,
            parentId=root_id,
            pathPart="{proxy+}"
        )
        proxy_id = res["id"]

    try:
        apigw.put_method(
            restApiId=api_id,
            resourceId=proxy_id,
            httpMethod="ANY",
            authorizationType="NONE",
            requestParameters={"method.request.path.proxy": True}
        )
        apigw.put_integration(
            restApiId=api_id,
            resourceId=proxy_id,
            httpMethod="ANY",
            type="HTTP_PROXY",
            integrationHttpMethod="ANY",
            uri=f"{backend_url}/{{proxy}}",
            requestParameters={"integration.request.path.proxy": "method.request.path.proxy"}
        )
        print("[✓] Configured /{proxy+} Proxy Method.")
    except ClientError:
        pass

    # 6. Deploy to Stage
    print(f"Deploying API to stage '{STAGE_NAME}'...")
    dep = apigw.create_deployment(
        restApiId=api_id,
        stageName=STAGE_NAME,
        description="Production deployment for Skin Cancer Diagnosis API"
    )

    invoke_url = f"https://{api_id}.execute-api.{REGION}.amazonaws.com/{STAGE_NAME}/"
    print("\n" + "="*50)
    print("         API GATEWAY SETUP SUMMARY")
    print("="*50)
    print(f"  API Name:   {API_NAME}")
    print(f"  API ID:     {api_id}")
    print(f"  Stage:      {STAGE_NAME}")
    print(f"  Invoke URL: {invoke_url}")
    print("="*50)

    return api_id, invoke_url

if __name__ == "__main__":
    setup_apigateway()
