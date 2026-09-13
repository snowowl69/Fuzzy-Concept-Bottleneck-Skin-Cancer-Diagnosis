"""
AWS CloudWatch Setup Script
=============================
Creates CloudWatch log groups, metric filters, alarms, and a monitoring
dashboard for the Fuzzy Concept Bottleneck Skin Cancer Diagnosis system.

Log Groups:
    SkinCancerApp/InferenceLogs   - AI inference predictions and metrics
    SkinCancerApp/ApplicationLogs - Flask/Gunicorn web app logs
    SkinCancerApp/ErrorLogs       - Application errors and exceptions
    SkinCancerApp/AccessLogs      - API Gateway access logs

Metric Filters:
    InferenceErrorCount  - Counts ERROR-level log entries
    InferenceLatency     - Extracts inference time metrics

Alarms:
    SkinCancer-HighErrorRate       - >5 errors in 5 minutes
    SkinCancer-HighInferenceLatency - avg latency >10s over 5 minutes

Dashboard:
    SkinCancerDiagnosisDashboard   - Combined monitoring view
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

# Configuration
LOG_GROUPS = [
    {
        "name": "SkinCancerApp/InferenceLogs",
        "retention_days": 30,
        "description": "AI inference predictions, concept scores, and timing metrics",
    },
    {
        "name": "SkinCancerApp/ApplicationLogs",
        "retention_days": 30,
        "description": "Flask/Gunicorn web application logs",
    },
    {
        "name": "SkinCancerApp/ErrorLogs",
        "retention_days": 60,
        "description": "Application errors and exception stack traces",
    },
    {
        "name": "SkinCancerApp/AccessLogs",
        "retention_days": 14,
        "description": "API Gateway access logs",
    },
]


def get_clients(region):
    """Create boto3 clients for CloudWatch and CloudWatch Logs."""
    kwargs = {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "region_name": region,
    }
    logs_client = boto3.client("logs", **kwargs)
    cw_client = boto3.client("cloudwatch", **kwargs)
    sns_client = boto3.client("sns", **kwargs)
    return logs_client, cw_client, sns_client


def create_log_groups(logs_client):
    """Create all required CloudWatch Log Groups with retention policies."""
    print("\n[+] Creating CloudWatch Log Groups...")

    existing = set()
    try:
        paginator = logs_client.get_paginator("describe_log_groups")
        for page in paginator.paginate(logGroupNamePrefix="SkinCancerApp/"):
            for group in page.get("logGroups", []):
                existing.add(group["logGroupName"])
    except ClientError:
        pass

    for lg in LOG_GROUPS:
        name = lg["name"]
        if name in existing:
            print(f"    [✓] Log group '{name}' already exists.")
        else:
            try:
                logs_client.create_log_group(logGroupName=name)
                print(f"    [✓] Created log group '{name}'")
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                    print(f"    [✓] Log group '{name}' already exists.")
                else:
                    print(f"    [!] Failed to create '{name}': {e}")
                    continue

        # Set retention policy
        try:
            logs_client.put_retention_policy(
                logGroupName=name, retentionInDays=lg["retention_days"]
            )
            print(f"        Retention: {lg['retention_days']} days")
        except ClientError as e:
            print(f"        [!] Failed to set retention for '{name}': {e}")

    # Tag log groups with project metadata
    for lg in LOG_GROUPS:
        try:
            logs_client.tag_log_group(
                logGroupName=lg["name"],
                tags={
                    "Project": "FuzzyConceptBottleneck",
                    "Environment": "Production",
                    "Description": lg["description"],
                },
            )
        except ClientError:
            pass  # Tagging is non-critical


def create_metric_filters(logs_client):
    """Create metric filters on InferenceLogs for error counting and latency tracking."""
    print("\n[+] Creating CloudWatch Metric Filters...")

    filters = [
        {
            "filterName": "InferenceErrorCount",
            "logGroupName": "SkinCancerApp/InferenceLogs",
            "filterPattern": "ERROR",
            "metricTransformations": [
                {
                    "metricName": "InferenceErrors",
                    "metricNamespace": "SkinCancerApp",
                    "metricValue": "1",
                    "defaultValue": 0,
                }
            ],
        },
        {
            "filterName": "InferenceSuccessCount",
            "logGroupName": "SkinCancerApp/InferenceLogs",
            "filterPattern": "Successful inference",
            "metricTransformations": [
                {
                    "metricName": "InferenceSuccesses",
                    "metricNamespace": "SkinCancerApp",
                    "metricValue": "1",
                    "defaultValue": 0,
                }
            ],
        },
        {
            "filterName": "ApplicationErrorCount",
            "logGroupName": "SkinCancerApp/ErrorLogs",
            "filterPattern": "ERROR Exception Traceback",
            "metricTransformations": [
                {
                    "metricName": "ApplicationErrors",
                    "metricNamespace": "SkinCancerApp",
                    "metricValue": "1",
                    "defaultValue": 0,
                }
            ],
        },
        {
            "filterName": "CounterfactualGenerationCount",
            "logGroupName": "SkinCancerApp/InferenceLogs",
            "filterPattern": "Counterfactual generated",
            "metricTransformations": [
                {
                    "metricName": "CounterfactualGenerations",
                    "metricNamespace": "SkinCancerApp",
                    "metricValue": "1",
                    "defaultValue": 0,
                }
            ],
        },
        {
            "filterName": "InterventionRequestCount",
            "logGroupName": "SkinCancerApp/InferenceLogs",
            "filterPattern": "Intervention requested",
            "metricTransformations": [
                {
                    "metricName": "InterventionRequests",
                    "metricNamespace": "SkinCancerApp",
                    "metricValue": "1",
                    "defaultValue": 0,
                }
            ],
        },
    ]

    for f in filters:
        try:
            logs_client.put_metric_filter(
                logGroupName=f["logGroupName"],
                filterName=f["filterName"],
                filterPattern=f["filterPattern"],
                metricTransformations=f["metricTransformations"],
            )
            print(f"    [✓] Metric filter '{f['filterName']}' created on '{f['logGroupName']}'")
        except ClientError as e:
            print(f"    [!] Failed to create metric filter '{f['filterName']}': {e}")


def get_or_create_sns_topic(sns_client):
    """Get or create the SNS topic for alarm notifications."""
    topic_name = "SkinCancerAlerts"
    try:
        response = sns_client.create_topic(
            Name=topic_name,
            Tags=[
                {"Key": "Project", "Value": "FuzzyConceptBottleneck"},
            ],
        )
        topic_arn = response["TopicArn"]
        print(f"\n[✓] SNS Topic '{topic_name}' ready: {topic_arn}")
        return topic_arn
    except ClientError as e:
        print(f"\n[!] SNS topic creation failed: {e}")
        return None


def create_alarms(cw_client, sns_topic_arn):
    """Create CloudWatch alarms for error rate and latency monitoring."""
    print("\n[+] Creating CloudWatch Alarms...")

    alarm_actions = [sns_topic_arn] if sns_topic_arn else []

    alarms = [
        {
            "AlarmName": "SkinCancer-HighErrorRate",
            "AlarmDescription": "Triggers when more than 5 inference errors occur within 5 minutes",
            "Namespace": "SkinCancerApp",
            "MetricName": "InferenceErrors",
            "Statistic": "Sum",
            "Period": 300,
            "EvaluationPeriods": 1,
            "Threshold": 5.0,
            "ComparisonOperator": "GreaterThanThreshold",
            "TreatMissingData": "notBreaching",
            "ActionsEnabled": True,
            "AlarmActions": alarm_actions,
        },
        {
            "AlarmName": "SkinCancer-HighApplicationErrorRate",
            "AlarmDescription": "Triggers when more than 10 application errors occur within 10 minutes",
            "Namespace": "SkinCancerApp",
            "MetricName": "ApplicationErrors",
            "Statistic": "Sum",
            "Period": 600,
            "EvaluationPeriods": 1,
            "Threshold": 10.0,
            "ComparisonOperator": "GreaterThanThreshold",
            "TreatMissingData": "notBreaching",
            "ActionsEnabled": True,
            "AlarmActions": alarm_actions,
        },
        {
            "AlarmName": "SkinCancer-S3BucketSizeAlert",
            "AlarmDescription": "Triggers when S3 bucket exceeds 5 GB",
            "Namespace": "AWS/S3",
            "MetricName": "BucketSizeBytes",
            "Statistic": "Average",
            "Period": 86400,
            "EvaluationPeriods": 1,
            "Threshold": 5368709120.0,  # 5 GB
            "ComparisonOperator": "GreaterThanThreshold",
            "Dimensions": [
                {"Name": "BucketName", "Value": os.getenv("S3_BUCKET_NAME", "skin-cancer-diagnosis-bucket")},
                {"Name": "StorageType", "Value": "StandardStorage"},
            ],
            "TreatMissingData": "notBreaching",
            "ActionsEnabled": True,
            "AlarmActions": alarm_actions,
        },
    ]

    for alarm in alarms:
        try:
            cw_client.put_metric_alarm(**alarm)
            print(f"    [✓] Alarm '{alarm['AlarmName']}' created.")
        except ClientError as e:
            print(f"    [!] Failed to create alarm '{alarm['AlarmName']}': {e}")


def create_dashboard(cw_client, region):
    """Create a CloudWatch dashboard for unified monitoring."""
    print("\n[+] Creating CloudWatch Dashboard...")

    bucket_name = os.getenv("S3_BUCKET_NAME", "skin-cancer-diagnosis-bucket")

    dashboard_body = {
        "widgets": [
            {
                "type": "text",
                "x": 0,
                "y": 0,
                "width": 24,
                "height": 1,
                "properties": {
                    "markdown": "# 🔬 Fuzzy Concept Bottleneck — Skin Cancer Diagnosis Monitoring"
                },
            },
            {
                "type": "metric",
                "x": 0,
                "y": 1,
                "width": 8,
                "height": 6,
                "properties": {
                    "title": "Inference Errors (5 min)",
                    "metrics": [
                        ["SkinCancerApp", "InferenceErrors", {"stat": "Sum", "period": 300}]
                    ],
                    "view": "timeSeries",
                    "region": region,
                    "yAxis": {"left": {"min": 0}},
                },
            },
            {
                "type": "metric",
                "x": 8,
                "y": 1,
                "width": 8,
                "height": 6,
                "properties": {
                    "title": "Successful Inferences (5 min)",
                    "metrics": [
                        ["SkinCancerApp", "InferenceSuccesses", {"stat": "Sum", "period": 300}]
                    ],
                    "view": "timeSeries",
                    "region": region,
                    "yAxis": {"left": {"min": 0}},
                },
            },
            {
                "type": "metric",
                "x": 16,
                "y": 1,
                "width": 8,
                "height": 6,
                "properties": {
                    "title": "Application Errors (10 min)",
                    "metrics": [
                        ["SkinCancerApp", "ApplicationErrors", {"stat": "Sum", "period": 600}]
                    ],
                    "view": "timeSeries",
                    "region": region,
                    "yAxis": {"left": {"min": 0}},
                },
            },
            {
                "type": "metric",
                "x": 0,
                "y": 7,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Counterfactual Generations & Interventions",
                    "metrics": [
                        ["SkinCancerApp", "CounterfactualGenerations", {"stat": "Sum", "period": 300}],
                        ["SkinCancerApp", "InterventionRequests", {"stat": "Sum", "period": 300}],
                    ],
                    "view": "timeSeries",
                    "region": region,
                    "yAxis": {"left": {"min": 0}},
                },
            },
            {
                "type": "metric",
                "x": 12,
                "y": 7,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "S3 Bucket Size (Bytes)",
                    "metrics": [
                        [
                            "AWS/S3",
                            "BucketSizeBytes",
                            "BucketName",
                            bucket_name,
                            "StorageType",
                            "StandardStorage",
                            {"stat": "Average", "period": 86400},
                        ]
                    ],
                    "view": "singleValue",
                    "region": region,
                },
            },
            {
                "type": "log",
                "x": 0,
                "y": 13,
                "width": 24,
                "height": 6,
                "properties": {
                    "title": "Recent Inference Logs",
                    "query": (
                        "SOURCE 'SkinCancerApp/InferenceLogs' "
                        "| fields @timestamp, @message "
                        "| sort @timestamp desc "
                        "| limit 50"
                    ),
                    "region": region,
                    "view": "table",
                },
            },
            {
                "type": "log",
                "x": 0,
                "y": 19,
                "width": 24,
                "height": 6,
                "properties": {
                    "title": "Recent Error Logs",
                    "query": (
                        "SOURCE 'SkinCancerApp/ErrorLogs' "
                        "| fields @timestamp, @message "
                        "| sort @timestamp desc "
                        "| limit 20"
                    ),
                    "region": region,
                    "view": "table",
                },
            },
        ]
    }

    try:
        cw_client.put_dashboard(
            DashboardName="SkinCancerDiagnosisDashboard",
            DashboardBody=json.dumps(dashboard_body),
        )
        print("[✓] Dashboard 'SkinCancerDiagnosisDashboard' created successfully.")
        print(f"    View at: https://{region}.console.aws.amazon.com/cloudwatch/home"
              f"?region={region}#dashboards:name=SkinCancerDiagnosisDashboard")
    except ClientError as e:
        print(f"[!] Dashboard creation failed: {e}")


def setup_cloudwatch():
    """Main entry point: set up all CloudWatch resources."""
    print("=" * 60)
    print("AWS CLOUDWATCH SETUP")
    print("=" * 60)

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("[!] AWS credentials not found in .env file. Aborting.")
        return False

    logs_client, cw_client, sns_client = get_clients(region)

    create_log_groups(logs_client)
    create_metric_filters(logs_client)
    sns_topic_arn = get_or_create_sns_topic(sns_client)
    create_alarms(cw_client, sns_topic_arn)
    create_dashboard(cw_client, region)

    print("\n" + "=" * 60)
    print("CLOUDWATCH SETUP COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = setup_cloudwatch()
    sys.exit(0 if success else 1)
