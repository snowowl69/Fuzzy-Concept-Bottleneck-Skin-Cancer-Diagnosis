import os
import json
import logging
import boto3
from botocore.exceptions import ClientError
import watchtower

class AWSManager:
    def __init__(self, region_name='us-east-1'):
        self.region_name = region_name
        self.s3_client = boto3.client('s3', region_name=self.region_name)
        self.secrets_client = boto3.client('secretsmanager', region_name=self.region_name)
        self.cognito_client = boto3.client('cognito-idp', region_name=self.region_name)
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'skin-cancer-diagnosis-bucket')
        self.setup_cloudwatch()

    def setup_cloudwatch(self):
        """Set up Amazon CloudWatch logging."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Assuming EC2 IAM role has CloudWatch logs access
        try:
            cw_handler = watchtower.CloudWatchLogHandler(
                log_group_name='SkinCancerApp/InferenceLogs',
                boto3_client=boto3.client('logs', region_name=self.region_name)
            )
            self.logger.addHandler(cw_handler)
        except Exception as e:
            print(f"CloudWatch setup failed (running locally?): {e}")

    def get_db_secret(self, secret_name='skincancer/rds/credentials'):
        """Retrieve RDS database credentials from AWS Secrets Manager."""
        try:
            get_secret_value_response = self.secrets_client.get_secret_value(SecretId=secret_name)
            if 'SecretString' in get_secret_value_response:
                secret = get_secret_value_response['SecretString']
                return json.loads(secret)
        except ClientError as e:
            self.logger.error(f"Error retrieving secret: {e}")
            print(f"Error retrieving secret: {e}")
            return None

    def upload_image_to_s3(self, file_bytes, object_name):
        """Upload uploaded dermoscopic images and generated masks to Amazon S3."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=file_bytes,
                ContentType='image/jpeg'
            )
            s3_url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{object_name}"
            self.logger.info(f"Successfully uploaded {object_name} to S3")
            return s3_url
        except ClientError as e:
            self.logger.error(f"S3 Upload Error: {e}")
            return None

    def verify_cognito_token(self, access_token):
        """Verify Amazon Cognito user access token."""
        try:
            response = self.cognito_client.get_user(AccessToken=access_token)
            return response['Username']
        except ClientError as e:
            self.logger.error(f"Cognito Auth Error: {e}")
            return None

    def log_inference(self, message):
        """Log inference metrics and errors to CloudWatch."""
        self.logger.info(message)


def require_cognito_auth(aws_mgr_getter):
    """Flask route decorator enforcing Amazon Cognito JWT bearer token validation.

    When REQUIRE_AUTH=true in .env, requires 'Authorization: Bearer <token>'.
    When REQUIRE_AUTH=false or unset (local development), allows anonymous requests.
    """
    from functools import wraps
    from flask import request, jsonify

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            require_auth = os.getenv("REQUIRE_AUTH", "false").lower() in ("true", "1", "yes")
            if not require_auth:
                return f(*args, **kwargs)

            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return jsonify({"error": "Authorization token required (Bearer <token>)"}), 401

            token = auth_header.split(" ", 1)[1].strip()
            _, _, _, aws_mgr, _ = aws_mgr_getter()

            if not aws_mgr:
                return jsonify({"error": "Cognito authentication service unavailable"}), 503

            username = aws_mgr.verify_cognito_token(token)
            if not username:
                return jsonify({"error": "Invalid or expired Cognito authorization token"}), 401

            # Attach user identity to request context if needed
            request.cognito_user = username
            return f(*args, **kwargs)

        return decorated_function

    return decorator

