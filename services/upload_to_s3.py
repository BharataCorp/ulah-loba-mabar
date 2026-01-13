# create class service to upload file result generate video to vultr s3 with parameters: with  ACL: "public-read", must return the public url of the uploaded file
# - file_path: str
# - upload_s3_endpoint: str
# - upload_s3_access_key: str
# - upload_s3_secret_key: str
# - upload_s3_bucket: str

import boto3
from botocore.client import Config
import os
from typing import Optional

class S3Uploader:
    # create documentation for the usage of the method upload_file_to_s3

    @staticmethod
    def upload_file_to_s3(
        file_path: str,
        upload_s3_endpoint: str,
        upload_s3_access_key: str,
        upload_s3_secret_key: str,
        upload_s3_bucket: str,
        s3_base_path: Optional[str] = None,
        file_name: Optional[str] = None
    ) -> str:
        """
        Uploads a file to S3 and returns the public URL of the uploaded file.

        :param file_path: Local path to the file to be uploaded.
        :param upload_s3_endpoint: S3 endpoint URL.
        :param upload_s3_access_key: S3 access key.
        :param upload_s3_secret_key: S3 secret key.
        :param upload_s3_bucket: S3 bucket name.
        :param s3_base_path: Base path in the S3 bucket where the file will be uploaded.
        :return: Public URL of the uploaded file.

        Example usage:
        public_url = S3Uploader.upload_file_to_s3(
            file_path="/path/to/local/file.mp4",
            upload_s3_endpoint="https://s3.example.com",
            upload_s3_access_key="your_access_key",
            upload_s3_secret_key="your_secret_key",
            upload_s3_bucket="your_bucket_name",
            s3_base_path="optional/base/path"
        )

        then returned public_url will be the URL to access the uploaded file.

        """
        s3_client = boto3.client(
            's3',
            endpoint_url=upload_s3_endpoint,
            aws_access_key_id=upload_s3_access_key,
            aws_secret_access_key=upload_s3_secret_key,
            config=Config(signature_version='s3v4')
        )

        # file_name = os.path.basename(file_path)

        if not file_name:
            file_name = os.path.basename(file_path)

        s3_key = f"{s3_base_path}/{file_name}" if s3_base_path else file_name

        s3_client.upload_file(
            Filename=file_path,
            Bucket=upload_s3_bucket,
            Key=s3_key,
            ExtraArgs={'ACL': 'public-read'}
        )

        public_url = f"{upload_s3_endpoint}/{upload_s3_bucket}/{s3_key}"
        return public_url
