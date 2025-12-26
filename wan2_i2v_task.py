# import package boto3 and requests
from services.http_clients import Requests
import boto3
import os
from wan.config import BASE_API_URL_MABAR
import json

while True:
    try:
        if not BASE_API_URL_MABAR:
            print("BASE_URL_MABAR not set, exiting pod for stop.")
            os._exit(0)

        # Create request to BASE_URL_MABAR + "/api/v2/wan2_i2v_task/waiting" with method GET
        response = Requests.request(
            method="GET",
            url=f"{BASE_API_URL_MABAR}/api/v2/wan2_i2v_task/waiting",
        )

        if response.status_code != 200:
            Requests.stop_pod()
            # throw error
            raise ValueError(f"Request failed with status code {response.status_code}")

        task = response.json()

        wan2_i2v = task.get("wan2_i2v")
        if wan2_i2v is None:
            print(f"Found tasks: {task}")

        if not task:
            print("No tasks available, retrying...")
            continue

        generate_number = wan2_i2v.get("generate_number")
        duration = wan2_i2v.get("duration")
        # upload_s3_base_path = "video/wan2.2-i2v/" + date_path, date_path is current date in format YYYY/MM/DD
        from datetime import datetime
        date_path = datetime.now().strftime("%Y/%m/%d")
        upload_s3_base_path = "video/wan2.2-i2v/" + date_path + "/" + str(generate_number)
        upload_s3_endpoint = wan2_i2v.get("upload_s3_endpoint")
        upload_s3_access_key = wan2_i2v.get("upload_s3_access_key")
        upload_s3_secret_key = wan2_i2v.get("upload_s3_secret_key")
        upload_s3_bucket = wan2_i2v.get("upload_s3_bucket")
        public_s3_base_url = wan2_i2v.get("public_s3_base_url") # example: https://{$vultr_s3_hostname}/{$bucket_name}
        callback_url = wan2_i2v.get("callback_url")
        prompts = wan2_i2v.get("prompts")
        items = wan2_i2v.get("items")

        # check if items is empty
        if not items:
            print("No items to process in the task, skipping...")
            continue

        # items is array of objects with image_path and prompt
        for item in items:
            storyboard_data = item.get("storyboard_data")
            order_index = item.get("order_index")

            # if storyboard_data is null or empty string
            if not storyboard_data:
                print("No storyboard data found for item, skipping...")
                continue


    except Exception as e:
        print(f"Error occurred while requesting task: {e}")
        Requests.stop_pod()
        continue