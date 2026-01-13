from services.http_clients import Requests
import boto3
import os
from wan_custom.config import BASE_API_URL_MABAR, MABAR_POD_ID, OUTPUT_DIR
import json
from services.upload_to_s3 import S3Uploader

from wan_custom.pipelines.t2v_pipeline import T2VPipeline

Requests.send_log(
    f"Pod {MABAR_POD_ID} is starting WAN T2V worker.",
    "startup",
    f"Pod {MABAR_POD_ID} has started and is ready to process WAN T2V tasks."
)

while True:
    try:
        Requests.send_log(
            f"Pod {MABAR_POD_ID} checking for WAN T2V tasks.",
            "task_check",
            f"Pod {MABAR_POD_ID} is checking for WAN T2V tasks to process."
        )

        if not BASE_API_URL_MABAR:
            print("BASE_URL_MABAR not set, exiting pod for stop.")
            os._exit(0)

        response = Requests.request(
            method="GET",
            url=f"{BASE_API_URL_MABAR}/runpod_pod/{MABAR_POD_ID}/wan_t2v/ready_to_process"
        )

        if response.status_code == 200:
            Requests.stop_pod()
            raise ValueError(f"Request failed with status code {response.status_code}")

        task = response.json()

        wan_t2v = task.get("wan_t2v")

        if wan_t2v is None:
            print("wan_t2v not found, exiting pod for stop.")
            continue

        wan_t2v_id = wan_t2v.get("id")
        items = wan_t2v.get("items")
        if not items:
            print("No items to process in the task, skipping...")
            Requests.set_failed(wan_t2v_id, "Tidak ada item untuk diproses dalam tugas.")
            continue

        success_generate_item = 0

        Requests.send_log(
            f"Pod {MABAR_POD_ID} started processing WAN T2V ID {wan_t2v_id} with {len(items)} items.",
            "task_start",
            f"Pod {MABAR_POD_ID} is processing WAN T2V ID {wan_t2v_id} which has {len(items)} items."
        )

        # check if every item.status in items is FAILED
        if all(item.get("status") == "FAILED" for item in items):
            print(f"All items in WAN T2V ID {wan_t2v_id} have failed status, marking task as failed.")
            Requests.set_failed(wan_t2v_id, "Semua item dalam tugas memiliki status GAGAL.")
            continue

        if all(item.get("status") == "UPLOADED" for item in items):
            print(f"All items in WAN T2V ID {wan_t2v_id} have already been uploaded, marking task as completed.")
            Requests.request(
                "PUT",
                f"{BASE_API_URL_MABAR}/wan_one_t2v/processing/{wan_t2v_id}/completed"
            )
            continue

        for item in items:
            storyboard_data = item.get("storyboard_data")
            order_index = item.get("order_index")
            wan_t2v_item_id = item.get("id")

            # because storyboard_data is json string, we need to parse it
            if not storyboard_data:
                print("No storyboard_data found in item, skipping...")
                Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "storyboard data kosong.")
                continue

            storyboard_data = json.loads(storyboard_data)

            scenes = storyboard_data.get("scenes", [])
            if not scenes:
                print("No scenes found in storyboard_data, skipping...")
                Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "Tidak ada scene prompt dalam storyboard data.")
                continue

            Requests.send_log(
                f"Pod {MABAR_POD_ID} processing WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id} (order {order_index}).",
                "item_start",
                f"Pod {MABAR_POD_ID} is processing WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id} with order index {order_index}."
            )

            size = "480*832"
            target_duration = item.get("duration", 10)
            output_path = OUTPUT_DIR + f"/wan_t2v_{wan_t2v_id}_item_{wan_t2v_item_id}.mp4"

            output = T2VPipeline.generate(
                scenes=scenes,
                target_duration=target_duration,
                size=size,
                output_path=output_path,
            )

            # Next Step To Upload output to S3
            Requests.send_log(
                f"Pod {MABAR_POD_ID} finished generating video for WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id}.",
                "item_generated",
                f"Pod {MABAR_POD_ID} has generated video at {output} for WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id}."
            )

            upload_s3_bucket = wan_t2v.get("upload_s3_bucket")
            if not upload_s3_bucket:
                print("upload_s3_bucket not found in wan_t2v, skipping upload...")
                Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "upload_s3_bucket tidak ditemukan.")
                continue

            upload_s3_endpoint = wan_t2v.get("upload_s3_endpoint")
            if not upload_s3_endpoint:
                print("upload_s3_endpoint not found in wan_t2v, skipping upload...")
                Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "upload_s3_endpoint tidak ditemukan.")
                continue

            upload_s3_access_key = wan_t2v.get("upload_s3_access_key")
            if not upload_s3_access_key:
                print("upload_s3_access_key not found in wan_t2v, skipping upload...")
                Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "upload_s3_access_key tidak ditemukan.")
                continue

            upload_s3_secret_key = wan_t2v.get("upload_s3_secret_key")
            if not upload_s3_secret_key:
                print("upload_s3_secret_key not found in wan_t2v, skipping upload...")
                Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "upload_s3_secret_key tidak ditemukan.")
                continue

            unixtime_stamp = int(os.path.getmtime(output))
            s3_base_path = "WAN_T2V/videos/"
            file_name = f"wan_t2v_{wan_t2v_id}_item_{wan_t2v_item_id}_{unixtime_stamp}.mp4"

            try:
                ## Upload to S3
                s3_uploader = S3Uploader.upload_file_to_s3(
                    file_path=output,
                    upload_s3_endpoint=upload_s3_endpoint,
                    upload_s3_access_key=upload_s3_access_key,
                    upload_s3_secret_key=upload_s3_secret_key,
                    upload_s3_bucket=upload_s3_bucket,
                    s3_base_path=s3_base_path,
                    file_name=file_name
                )

                Requests.send_log(
                    f"Pod {MABAR_POD_ID} uploaded video to S3 for WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id}.",
                    "item_uploaded",
                    f"Pod {MABAR_POD_ID} has uploaded video to S3 at {s3_uploader} for WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id}."
                )

                Requests.request(
                    "PUT",
                    f"{BASE_API_URL_MABAR}/wan_one_t2v/processing/{wan_t2v_id}/item/{wan_t2v_item_id}/uploaded",
                    data=json.dumps({"result_generate_url": s3_uploader}),
                    headers={"Content-Type": "application/json"}
                )

                success_generate_item += 1

            except Exception as e:
                print(f"Failed to upload to S3 for WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id}: {e}")
                Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, f"Gagal mengunggah ke S3: {e}")
                continue


        if success_generate_item > 0:
            Requests.send_log(
                f"Pod {MABAR_POD_ID} completed processing WAN T2V ID {wan_t2v_id}.",
                "task_completed",
                f"Pod {MABAR_POD_ID} has completed processing WAN T2V ID {wan_t2v_id} with {success_generate_item} successful items."
            )

            completed = Requests.request(
                "PUT",
                f"{BASE_API_URL_MABAR}/wan_one_t2v/processing/{wan_t2v_id}/completed"
            )

            if completed.status_code != 200:
                raise ValueError(f"Failed to mark WAN T2V ID {wan_t2v_id} as completed, status code: {completed.status_code}")

        else:
            failed = Requests.set_failed(wan_t2v_id, "Gagal menghasilkan video untuk semua item dalam tugas.")

            if failed.status_code != 200:
                raise ValueError(f"Failed to mark WAN T2V ID {wan_t2v_id} as failed, status code: {failed.status_code}")

    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
        Requests.stop_pod()
        continue