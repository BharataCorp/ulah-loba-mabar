from services.http_clients import Requests
import os
import sys
import time
import signal
import json
from typing import Optional

from wan_custom.config import BASE_API_URL_MABAR, MABAR_POD_ID, OUTPUT_DIR
from services.upload_to_s3 import S3Uploader
from wan_custom.pipelines.t2v_pipeline import T2VPipeline

# Configuration for polling and timeouts
POLL_INTERVAL_SECONDS = 5
MAX_IDLE_POLLS_BEFORE_STOP = 12  # stop after ~1 minute of no tasks
REQUEST_TIMEOUT_SECONDS = 20
CONSECUTIVE_ERRORS_BEFORE_STOP = 10  # stop pod after N consecutive network errors

shutdown_requested = False


def handle_sigterm(signum, frame):
    """Graceful shutdown on SIGTERM/SIGINT."""
    global shutdown_requested
    shutdown_requested = True
    try:
        Requests.send_log(
            f"Pod {MABAR_POD_ID} received SIGTERM, stopping.",
            "shutdown",
            f"Pod {MABAR_POD_ID} is shutting down on SIGTERM."
        )
    except Exception:
        pass
    try:
        Requests.stop_pod()
    except Exception:
        sys.exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

# Ensure OUTPUT_DIR exists for cleanup and generation
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception:
    # If we cannot create output dir, log and stop to avoid further issues
    try:
        Requests.send_log(
            f"Pod {MABAR_POD_ID} failed to create OUTPUT_DIR: {OUTPUT_DIR}",
            "startup_error",
            f"Could not create OUTPUT_DIR {OUTPUT_DIR}."
        )
    except Exception:
        sys.exit(1)


def safe_request(method: str, url: str, **kwargs):
    """Wrapper around Requests.request to ensure timeouts and consistent usage.
    This uses the project's Requests.request wrapper; we add a default timeout so
    network calls don't hang indefinitely.
    """
    kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
    return Requests.request(method=method, url=url, **kwargs)


def parse_storyboard(data_str: str) -> Optional[dict]:
    try:
        data = json.loads(data_str)
        if not isinstance(data, dict):
            return None
        scenes = data.get("scenes", [])
        if not isinstance(scenes, list) or len(scenes) == 0:
            return None
        return data
    except Exception:
        return None


def file_exists_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def stop_pod_and_exit(message: str):
    print(message)
    try:
        Requests.send_log(message, "shutdown", message)
        # Requests.stop_pod()
    except Exception:
        pass
    try:
        Requests.stop_pod()
    except Exception:
        sys.exit(0)


def _cleanup_local_files(final_output: str):
    try:
        # remove final output if present
        if final_output and os.path.exists(final_output):
            try:
                os.remove(final_output)
            except Exception:
                pass

        # clean up scene files created by T2V pipeline
        if os.path.isdir(OUTPUT_DIR):
            for fname in os.listdir(OUTPUT_DIR):
                if fname.startswith("t2v_scene_") and fname.endswith(".mp4"):
                    fpath = os.path.join(OUTPUT_DIR, fname)
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
            # remove concat list files
            for fname in os.listdir(OUTPUT_DIR):
                if fname.endswith(".txt") and fname.startswith("t2v_"):
                    fpath = os.path.join(OUTPUT_DIR, fname)
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
    except Exception:
        pass


def main():
    print(f"Pod {MABAR_POD_ID} starting WAN T2V worker.")

    Requests.send_log(
        f"Pod {MABAR_POD_ID} is starting WAN T2V worker.",
        "startup",
        f"Pod {MABAR_POD_ID} has started and is ready to process WAN T2V tasks."
    )

    if not BASE_API_URL_MABAR:
        stop_pod_and_exit("BASE_API_URL_MABAR not set, stopping pod to avoid billing.")

    idle_polls = 0
    consecutive_errors = 0

    while not shutdown_requested:
        try:
            Requests.send_log(
                f"Pod {MABAR_POD_ID} checking for WAN T2V tasks.",
                "task_check",
                f"Pod {MABAR_POD_ID} is checking for WAN T2V tasks to process."
            )

            resp = safe_request(
                method="GET",
                url=f"{BASE_API_URL_MABAR}/runpod_pod/{MABAR_POD_ID}/wan_t2v/ready_to_process"
            )

            print(f"Pod {MABAR_POD_ID} received response with status code: {getattr(resp, 'status_code', None)}")

            # print for json response if available
            # avoid printing full response JSON (may contain sensitive credentials)
            try:
                _ = resp.json()
                print(f"Pod {MABAR_POD_ID} received a JSON response (redacted).")
            except Exception:
                pass

            # If we got here without exception, reset consecutive error counter
            consecutive_errors = 0

            # Interpret status codes:
            # 200 -> task payload
            # 204/404 -> no task available
            # 410 -> server requests pod to stop
            if resp is None:
                # If wrapper returns None for some reason, backoff
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            status = getattr(resp, 'status_code', None)
            if status == 410:
                stop_pod_and_exit("Server requested pod stop (410). Stopping to avoid billing.")

            if status in (204, 404):
                idle_polls += 1
                if idle_polls >= MAX_IDLE_POLLS_BEFORE_STOP:
                    stop_pod_and_exit("No tasks for a while. Stopping pod to avoid idle billing.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            if status != 200:
                # transient issue or unexpected response; log and retry
                Requests.send_log(
                    f"Unexpected status fetching task: {status}",
                    "task_error",
                    f"Received status {status} when checking for tasks. Retrying after sleep."
                )
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Reset idle counter when a task is found
            idle_polls = 0

            try:
                task = resp.json()
            except Exception as e:
                Requests.send_log(
                    f"Failed to parse task JSON: {e}",
                    "task_error",
                    f"Failed to parse task JSON: {e}"
                )
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            wan_t2v = task.get("wan_t2v")
            if wan_t2v is None:
                # nothing to do; continue polling
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            wan_t2v_id = wan_t2v.get("id")
            product_reference_name = wan_t2v.get("product_reference_name", "")
            product_prompt_description = wan_t2v.get("product_prompt_description", "")
            characters = wan_t2v.get("characters", [])
            items = wan_t2v.get("items") or []

            if not items:
                Requests.set_failed(wan_t2v_id, "Tidak ada item untuk diproses dalam tugas.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            Requests.send_log(
                f"Pod {MABAR_POD_ID} started processing WAN T2V ID {wan_t2v_id} with {len(items)} items.",
                "task_start",
                f"Pod {MABAR_POD_ID} is processing WAN T2V ID {wan_t2v_id} which has {len(items)} items."
            )

            # check if every item.status in items is FAILED
            if all(item.get("status") == "FAILED" for item in items):
                Requests.set_failed(wan_t2v_id, "Semua item dalam tugas memiliki status GAGAL.")
                continue

            if all(item.get("status") == "UPLOADED" for item in items):
                # mark completed
                safe_request(
                    "PUT",
                    f"{BASE_API_URL_MABAR}/wan_one_t2v/processing/{wan_t2v_id}/completed"
                )
                continue

            success_generate_item = 0

            upload_s3_hostname = wan_t2v.get("upload_s3_hostname")

            # Validate S3 params once
            upload_s3_bucket = wan_t2v.get("upload_s3_bucket")
            # Build endpoint safely: prefer explicit endpoint, otherwise use hostname if present
            if wan_t2v.get("upload_s3_endpoint"):
                upload_s3_endpoint = wan_t2v.get("upload_s3_endpoint")
            elif upload_s3_hostname:
                upload_s3_endpoint = "https://" + str(upload_s3_hostname)
            else:
                upload_s3_endpoint = None
            upload_s3_access_key = wan_t2v.get("upload_s3_access_key")
            upload_s3_secret_key = wan_t2v.get("upload_s3_secret_key")

            if not all([upload_s3_bucket, upload_s3_endpoint, upload_s3_access_key, upload_s3_secret_key]):
                Requests.set_failed(wan_t2v_id, "Konfigurasi S3 tidak lengkap.")
                continue

            for item in items:
                if shutdown_requested:
                    break

                wan_t2v_item_id = item.get("id")
                order_index = item.get("order_index")
                storyboard_raw = item.get("storyboard_data")

                # Normalize item duration safely (handle None or string values)
                raw_item_duration = item.get("duration")
                if raw_item_duration is None:
                    target_duration = 10
                else:
                    try:
                        target_duration = int(raw_item_duration)
                    except Exception:
                        target_duration = 10

                if not storyboard_raw:
                    Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "storyboard data kosong.")
                    continue

                storyboard_data = parse_storyboard(storyboard_raw)
                if storyboard_data is None:
                    Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "storyboard data tidak valid atau tidak memiliki scenes.")
                    continue

                scenes = storyboard_data.get("scenes", [])
                if not scenes:
                    Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "Tidak ada scene prompt dalam storyboard data.")
                    continue

                # Normalize each scene duration to int (handle strings like "5" and None)
                for s_idx, sc in enumerate(scenes):
                    raw_d = sc.get("duration", 5)
                    try:
                        d_int = int(raw_d)
                    except Exception:
                        # fallback to 5 seconds if conversion fails
                        d_int = 5
                    # enforce minimum 1 second
                    if d_int <= 0:
                        d_int = 1
                    sc["duration"] = d_int

                Requests.send_log(
                    f"Pod {MABAR_POD_ID} processing WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id} (order {order_index}).",
                    "item_start",
                    f"Pod {MABAR_POD_ID} is processing WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id} with order index {order_index}."
                )

                size = "480*832"
                output_path = os.path.join(OUTPUT_DIR, f"wan_t2v_{wan_t2v_id}_item_{wan_t2v_item_id}.mp4")

                try:
                    output = T2VPipeline.generate(
                        scenes=scenes,
                        target_duration=target_duration,
                        size=size,
                        output_path=output_path,
                        product_reference_name=product_reference_name,
                        product_prompt_description=product_prompt_description,
                        characters=characters
                    )
                except Exception as gen_err:
                    Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, f"Gagal generate: {gen_err}")
                    continue

                final_output = output if isinstance(output, str) else output_path

                if not file_exists_nonempty(final_output):
                    Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, "File output tidak ditemukan atau kosong.")
                    continue

                Requests.send_log(
                    f"Pod {MABAR_POD_ID} finished generating video for WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id}.",
                    "item_generated",
                    f"Pod {MABAR_POD_ID} has generated video at {final_output} for WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id}."
                )

                try:
                    unixtime_stamp = int(os.path.getmtime(final_output))
                except Exception:
                    unixtime_stamp = int(time.time())

                s3_base_path = "WAN_T2V/videos"
                file_name = f"wan_t2v_{wan_t2v_id}_item_{wan_t2v_item_id}_{unixtime_stamp}.mp4"

                try:
                    s3_url = S3Uploader.upload_file_to_s3(
                        file_path=final_output,
                        upload_s3_endpoint=upload_s3_endpoint,
                        upload_s3_access_key=upload_s3_access_key,
                        upload_s3_secret_key=upload_s3_secret_key,
                        upload_s3_bucket=upload_s3_bucket,
                        s3_base_path=s3_base_path,
                        file_name=file_name
                    )
                except Exception as e:
                    Requests.set_item_failed(wan_t2v_id, wan_t2v_item_id, f"Gagal mengunggah ke S3: {e}")
                    continue

                Requests.send_log(
                    f"Pod {MABAR_POD_ID} uploaded video to S3 for WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id}.",
                    "item_uploaded",
                    f"Pod {MABAR_POD_ID} has uploaded video to S3 at {s3_url} for WAN T2V ID {wan_t2v_id} item {wan_t2v_item_id}."
                )

                safe_request(
                    "PUT",
                    f"{BASE_API_URL_MABAR}/wan_one_t2v/processing/{wan_t2v_id}/item/{wan_t2v_item_id}/uploaded",
                    data=json.dumps({"result_generate_url": s3_url}),
                    headers={"Content-Type": "application/json"}
                )

                # cleanup local files for this item to avoid disk accumulation
                _cleanup_local_files(final_output)

                success_generate_item += 1

            # After processing all items for this task
            if success_generate_item > 0:
                Requests.send_log(
                    f"Pod {MABAR_POD_ID} completed processing WAN T2V ID {wan_t2v_id}.",
                    "task_completed",
                    f"Pod {MABAR_POD_ID} has completed processing WAN T2V ID {wan_t2v_id} with {success_generate_item} successful items."
                )

                completed = safe_request(
                    "PUT",
                    f"{BASE_API_URL_MABAR}/wan_one_t2v/processing/{wan_t2v_id}/completed"
                )
                status_completed = getattr(completed, 'status_code', None)
                if status_completed != 200:
                    Requests.send_log(
                        f"Failed to mark completed for {wan_t2v_id}, status: {status_completed}",
                        "task_error",
                        f"Failed to mark completed for {wan_t2v_id}, status: {status_completed}"
                    )
            else:
                failed = Requests.set_failed(wan_t2v_id, "Gagal menghasilkan video untuk semua item dalam tugas.")
                status_failed = getattr(failed, 'status_code', None)
                if status_failed not in (None, 200):
                    Requests.send_log(
                        f"Failed to mark failed for {wan_t2v_id}, status: {status_failed}",
                        "task_error",
                        f"Failed to mark failed for {wan_t2v_id}, status: {status_failed}"
                    )

            # Sleep briefly before checking next task
            time.sleep(POLL_INTERVAL_SECONDS)

        except Exception as e:
            # Unexpected error handling: log and backoff. Avoid immediate stop to be resilient.
            try:
                Requests.send_log(
                    f"Pod {MABAR_POD_ID} encountered unexpected error: {e}",
                    "error",
                    f"Pod {MABAR_POD_ID} unexpected error: {e}"
                )
            except Exception:
                pass

            # Increment consecutive error counter; if too many, stop pod to avoid billing
            consecutive_errors += 1
            if consecutive_errors >= CONSECUTIVE_ERRORS_BEFORE_STOP:
                stop_pod_and_exit(f"Too many consecutive errors ({consecutive_errors}), stopping pod to avoid billing.")

            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
