import json
import time

import requests
import os
from wan_custom.config import BASE_API_URL_MABAR, MABAR_POD_ID, KEY_MANAGEMENT_ID

# create function with param method, url, headers=None, data=None:
class Requests:
    @staticmethod
    def request(method, url, headers=None, data=None):
        try:
            if headers is None:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }

            response = requests.request(method, url, headers=headers, data=data)
            response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
            return response

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            # throw error
            raise ValueError(f"Request failed: {e}")


    @staticmethod
    def send_callback(callback_url, payload, method="POST"):
        if not callback_url:
            return

        # if end with /, remove it
        if callback_url.endswith("/"):
            callback_url = callback_url[:-1]

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            response = Requests.request(method, callback_url, headers=headers, data=payload)
            print(f"Callback sent to {callback_url}, response status: {response.status_code}")
        except ValueError as e:
            print(f"Failed to send callback to {callback_url}: {e}")
        except Exception as e:
            print(f"Unexpected error sending callback to {callback_url}: {e}")

    @staticmethod
    def stop_pod(retry_count=3):
        # if not BASE_URL_MABAR Force PID 1 EXIT FOR STOP THE POD
        if not BASE_API_URL_MABAR:
            print("BASE_URL_MABAR not set, exiting pod for stop.")
            os._exit(0)

        if not MABAR_POD_ID or not KEY_MANAGEMENT_ID:
            print("MABAR_POD_ID or KEY_MANAGEMENT_ID not set, cannot stop pod via API.")
            os._exit(0)

        try:
            response = Requests.request(
                method="POST",
                url=f"{BASE_API_URL_MABAR}/runpod_pod/stop/{MABAR_POD_ID}/key_management/{KEY_MANAGEMENT_ID}",
            )
            if response.status_code == 200:
                print("Pod stop request sent successfully, exiting pod.")
            else:
            # retry after 5 seconds
                if retry_count > 0:
                    print(f"Pod stop request failed with status code {response.status_code}, retrying...")
                    time.sleep(5)
                    Requests.stop_pod(retry_count - 1)
                else:
                    print(f"Pod stop request failed after retries, exiting pod.")
                    os._exit(0)
        except Exception as e:
            print(f"Error occurred while sending pod stop request: {e}, exiting pod.")
            os._exit(0)


    @staticmethod
    def terminate_pod(retry_count=3):
        if not BASE_API_URL_MABAR:
            print("BASE_URL_MABAR not set, cannot delete pod via API.")
            return

        try:
            response = Requests.request(
                method="POST",
                url=f"{BASE_API_URL_MABAR}/runpod_pod/terminate/{MABAR_POD_ID}/key_management/{KEY_MANAGEMENT_ID}",
            )
            if response.status_code == 200:
                print("Pod delete request sent successfully.")
            else:
                # retry after 5 seconds
                if retry_count > 0:
                    print(f"Pod delete request failed with status code {response.status_code}, retrying...")
                    time.sleep(5)
                    Requests.delete_pod(retry_count - 1)
                else:
                    print(f"Pod delete request failed after retries.")
                    raise ValueError(f"Pod delete request failed with status code {response.status_code}")
        except Exception as e:
            print(f"Error occurred while sending pod delete request: {e}")
            os._exit(0)

    @staticmethod
    def send_log(
        title: str,
        step: str,
        message: str,
        data: dict = None,
    ):
        if not BASE_API_URL_MABAR:
            print("BASE_URL_MABAR not set, cannot send log via API.")
            return

        payload = {
            "title": title,
            "step": step,
            "message": message,
            "data": data or {},
            "pod_id": MABAR_POD_ID,
            "key_management_id": KEY_MANAGEMENT_ID,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            response = Requests.request(
                method="POST",
                url=f"{BASE_API_URL_MABAR}/runpod_log/{MABAR_POD_ID}",
                headers=headers,
                data=json.dumps(payload),
            )
            if response.status_code == 200:
                print("Log sent successfully.")
            else:
                print(f"Failed to send log, status code: {response.status_code}")
                raise ValueError(f"Failed to send log, status code: {response.status_code}")
        except Exception as e:
            print(f"Error occurred while sending log: {e}")
            os._exit(0)

    @staticmethod
    def set_failed(wan_t2v_id, failed_reason: str):
        if not BASE_API_URL_MABAR:
            print("BASE_URL_MABAR not set, cannot set failed status via API.")
            return

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = json.dumps({"failed_reason": failed_reason})

        try:
            response = Requests.request(
                method="POST",
                url=f"{BASE_API_URL_MABAR}/wan_one_t2v/processing/{wan_t2v_id}/processing_failed",
                headers=headers,
                data=data,
            )

            if response.status_code == 200:
                print(f"WAN T2V ID {wan_t2v_id} set to failed successfully.")
            else:
                print(f"Failed to set WAN T2V ID {wan_t2v_id} to failed, status code: {response.status_code}")
                raise ValueError(f"Failed to set WAN T2V ID {wan_t2v_id} to failed, status code: {response.status_code}")
        except Exception as e:
            print(f"Error occurred while setting WAN T2V ID {wan_t2v_id} to failed: {e}")
            os._exit(0)

    @staticmethod
    def set_item_failed(wan_t2v_id, wan_t2v_item_id, failed_reason: str):
        if not BASE_API_URL_MABAR:
            print("BASE_URL_MABAR not set, cannot set item failed status via API.")
            return

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = json.dumps({"failed_reason": failed_reason})

        try:
            response = Requests.request(
                method="POST",
                url=f"{BASE_API_URL_MABAR}/wan_one_t2v/processing/{wan_t2v_id}/item/{wan_t2v_item_id}/processing_failed",
                headers=headers,
                data=data,
            )

            if response.status_code == 200:
                print(f"WAN T2V Item ID {wan_t2v_item_id} set to failed successfully.")
            else:
                print(f"Failed to set WAN T2V Item ID {wan_t2v_item_id} to failed, status code: {response.status_code}")
                raise ValueError(f"Failed to set WAN T2V Item ID {wan_t2v_item_id} to failed, status code: {response.status_code}")
        except Exception as e:
            print(f"Error occurred while setting WAN T2V Item ID {wan_t2v_item_id} to failed: {e}")
            os._exit(0)


