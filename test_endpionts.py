import argparse
import time
import requests

import os
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

def test_stream_and_poll(image_path: str):
    print(f"\n=== Loading image: {image_path} ===")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # --- Test 1: missing image (expect 400) ---
    print("\n[Test 1] POST /api/stream with no image → expect 400")
    resp = requests.post(f"{BASE_URL}/api/stream", data={
        "user_id": "u1", "stream_id": "s1", "frame_id": "f1", "timestamp": 1000
    })
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    print("  PASSED")

    # --- Test 2: valid frame submission (expect 200 + task_id) ---
    print("\n[Test 2] POST /api/stream with valid image → expect 200 + task_id")
    resp = requests.post(f"{BASE_URL}/api/stream",
        data={
            "user_id": "u1",
            "stream_id": "s1",
            "frame_id": "f1",
            "timestamp": int(time.time()),
        },
        files={"image": ("test_face.jpg", image_bytes, "image/jpeg")},  # ← key change
    )
    print(f"  Status: {resp.status_code}")
    print(f"  Body:   {resp.json()}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    task_id = resp.json()["task_id"]
    assert task_id, "task_id is empty"
    print(f"  PASSED — task_id: {task_id}")

    # --- Test 3: poll result while pending (expect 202) ---
    print(f"\n[Test 3] GET /api/result/{task_id} immediately → expect 202")
    resp = requests.get(f"{BASE_URL}/api/result/{task_id}")
    print(f"  Status: {resp.status_code} — {resp.json()}")
    assert resp.status_code == 202, f"Expected 202, got {resp.status_code}"
    print("  PASSED")

    # --- Test 4: poll until done (timeout 30s) ---
    print(f"\n[Test 4] Polling until result is ready (timeout 30s)...")
    for attempt in range(15):
        time.sleep(2)
        resp = requests.get(f"{BASE_URL}/api/result/{task_id}")
        print(f"  Attempt {attempt+1}: HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Result: {data}")
            assert "items" in data, "Response missing 'items'"
            print("  PASSED")
            break
        elif resp.status_code == 202:
            continue
        else:
            print(f"  FAILED — unexpected status: {resp.status_code}: {resp.text}")
            break
    else:
        print("  TIMEOUT — task did not complete in 30s")

    # --- Test 5: non-existent task_id (expect 202 or 500) ---
    print(f"\n[Test 5] GET /api/result/fake-id → expect 202 (pending) or 500")
    resp = requests.get(f"{BASE_URL}/api/result/fake-task-id-00000")
    print(f"  Status: {resp.status_code} — {resp.json()}")
    assert resp.status_code in (202, 500), f"Unexpected: {resp.status_code}"
    print("  PASSED")

    print("\n=== All tests done ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a JPEG/PNG image file")
    args = parser.parse_args()
    test_stream_and_poll(args.image)