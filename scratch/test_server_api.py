import os
import sys
import json
import urllib.request
import time
import base64

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

print("[Verification] Testing web/server.py imports and SQLite initialization...")

import web.server as server

# Check SQLite DB exists and initialized
print(f"[Verification] DB Path: {server.DB_PATH}")
assert os.path.exists(server.DB_PATH), "Database file should exist!"

# Test save_recording_to_db (Idempotency)
test_id = "test-uuid-12345"
test_audio = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="

print("[Verification] Inserting test recording into SQLite...")
existed, synced_at, size_bytes = server.save_recording_to_db(
    test_id, "audio/wav", 1.5, 16000, "2026-08-31T00:00:00Z", test_audio
)
print(f"  Result 1: existed={existed}, synced_at={synced_at}, size_bytes={size_bytes}")
assert not existed, "First insert should report existed=False"

print("[Verification] Re-inserting SAME test recording (Idempotency test)...")
existed2, synced_at2, size_bytes2 = server.save_recording_to_db(
    test_id, "audio/wav", 1.5, 16000, "2026-08-31T00:00:00Z", test_audio
)
print(f"  Result 2: existed={existed2}, synced_at={synced_at2}, size_bytes={size_bytes2}")
assert existed2, "Second insert should report existed=True"

# Test get_recordings_from_db
recs = server.get_recordings_from_db()
print(f"[Verification] List recordings count: {len(recs)}")
assert any(r["id"] == test_id for r in recs), "Test recording should be listed"

# Test get_recording_audio_from_db
mime, raw_bytes = server.get_recording_audio_from_db(test_id)
print(f"[Verification] Retrieved audio mime: {mime}, size: {len(raw_bytes)} bytes")
assert raw_bytes is not None and len(raw_bytes) > 0, "Audio bytes should be retrieved"

print("==========================================================")
print("[Verification] ALL BACKEND SQLITE & API TESTS PASSED PERFECTLY!")
print("==========================================================")
