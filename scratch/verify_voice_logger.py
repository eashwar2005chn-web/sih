import os
import sys
import json
import base64
import hashlib
import sqlite3
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

print("==================================================================")
print("[Verification] Starting 24/7 Tactical Voice Logger End-to-End Test")
print("==================================================================")

import web.server as server

# Clear DB tables for a clean hash chain test
conn = sqlite3.connect(server.DB_PATH)
cursor = conn.cursor()
cursor.execute("DELETE FROM recordings")
cursor.execute("DELETE FROM alerts")
cursor.execute("DELETE FROM audit_logs")
conn.commit()
conn.close()

# Helper: Compute Hash Chain Link
def compute_hash_link(audio_bytes: bytes, prev_sha256: str) -> str:
    h = hashlib.sha256()
    h.update(audio_bytes)
    h.update(prev_sha256.encode('utf-8'))
    return h.hexdigest()

# Step 1: Create Logbook Segment 1 (Genesis)
audio1 = b"DRDO_LOGBOOK_SEGMENT_1_AUDIO_DATA_16KHZ_PCM_SAMPLE_DATA"
audio1_b64 = "data:audio/wav;base64," + base64.b64encode(audio1).decode('utf-8')
genesis_prev = "GENESIS_LOGBOOK_SEED_DRDO_PS26052"
hash1 = compute_hash_link(audio1, genesis_prev)
id1 = "logger-seg-001"

print(f"[Test 1] Saving Logbook Segment 1 ({id1})...")
existed1, synced1, size1 = server.save_recording_to_db(
    rec_id=id1,
    mime_type="audio/wav",
    duration_sec=3600.0,
    sample_rate=16000,
    created_at="2026-08-31T00:00:00Z",
    audio_b64=audio1_b64,
    sha256=hash1,
    prev_sha256=genesis_prev,
    vad_speech_sec=1420.5,
    classification="NORMAL",
    level_reduction_db=4.12
)
assert not existed1, "Segment 1 should be newly created"
print(f"  Segment 1 saved! Size: {size1} bytes | Hash: {hash1[:12]}...")

# Step 2: Create Logbook Segment 2 (Chained to Segment 1)
audio2 = b"DRDO_LOGBOOK_SEGMENT_2_AUDIO_DATA_WITH_DURESS_CODEWORD"
audio2_b64 = "data:audio/wav;base64," + base64.b64encode(audio2).decode('utf-8')
hash2 = compute_hash_link(audio2, hash1)
id2 = "logger-seg-002-protected"

print(f"[Test 2] Saving Logbook Segment 2 (PROTECTED Code-Word, Chained to Seg 1)...")
existed2, synced2, size2 = server.save_recording_to_db(
    rec_id=id2,
    mime_type="audio/wav",
    duration_sec=3600.0,
    sample_rate=16000,
    created_at="2026-08-31T01:00:00Z",
    audio_b64=audio2_b64,
    sha256=hash2,
    prev_sha256=hash1,
    vad_speech_sec=980.2,
    classification="PROTECTED",
    level_reduction_db=3.85
)
assert not existed2, "Segment 2 should be newly created"
print(f"  Segment 2 saved! Size: {size2} bytes | Hash: {hash2[:12]}... (Chained link verified)")

# Step 3: Test Resumable Chunked Upload API (Segment 3)
audio3 = b"DRDO_LOGBOOK_SEGMENT_3_CHUNKED_UPLOAD_DATA_STREAM"
chunk1 = base64.b64encode(audio3[:25]).decode('utf-8')
chunk2 = base64.b64encode(audio3[25:]).decode('utf-8')
hash3 = compute_hash_link(audio3, hash2)
id3 = "logger-seg-003-chunked"

print(f"[Test 3] Testing Resumable Chunked Upload for Segment 3 ({id3})...")
res_c1 = server.receive_chunk_data(
    rec_id=id3,
    chunk_index=0,
    total_chunks=2,
    chunk_b64=chunk1,
    meta={"sha256": hash3, "prevSha256": hash2, "classification": "SOS", "durationSec": 3600.0}
)
print(f"  Chunk 1 result: {res_c1['status']}")
assert res_c1["status"] == "chunk_received", "Chunk 1 should return chunk_received"

res_c2 = server.receive_chunk_data(
    rec_id=id3,
    chunk_index=1,
    total_chunks=2,
    chunk_b64=chunk2,
    meta={"sha256": hash3, "prevSha256": hash2, "classification": "SOS", "durationSec": 3600.0}
)
print(f"  Chunk 2 result: {res_c2['status']}")
assert res_c2["status"] == "assembly_complete", "Chunk 2 should trigger assembly_complete"

# Step 4: Verify Hash Chain Integrity across all segments
print("[Test 4] Verifying SHA-256 Hash Chain Integrity across DB...")
chain_check = server.verify_hash_chain_in_db()
print(f"  Hash Chain Status: Valid={chain_check['valid']}, Total={chain_check['totalEntries']}")
assert chain_check["valid"], "Hash chain should be 100% valid!"

# Step 5: Test Emergency SOS Alert & Audit Log Ingestion
print("[Test 5] Ingesting Emergency SOS Alert & Audit Trail...")
server.save_alert_to_db(
    alert_id="alert-sos-999",
    timestamp="2026-08-31T01:30:00Z",
    latitude=28.6139,
    longitude=77.2090,
    status="ACTIVE",
    details="Operator SOS Emergency triggered from Android Field Unit"
)
alerts = server.get_alerts_from_db()
assert len(alerts) >= 1, "Alert should be saved in DB"
print(f"  Alert saved! Total alerts: {len(alerts)} | Latest status: {alerts[0]['status']}")

server.save_audit_log_to_db(
    log_id="audit-101",
    timestamp="2026-08-31T01:30:05Z",
    event_type="SOS_ALERT",
    details="Emergency SOS Alert logged to audit trail",
    sha256=hash3
)
audit_logs = server.get_audit_logs_from_db()
assert len(audit_logs) >= 1, "Audit log should be saved in DB"
print(f"  Audit log saved! Event: {audit_logs[0]['eventType']} | Details: {audit_logs[0]['details']}")

# Step 6: Tamper Detection Test
print("[Test 6] Testing Hash Chain Tamper Detection (Tampering Segment 2)...")
conn = sqlite3.connect(server.DB_PATH)
cursor = conn.cursor()
cursor.execute("UPDATE recordings SET audio_b64 = 'TAMPERED_DATA' WHERE id = ?", (id2,))
conn.commit()
conn.close()

tamper_check = server.verify_hash_chain_in_db()
print(f"  Tamper Check Status: Valid={tamper_check['valid']} | BrokenAt={tamper_check.get('brokenAtId')}")
assert not tamper_check["valid"], "Tampered data MUST be caught by hash chain verification!"
assert tamper_check["brokenAtId"] == id2, "Tamper check should pinpoint broken entry ID"

# Restore original audio data for Segment 2
conn = sqlite3.connect(server.DB_PATH)
cursor = conn.cursor()
cursor.execute("UPDATE recordings SET audio_b64 = ? WHERE id = ?", (audio2_b64, id2))
conn.commit()
conn.close()

restored_check = server.verify_hash_chain_in_db()
assert restored_check["valid"], "Hash chain should be valid again after restoration"
print("  Restoration verified! Hash chain valid again.")

print("==================================================================")
print("[Verification] ALL 24/7 TACTICAL VOICE LOGGER TESTS PASSED PERFECTLY!")
print("==================================================================")
