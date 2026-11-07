from cloud.live import LiveFrameStore

def test_progress_received_then_done():
    store=LiveFrameStore()
    store.publish_progress({"phase":"PROCESSING","uid":"ABC","action":"","plate":""},"pi")
    assert store.progress_status()["phase"]=="PROCESSING"
    store.publish_progress({"phase":"DONE","uid":"ABC","action":"ENTRY","plate":"29A12345"},"pi")
    assert store.progress_status()["action"]=="ENTRY"
    assert store.progress_status()["device_id"]=="pi"

def test_progress_expires():
    store=LiveFrameStore()
    store.publish_progress({"phase":"PROCESSING","uid":"ABC","action":"","plate":""},"pi")
    store._rfid_progress_at -= 21
    assert store.progress_status() is None
