from edge.live_uplink import LiveFrameUplink
from types import SimpleNamespace

def test_processing_before_result():
    # No network/camera access required for UI-state methods.
    o=object.__new__(LiveFrameUplink)
    import threading
    o._lock=threading.Lock()
    o._rfid_status=None
    o._rfid_at=0.0
    o._rfid_version=0
    o._last_event=""
    o._event_at=0.0
    o.note_processing("B387E238")
    assert o._rfid_status["phase"]=="PROCESSING"
    assert o._rfid_status["uid"]=="B387E238"
    o.note_event(SimpleNamespace(action="ENTRY",plate="29A12345",uid="B387E238"))
    assert o._rfid_status["phase"]=="DONE"
    assert o._rfid_status["action"]=="ENTRY"

def test_failure_transition():
    o=object.__new__(LiveFrameUplink)
    import threading
    o._lock=threading.Lock()
    o._rfid_status=None
    o._rfid_at=0.0
    o._rfid_version=0
    o.note_processing("B387E238")
    o.note_failed()
    assert o._rfid_status["phase"]=="ERROR"
