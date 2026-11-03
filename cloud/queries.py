from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from .models import EvidenceImage, EdgeEvent, ParkingSession, RFIDCard

VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def local_text(value: datetime | None) -> str:
    value = _aware(value)
    return value.astimezone(VIETNAM_TZ).strftime("%d/%m/%Y %H:%M:%S") if value else "—"


def utc_day_bounds(day: date | None = None) -> tuple[datetime, datetime]:
    local_day = day or datetime.now(VIETNAM_TZ).date()
    start = datetime.combine(local_day, time.min, tzinfo=VIETNAM_TZ)
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(
        timezone.utc
    )


def format_confidence(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def event_dict(row: EdgeEvent) -> dict:
    return {
        "time": local_text(row.event_timestamp),
        "received": local_text(row.received_at),
        "uid": row.uid or "—",
        "plate": row.plate or "—",
        "type": row.event_type,
        "ocr": format_confidence(row.ocr_confidence),
        "detection": format_confidence(row.detection_confidence),
        "reason": row.reason or "—",
    }


def dashboard_snapshot(factory: sessionmaker[Session]) -> dict:
    start, end = utc_day_bounds()
    with factory() as db:
        inside = (
            db.scalar(
                select(func.count())
                .select_from(ParkingSession)
                .where(
                    ParkingSession.entry_at.is_not(None),
                    ParkingSession.exit_at.is_(None),
                )
            )
            or 0
        )
        grouped = dict(
            db.execute(
                select(EdgeEvent.event_type, func.count())
                .where(
                    EdgeEvent.event_timestamp >= start, EdgeEvent.event_timestamp < end
                )
                .group_by(EdgeEvent.event_type)
            ).all()
        )
        recent = db.scalars(
            select(EdgeEvent).order_by(EdgeEvent.event_timestamp.desc()).limit(10)
        ).all()
        last_received = db.scalar(select(func.max(EdgeEvent.received_at)))
    return {
        "inside": inside,
        "entry": grouped.get("ENTRY", 0),
        "exit": grouped.get("EXIT", 0),
        "alerts": sum(grouped.get(kind, 0) for kind in ("REVIEW", "DENY", "ERROR")),
        "recent": [event_dict(row) for row in recent],
        "last_received": local_text(last_received),
    }


def history_page(
    factory: sessionmaker[Session],
    *,
    search: str = "",
    event_type: str = "ALL",
    day: date | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    conditions = []
    search = search.strip().upper()
    if search:
        like = f"%{search}%"
        conditions.append(or_(EdgeEvent.uid.ilike(like), EdgeEvent.plate.ilike(like)))
    if event_type != "ALL":
        conditions.append(EdgeEvent.event_type == event_type)
    if day:
        start, end = utc_day_bounds(day)
        conditions.extend(
            (EdgeEvent.event_timestamp >= start, EdgeEvent.event_timestamp < end)
        )
    where = and_(*conditions) if conditions else True
    with factory() as db:
        total = db.scalar(select(func.count()).select_from(EdgeEvent).where(where)) or 0
        rows = db.scalars(
            select(EdgeEvent)
            .where(where)
            .order_by(EdgeEvent.event_timestamp.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()
    return {"rows": [event_dict(row) for row in rows], "total": total}


def inside_sessions(factory: sessionmaker[Session]) -> list[dict]:
    now = datetime.now(timezone.utc)
    with factory() as db:
        rows = db.scalars(
            select(ParkingSession)
            .where(
                ParkingSession.entry_at.is_not(None), ParkingSession.exit_at.is_(None)
            )
            .order_by(ParkingSession.entry_at.desc())
        ).all()
    result = []
    for row in rows:
        entry = _aware(row.entry_at)
        seconds = max(0, int((now - entry).total_seconds())) if entry else 0
        hours, remainder = divmod(seconds, 3600)
        minutes = remainder // 60
        result.append(
            {
                "plate": row.plate or "—",
                "uid": row.uid or "—",
                "entry": local_text(entry),
                "duration": f"{hours} giờ {minutes} phút",
            }
        )
    return result


def rfid_cards(factory: sessionmaker[Session]) -> list[dict]:
    """Actual registered cards (not merely UIDs seen in events)."""
    with factory() as db:
        cards = db.scalars(select(RFIDCard).order_by(RFIDCard.device_id, RFIDCard.uid)).all()
        stats = db.execute(
            select(
                EdgeEvent.device_id, EdgeEvent.uid,
                func.count().label("event_count"),
                func.max(EdgeEvent.event_timestamp).label("last_seen"),
            ).where(EdgeEvent.uid.is_not(None))
            .group_by(EdgeEvent.device_id, EdgeEvent.uid)
        ).all()
    by_id = {(row.device_id, row.uid): row for row in stats}
    return [
        {
            "device_id": row.device_id,
            "uid": row.uid,
            "row_id": f"{row.device_id}:{row.uid}",
            "status": "Đang kích hoạt" if row.enabled else "Vô hiệu hóa",
            "created_at": local_text(row.created_at),
            "synced_at": local_text(row.synced_at),
            "last_seen": local_text(by_id[(row.device_id, row.uid)].last_seen)
            if (row.device_id, row.uid) in by_id else "—",
            "count": by_id[(row.device_id, row.uid)].event_count
            if (row.device_id, row.uid) in by_id else 0,
        }
        for row in cards
    ]


def device_status(factory: sessionmaker[Session], recent_minutes: int) -> dict:
    with factory() as db:
        row = db.execute(
            select(
                EdgeEvent.device_id,
                func.max(EdgeEvent.received_at).label("last_received"),
                func.count().label("event_count"),
            ).group_by(EdgeEvent.device_id)
        ).first()
        last_type = None
        if row:
            last_type = db.scalar(
                select(EdgeEvent.event_type)
                .where(EdgeEvent.device_id == row.device_id)
                .order_by(EdgeEvent.received_at.desc())
                .limit(1)
            )
    if not row:
        return {
            "device_id": "raspberrypi_01",
            "last_received": "—",
            "last_type": "—",
            "count": 0,
            "elapsed": "Chưa nhận sự kiện",
            "state": "Unknown",
        }
    last = _aware(row.last_received)
    delta = max(timedelta(0), datetime.now(timezone.utc) - last)
    seconds = int(delta.total_seconds())
    if seconds < 60:
        elapsed = f"{seconds} giây trước"
    elif seconds < 3600:
        elapsed = f"{seconds // 60} phút trước"
    else:
        elapsed = f"{seconds // 3600} giờ trước"
    return {
        "device_id": row.device_id,
        "last_received": local_text(last),
        "last_type": last_type or "—",
        "count": row.event_count,
        "elapsed": elapsed,
        "state": "Recently active"
        if delta <= timedelta(minutes=recent_minutes)
        else "No recent events",
    }


def plate_metadata(factory: sessionmaker[Session], limit: int = 100) -> list[dict]:
    with factory() as db:
        rows = db.scalars(
            select(EdgeEvent)
            .where(EdgeEvent.plate.is_not(None))
            .order_by(EdgeEvent.event_timestamp.desc())
            .limit(limit)
        ).all()
    with factory() as db:
        ids = [row.event_id for row in rows]
        available = set(db.execute(
            select(EvidenceImage.event_id, EvidenceImage.kind)
            .where(EvidenceImage.event_id.in_(ids))
        ).all()) if ids else set()
        present = {eid for eid, kind in available if kind == 'annotated'}
        present_plate = {eid for eid, kind in available if kind == 'plate'}
    return [
        {
            "plate": row.plate,
            "time": local_text(row.event_timestamp),
            "uid": row.uid or "—",
            "ocr": format_confidence(row.ocr_confidence),
            "detection": format_confidence(row.detection_confidence),
            "has_image": row.event_id in present,
            "image_url": f"/api/evidence/{row.event_id}/annotated.jpg" if row.event_id in present else "",
            "has_plate_image": row.event_id in present_plate,
            "plate_image_url": f"/api/evidence/{row.event_id}/plate.jpg" if row.event_id in present_plate else "",
            "image": "Xem ảnh" if row.event_id in present else "Chưa có ảnh",
        }
        for row in rows
    ]
