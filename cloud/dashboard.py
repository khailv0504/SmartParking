from __future__ import annotations

import html
import logging
from datetime import date

from fastapi import FastAPI, Request
from nicegui import run, ui
from sqlalchemy.orm import Session, sessionmaker

from .auth import CSRF_COOKIE, resolve_admin_session
from .config import Settings
from .queries import (
    dashboard_snapshot,
    device_status,
    history_page,
    inside_sessions,
    plate_metadata,
    rfid_cards,
)

LOG = logging.getLogger(__name__)

EVENT_COLUMNS = [
    {"name": "time", "label": "Thời gian", "field": "time", "sortable": True},
    {"name": "uid", "label": "UID", "field": "uid"},
    {"name": "plate", "label": "Biển số", "field": "plate"},
    {"name": "type", "label": "Loại", "field": "type"},
    {"name": "ocr", "label": "OCR", "field": "ocr"},
    {"name": "detection", "label": "Detection", "field": "detection"},
    {"name": "reason", "label": "Lý do", "field": "reason"},
]


def _error_label():
    label = ui.label().classes("text-red-700 bg-red-50 rounded-lg px-4 py-2 w-full")
    label.set_visibility(False)
    return label


def _loading_label():
    return ui.label("Đang tải dữ liệu…").classes("text-blue-700 text-sm py-2")


def _empty_label(text: str):
    label = ui.label(text).classes("text-slate-500 py-6 text-center w-full")
    label.set_visibility(False)
    return label


def _shell(title: str, request: Request) -> None:
    csrf = html.escape(request.cookies.get(CSRF_COOKIE, ""), quote=True)
    with ui.header().classes(
        "bg-white text-slate-900 border-b border-slate-200 shadow-none"
    ):
        ui.button(icon="menu").props("flat round").on("click", lambda: drawer.toggle())
        ui.label(title).classes("text-lg font-semibold")
        ui.space()
        ui.html(
            f'<form method="post" action="/auth/logout">'
            f'<input type="hidden" name="csrf_token" value="{csrf}">'
            '<button class="text-sm px-3 py-2 rounded-lg hover:bg-slate-100" '
            'type="submit">'
            "Đăng xuất</button></form>"
        )
    with ui.left_drawer(value=True).classes("bg-slate-950 text-white") as drawer:
        ui.label("SMARTPARKING").classes("text-xl font-bold tracking-wide px-3 py-5")
        links = [
            ("/", "dashboard", "Tổng quan"),
            ("/history", "history", "Lịch sử vào/ra"),
            ("/inside", "directions_car", "Xe trong bãi"),
            ("/rfid", "contactless", "Thẻ RFID"),
            ("/device", "memory", "Raspberry Pi"),
            ("/plates", "photo_camera", "Ảnh biển số"),
        ]
        for target, icon, label in links:
            with ui.link(target=target).classes(
                "flex items-center gap-3 text-white no-underline px-3 py-3 "
                "rounded-lg hover:bg-slate-800"
            ):
                ui.icon(icon)
                ui.label(label)
    ui.label(title).classes("text-2xl font-bold text-slate-900 mt-3")


def _table(columns: list[dict], rows: list[dict] | None = None):
    return (
        ui.table(columns=columns, rows=rows or [], row_key=columns[0]["name"])
        .classes("w-full bg-white rounded-xl")
        .props("flat bordered wrap-cells")
    )


async def _load(function, *args, **kwargs):
    # SQLAlchemy sync chạy trong worker thread để không block NiceGUI event loop.
    return await run.io_bound(function, *args, **kwargs)


async def _session_is_valid(request: Request, factory: sessionmaker[Session]) -> bool:
    """Kiểm tra lại session cho kết nối NiceGUI sống lâu.

    Cookie hết hạn không tự đóng WebSocket đang mở. Việc kiểm tra định kỳ làm cho
    logout ở tab khác hoặc session hết hạn có hiệu lực trong tối đa một chu kỳ 5 giây.
    """
    return await _load(resolve_admin_session, request, factory) is not None


def register_dashboard(
    fastapi_app: FastAPI, settings: Settings, factory: sessionmaker[Session]
) -> None:
    ui.add_css(
        """
        body { background: #f8fafc; color: #0f172a; }
        .q-page { padding: 1rem; max-width: 1500px; margin: 0 auto; }
        .metric-card { min-width: 180px; flex: 1 1 210px; }
        @media (max-width: 700px) { .q-page { padding: .65rem; } }
        """
    )

    @ui.page("/login")
    def login_page(request: Request):
        if resolve_admin_session(request, factory):
            ui.navigate.to("/")
            return
        with ui.card().classes(
            "absolute-center w-[min(92vw,420px)] p-8 rounded-2xl shadow-lg"
        ):
            ui.label("SmartParking Admin").classes("text-2xl font-bold")
            ui.label("Đăng nhập để xem dữ liệu bãi xe").classes("text-slate-500 mb-3")
            if request.query_params.get("error"):
                ui.label("Tên đăng nhập hoặc mật khẩu không đúng.").classes(
                    "text-red-700 bg-red-50 rounded-lg p-3 w-full"
                )
            ui.html(
                """
                <form method="post" action="/auth/login"
                  class="column q-gutter-md full-width">
                  <label class="text-sm text-slate-600">Tên đăng nhập</label>
                  <input name="username" required maxlength="80" autocomplete="username"
                    class="border rounded-lg px-3 py-3" />
                  <label class="text-sm text-slate-600">Mật khẩu</label>
                  <input name="password" type="password" required
                    autocomplete="current-password"
                    class="border rounded-lg px-3 py-3" />
                  <button type="submit"
                    class="bg-blue-700 text-white rounded-lg px-4 py-3 font-medium">
                    Đăng nhập
                  </button>
                </form>
                """
            ).classes("w-full")

    @ui.page("/")
    async def dashboard_page(request: Request):
        _shell("Tổng quan", request)
        loading = _loading_label()
        error = _error_label()
        with ui.row().classes("w-full gap-4 flex-wrap"):
            cards = {}
            for key, label, icon in (
                ("inside", "Xe đang trong bãi", "local_parking"),
                ("entry", "ENTRY hôm nay", "login"),
                ("exit", "EXIT hôm nay", "logout"),
                ("alerts", "REVIEW / DENY / ERROR", "warning"),
            ):
                with ui.card().classes("metric-card rounded-xl shadow-sm"):
                    with ui.row().classes("items-center justify-between w-full"):
                        ui.label(label).classes("text-slate-500")
                        ui.icon(icon).classes("text-blue-700")
                    cards[key] = ui.label("—").classes("text-3xl font-bold")
        last = ui.label("Dữ liệu gần nhất: —").classes("text-sm text-slate-500")
        ui.label("Sự kiện gần nhất").classes("text-lg font-semibold mt-4")
        table = _table(EVENT_COLUMNS)
        empty = _empty_label("Chưa có sự kiện nào.")

        async def refresh():
            loading.set_visibility(True)
            try:
                if not await _session_is_valid(request, factory):
                    ui.navigate.to("/login")
                    return
                data = await _load(dashboard_snapshot, factory)
                for key, label in cards.items():
                    label.set_text(str(data[key]))
                last.set_text(f"Dữ liệu gần nhất: {data['last_received']}")
                table.rows = data["recent"]
                table.update()
                empty.set_visibility(not data["recent"])
                error.set_visibility(False)
            except Exception:
                LOG.exception("Dashboard refresh failed")
                error.set_text(
                    "Tạm thời không thể đọc cơ sở dữ liệu. "
                    "Hệ thống sẽ thử lại sau 5 giây."
                )
                error.set_visibility(True)
            finally:
                loading.set_visibility(False)

        await refresh()
        ui.timer(5.0, refresh)

    @ui.page("/history")
    async def history_page_ui(request: Request):
        _shell("Lịch sử xe vào/ra", request)
        loading = _loading_label()
        error = _error_label()
        state = {"page": 1, "pages": 1}
        with ui.row().classes("w-full gap-3 items-end flex-wrap"):
            search = ui.input("UID hoặc biển số").props("clearable").classes("min-w-64")
            event_type = ui.select(
                ["ALL", "ENTRY", "EXIT", "DENY", "REVIEW", "ERROR"],
                value="ALL",
                label="Loại sự kiện",
            ).classes("min-w-48")
            day = ui.input("Ngày (Asia/Ho_Chi_Minh)").props("type=date clearable")
            ui.button("Lọc", icon="filter_alt", on_click=lambda: apply_filters())
        table = _table(EVENT_COLUMNS).props("hide-pagination")
        empty = _empty_label("Không có sự kiện phù hợp bộ lọc.")
        with ui.row().classes("items-center justify-end w-full"):
            previous = ui.button(icon="chevron_left").props("flat round")
            page_label = ui.label("Trang 1 / 1")
            following = ui.button(icon="chevron_right").props("flat round")

        async def refresh():
            loading.set_visibility(True)
            try:
                if not await _session_is_valid(request, factory):
                    ui.navigate.to("/login")
                    return
                selected_day = date.fromisoformat(day.value) if day.value else None
                data = await _load(
                    history_page,
                    factory,
                    search=search.value or "",
                    event_type=event_type.value or "ALL",
                    day=selected_day,
                    page=state["page"],
                )
                state["pages"] = max(1, (data["total"] + 19) // 20)
                if state["page"] > state["pages"]:
                    state["page"] = state["pages"]
                    return await refresh()
                table.rows = data["rows"]
                table.update()
                empty.set_visibility(not data["rows"])
                page_label.set_text(
                    f"Trang {state['page']} / {state['pages']} · "
                    f"{data['total']} sự kiện"
                )
                previous.set_enabled(state["page"] > 1)
                following.set_enabled(state["page"] < state["pages"])
                error.set_visibility(False)
            except (ValueError, TypeError):
                error.set_text("Ngày lọc không hợp lệ.")
                error.set_visibility(True)
            except Exception:
                LOG.exception("History refresh failed")
                error.set_text(
                    "Tạm thời không thể đọc dữ liệu. Hệ thống sẽ tự thử lại."
                )
                error.set_visibility(True)
            finally:
                loading.set_visibility(False)

        async def apply_filters():
            state["page"] = 1
            await refresh()

        async def move(delta: int):
            state["page"] = min(state["pages"], max(1, state["page"] + delta))
            await refresh()

        previous.on("click", lambda: move(-1))
        following.on("click", lambda: move(1))
        await refresh()
        ui.timer(5.0, refresh)

    @ui.page("/inside")
    async def inside_page(request: Request):
        _shell("Xe đang trong bãi", request)
        loading = _loading_label()
        error = _error_label()
        columns = [
            {"name": "plate", "label": "Biển số", "field": "plate"},
            {"name": "uid", "label": "UID", "field": "uid"},
            {"name": "entry", "label": "Thời gian vào", "field": "entry"},
            {"name": "duration", "label": "Thời lượng hiện tại", "field": "duration"},
        ]
        table = _table(columns)
        empty = _empty_label("Hiện không có xe nào trong bãi.")

        async def refresh():
            loading.set_visibility(True)
            try:
                if not await _session_is_valid(request, factory):
                    ui.navigate.to("/login")
                    return
                rows = await _load(inside_sessions, factory)
                table.rows = rows
                table.update()
                empty.set_visibility(not rows)
                error.set_visibility(False)
            except Exception:
                LOG.exception("Inside sessions refresh failed")
                error.set_text("Tạm thời không thể đọc danh sách xe trong bãi.")
                error.set_visibility(True)
            finally:
                loading.set_visibility(False)

        await refresh()
        ui.timer(5.0, refresh)

    @ui.page("/rfid")
    async def rfid_page(request: Request):
        _shell("Quản lý thẻ RFID", request)
        ui.label(
            "Chỉ đọc: trạng thái enabled/disabled vẫn do SQLite trên Raspberry Pi "
            "quản lý và chưa được đồng bộ."
        ).classes("bg-amber-50 text-amber-900 rounded-lg p-4 w-full")
        loading = _loading_label()
        error = _error_label()
        columns = [
            {"name": "uid", "label": "UID đã quan sát", "field": "uid"},
            {"name": "last_seen", "label": "Lần xuất hiện cuối", "field": "last_seen"},
            {"name": "count", "label": "Số sự kiện", "field": "count"},
            {"name": "last_type", "label": "Loại gần nhất", "field": "last_type"},
        ]
        table = _table(columns)
        empty = _empty_label("Chưa quan sát thấy UID nào.")

        async def refresh():
            loading.set_visibility(True)
            try:
                if not await _session_is_valid(request, factory):
                    ui.navigate.to("/login")
                    return
                rows = await _load(rfid_cards, factory)
                table.rows = rows
                table.update()
                empty.set_visibility(not rows)
                error.set_visibility(False)
            except Exception:
                LOG.exception("RFID page refresh failed")
                error.set_text("Tạm thời không thể đọc dữ liệu RFID.")
                error.set_visibility(True)
            finally:
                loading.set_visibility(False)

        await refresh()
        ui.timer(5.0, refresh)

    @ui.page("/device")
    async def device_page(request: Request):
        _shell("Trạng thái Raspberry Pi", request)
        ui.label(
            "Trạng thái dựa trên lần nhận event, không phải heartbeat và không "
            "khẳng định thiết bị ONLINE."
        ).classes("bg-blue-50 text-blue-900 rounded-lg p-4 w-full")
        loading = _loading_label()
        error = _error_label()
        values = {}
        with ui.card().classes("w-full max-w-3xl rounded-xl"):
            for key, label in (
                ("device_id", "Device ID"),
                ("state", "Trạng thái liên lạc"),
                ("last_received", "Lần nhận sự kiện cuối"),
                ("elapsed", "Thời gian kể từ lần nhận cuối"),
                ("last_type", "Loại sự kiện cuối"),
                ("count", "Tổng event đã nhận"),
            ):
                with ui.row().classes(
                    "w-full justify-between border-b border-slate-100 py-2"
                ):
                    ui.label(label).classes("text-slate-500")
                    values[key] = ui.label("—").classes("font-medium")

        async def refresh():
            loading.set_visibility(True)
            try:
                if not await _session_is_valid(request, factory):
                    ui.navigate.to("/login")
                    return
                data = await _load(
                    device_status, factory, settings.recent_device_minutes
                )
                for key, label in values.items():
                    label.set_text(str(data[key]))
                error.set_visibility(False)
            except Exception:
                LOG.exception("Device page refresh failed")
                error.set_text("Tạm thời không thể đọc trạng thái thiết bị.")
                error.set_visibility(True)
            finally:
                loading.set_visibility(False)

        await refresh()
        ui.timer(5.0, refresh)

    @ui.page("/plates")
    async def plates_page(request: Request):
        _shell("Ảnh biển số", request)
        ui.label(
            "Phiên bản hiện tại chỉ đồng bộ metadata. Ảnh chưa được đồng bộ lên Cloud."
        ).classes("bg-violet-50 text-violet-900 rounded-lg p-4 w-full")
        loading = _loading_label()
        error = _error_label()
        columns = [
            {"name": "plate", "label": "Biển số", "field": "plate"},
            {"name": "time", "label": "Thời gian", "field": "time"},
            {"name": "uid", "label": "UID", "field": "uid"},
            {"name": "ocr", "label": "OCR confidence", "field": "ocr"},
            {
                "name": "detection",
                "label": "Detection confidence",
                "field": "detection",
            },
            {"name": "image", "label": "Ảnh", "field": "image"},
        ]
        table = _table(columns)
        empty = _empty_label("Chưa có metadata biển số.")

        async def refresh():
            loading.set_visibility(True)
            try:
                if not await _session_is_valid(request, factory):
                    ui.navigate.to("/login")
                    return
                rows = await _load(plate_metadata, factory)
                table.rows = rows
                table.update()
                empty.set_visibility(not rows)
                error.set_visibility(False)
            except Exception:
                LOG.exception("Plate page refresh failed")
                error.set_text("Tạm thời không thể đọc metadata biển số.")
                error.set_visibility(True)
            finally:
                loading.set_visibility(False)

        await refresh()
        ui.timer(5.0, refresh)

    ui.run_with(
        fastapi_app,
        storage_secret=settings.nicegui_storage_secret,
        title="SmartParking Admin",
        favicon="🚗",
    )
