# Đồng bộ danh sách thẻ RFID — SmartParking

## Luồng dữ liệu

- Pi SQLite `cards` (nguồn dữ liệu gốc) -> HTTPS POST `/api/v1/edge/card-snapshots`
- Cloud PostgreSQL `rfid_cards` + `rfid_snapshot_state` -> NiceGUI `/rfid` (chỉ xem)
- Mọi sửa đổi/ủy quyền thẻ vẫn do SQLite/Edge quyết định; Web không điều khiển cổng.
- `snapshot_version` tăng trong transaction SQLite mỗi lần đăng ký/vô hiệu hóa.
- Cloud bác bỏ phiên bản cũ hoặc cùng phiên bản nhưng khác nội dung (409).
- Pi lưu phiên bản đã ACK; offline tự retry với exponential backoff tới 300 giây.

## Triển khai Cloud (sau khi tests thành công)

1. Backup PostgreSQL và bảo đảm Render `DATABASE_URL` cùng token cấu hình ổn định.
2. Merge nhánh vào `main` để Render tự deploy (không chạy deploy thủ công khi auto-deploy bật).
3. Theo dõi log Alembic chạy revision `20261102_0002` và Web Service chuyển `Live`.
4. Đăng nhập `/rfid`. Ban đầu sẽ trống cho tới khi Pi gửi snapshot.

## Triển khai Edge trên Raspberry Pi

**Backup trước khi sửa:**

```bash
cp ~/SmartParking/edge/data/parking.db ~/parking-before-rfid-sync.db
```

Thay source code trong `~/SmartParking/edge` bằng các file cập nhật trong GitHub
(giữ nguyên `edge/data/`, model weights và `.env`). Cấu hình biến môi trường:

```bash
export SMARTPARKING_DEVICE_ID=raspberrypi_01
export SMARTPARKING_CLOUD_URL=https://smartparking-cloud.onrender.com/api/v1/edge/events
# token cùng giá trị đã cấu hình trong Render; nhập riêng, KHÔNG đưa vào lệnh/git
read -rsp 'Device token: ' SMARTPARKING_DEVICE_TOKEN; echo
export SMARTPARKING_DEVICE_TOKEN
```

Sau đó, từ root dự án Pi:

```bash
python -m edge.main sync-once
python -m edge.main status
sqlite3 edge/data/parking.db 'SELECT snapshot_version,synced_version,attempts,last_error FROM card_sync_state;'
```

`python -m edge.main run --real` tự chạy cả sync events và cards mỗi 5 giây.
Nếu dùng `systemd`, bảo đảm `User`, `WorkingDirectory`, `ExecStart`, `EnvironmentFile`
trùng máy thực tế (file template cũ đang đặt `User=pi` và `/opt/SmartParking`).

Trên Web xem `https://smartparking-cloud.onrender.com/rfid` sau đăng nhập.
Không cần upload file `.db`, model, UID hoặc token lên GitHub.
