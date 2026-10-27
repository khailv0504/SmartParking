# SmartParking Cloud

Một ASGI app duy nhất gồm FastAPI (nhận event) + NiceGUI (Admin dashboard), dùng
PostgreSQL làm nguồn dữ liệu Cloud. Cloud không chạy YOLO/OCR, không lưu ảnh và
không thay đổi quyền thẻ trên Raspberry Pi.

## 1. Kiến trúc và quyết định quan trọng

- `edge_events` là log bất biến. `event_id` là khóa chính và `payload_hash` dùng
  phát hiện cùng ID nhưng khác nội dung.
- `parking_sessions` là projection từ ENTRY/EXIT theo `session_id`. Các cột ENTRY
  nullable để EXIT tới trước vẫn được lưu, sau đó ENTRY hoàn thiện cùng một row.
- `REVIEW`, `DENY`, `ERROR` không mở/đóng session.
- Admin session lưu phía server trong PostgreSQL. Cookie chỉ chứa token ngẫu nhiên,
  có `HttpOnly`, `SameSite=Strict`, `Secure` ở production; logout thu hồi token.
- Timestamp lưu UTC; thống kê theo ngày chuyển biên ngày `Asia/Ho_Chi_Minh` sang UTC.
- Mỗi page có đúng một timer 5 giây và query SQLAlchemy chạy qua worker thread,
  không reload toàn trang và không block event loop.

## 2. Chạy local trên Ubuntu

Yêu cầu Python 3.11–3.14 và PostgreSQL. Từ repository root:

```bash
python3 -m venv .venv-cloud
source .venv-cloud/bin/activate
python -m pip install -U pip
pip install -r cloud/requirements-dev.txt
sudo -u postgres createuser --pwprompt smartparking
sudo -u postgres createdb --owner=smartparking smartparking
cp cloud/.env.example .env
```

Sửa `.env`; tạo secret/hash bằng:

```bash
python -m cloud.auth hash-password
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Không đưa password plaintext vào `ADMIN_PASSWORD_HASH`; giá trị phải bắt đầu bằng
`$argon2id$`. Chạy migration và app:

```bash
alembic -c cloud/alembic.ini upgrade head
uvicorn cloud.main:app --host 127.0.0.1 --port 8000 --reload
```

Mở `http://127.0.0.1:8000/login`. Với local HTTP, đặt `APP_ENV=development` để
cookie không có cờ Secure. Production bắt buộc `APP_ENV=production` và HTTPS.

Kiểm tra kết nối/migration:

```bash
psql "$DATABASE_URL" -c 'select version();'
alembic -c cloud/alembic.ini current
curl http://127.0.0.1:8000/api/health
```

App không gọi `create_all()` khi chạy. Schema production chỉ được thay đổi qua
Alembic.

## 3. Test

```bash
pytest -q cloud/tests -m "not postgres"
python -m unittest discover -s edge/tests -v
```

Các test thông thường dùng SQLite tạm để kiểm API/logic nhanh, không giả là
PostgreSQL. Test end-to-end PostgreSQL thật cần một database **chỉ dành cho test**
vì test xóa dữ liệu trong ba bảng:

```bash
export TEST_POSTGRES_URL='postgresql+psycopg://user:pass@localhost:5432/smartparking_test'
pytest -q cloud/tests/test_postgres_integration.py
```

Test đó khởi động HTTP server local, dùng chính `edge.database.Database` và
`edge.sync.SyncWorker`, gửi payload qua HTTP, xác nhận PostgreSQL, ACK, `SYNCED`,
dedupe và shutdown server. Không có `TEST_POSTGRES_URL` thì kết quả phải là SKIPPED.

## 4. API contract

### `POST /api/v1/edge/events`

Headers:

```text
Authorization: Bearer <SMARTPARKING_DEVICE_TOKEN>
Idempotency-Key: <event_id>
Content-Type: application/json
```

- Event mới: `201`.
- Cùng `event_id`, cùng nội dung canonical: `201`, `duplicate=true`.
- Cùng `event_id`, khác nội dung: `409`; row cũ không bị ghi đè.
- Token sai/thiếu: `401`; payload/header sai: `422`.

### `GET /api/health`

Trả `{"status":"ok"}` và không query/hiển thị credential, UID hay biển số. API
docs chỉ bật khi `APP_ENV` khác `production`.

### Xử lý event xung đột 409

Edge hiện retry mọi HTTP error nên 409 không tự hết. Trên Pi:

```bash
sqlite3 edge/data/parking.db \
  "select event_id,attempts,last_error,payload from sync_outbox where status='PENDING';"
```

Đối chiếu `event_id` đó trong PostgreSQL và payload SQLite (log Cloud cố ý không
ghi UID/plate/token). Xác định bản nào đúng từ SQLite `events` và thực tế vận hành;
backup cả hai DB trước khi sửa. Không đổi `event_id` hoặc ghi đè âm thầm. Nếu row
Cloud đúng và outbox Pi là bản sai/không cần gửi, operator có thể đánh dấu row
outbox đã xử lý sau khi lập biên bản. Nếu Cloud sai, sửa dữ liệu bằng một migration/
script kiểm soát và rebuild projection; phiên bản này cố ý không cung cấp API xóa.

## 5. Deploy Render bằng Blueprint

`render.yaml` tạo một Python Web Service và một PostgreSQL database cùng region.
Secrets không nằm trong Git. Do free Web Service không có pre-deploy command,
`startCommand` chạy `alembic upgrade head` trước khi mở port Uvicorn. Nếu nâng lên
paid plan, nên chuyển migration sang `preDeployCommand`.

Blueprint pin `PYTHON_VERSION=3.14.8` thay vì phụ thuộc vào default thay đổi theo
ngày tạo service của Render.

Blueprint chạy từ repository root (không đặt Root Directory là `cloud`) để import
package `cloud.*` và đồng thời dùng Edge integration tests. Cấu hình tương đương:

```text
Root Directory: (để trống / repository root)
Build Command: pip install -r cloud/requirements.txt
Start Command: alembic -c cloud/alembic.ini upgrade head && uvicorn cloud.main:app --host 0.0.0.0 --port $PORT
```

1. Tạo Git repository trên GitHub/GitLab/Bitbucket, commit và push `render.yaml`.
2. Mở Render Dashboard → **New → Blueprint** → chọn repository.
3. Điền các biến `sync: false`:
   - `SMARTPARKING_DEVICE_TOKEN`: token ngẫu nhiên dài, riêng cho Pi.
   - `ADMIN_USERNAME`: ví dụ `admin`.
   - `ADMIN_PASSWORD_HASH`: Argon2 hash tạo local, không phải password plaintext.
4. `NICEGUI_STORAGE_SECRET` được Render sinh; `DATABASE_URL` lấy internal connection
   string bằng `fromDatabase`.
5. Apply Blueprint, theo dõi build/start log. Chỉ coi deploy thành công khi trạng
   thái live và `GET /api/health` trả 200.
6. Kiểm tra log có dòng Alembic nâng tới revision `20261027_0001`, sau đó đăng nhập
   và thử một event test không trùng ID production.

Nếu có Render CLI phiên bản hỗ trợ:

```bash
render blueprints validate render.yaml
```

Blueprint đang chọn plan `free` và region `singapore`. Hãy xác nhận plan Free còn
được workspace mới cung cấp tại thời điểm tạo; Render có thể thay đổi tên plan,
giới hạn và thời hạn database. Không có tài nguyên nào được tạo bởi repository này.

### Rollback deploy

- Nếu build/start/health check thất bại, xem stage lỗi đầu tiên; sửa và deploy lại.
- Rollback app trong Render Dashboard chỉ rollback artifact ứng dụng, **không**
  rollback schema/data PostgreSQL.
- Chỉ chạy `alembic downgrade <revision>` khi migration tương ứng đã được kiểm tra
  downgrade và app cũ tương thích. An toàn hơn là forward-fix.
- Trước migration phá vỡ tương thích, `pg_dump` và thử restore trên DB khác.

### Backup/export

Render Free không nên được xem là có managed backup/PITR. Dùng External Database
URL (TLS) từ Dashboard trên máy quản trị, không lưu URL vào Git:

```bash
pg_dump --format=custom --no-owner --no-acl "$RENDER_EXTERNAL_DATABASE_URL" \
  > "smartparking-$(date +%F).dump"
createdb smartparking_restore_test
pg_restore --no-owner --no-acl --dbname=smartparking_restore_test smartparking-YYYY-MM-DD.dump
```

Giữ file backup mã hóa ngoài Render và định kỳ test restore. Filesystem Web Service
là ephemeral nên không ghi backup/ảnh vào đó.

## 6. Cấu hình Raspberry Pi

Trên Pi, giữ SQLite hiện tại và đặt đúng URL endpoint:

```bash
export SMARTPARKING_DEVICE_ID='raspberrypi_01'
export SMARTPARKING_CLOUD_URL='https://YOUR-SERVICE.onrender.com/api/v1/edge/events'
export SMARTPARKING_DEVICE_TOKEN='SAME_SECRET_AS_RENDER'
python -m edge.main sync-once
```

Sau đó cấu hình ba biến này trong `EnvironmentFile=` của systemd. Không dùng Admin
password làm device token. `SyncWorker` timeout 6 giây, at-least-once và exponential
backoff tối đa thực tế 256 giây; Cloud mất kết nối thì event vẫn `PENDING` trong
SQLite.

## 7. Giới hạn có chủ ý

- Không heartbeat: trang thiết bị chỉ nói `Recently active`, `No recent events` hoặc
  `Unknown`, không nói ONLINE. Giai đoạn sau có thể thêm
  `POST /api/v1/edge/heartbeat` dùng cùng device token và lưu `sent_at`, phiên bản
  phần mềm, nhưng không giả lập CPU/RAM/nhiệt độ nếu Pi không gửi.
- Cloud chỉ biết UID đã xuất hiện; không biết `cards.enabled` trên Pi. Giai đoạn sau
  có thể thêm `POST /api/v1/edge/card-snapshots`: Pi chỉ publish snapshot
  `uid/enabled/version`, Cloud tuyệt đối không gửi lệnh thay đổi thẻ ngược về Pi.
- `image_path` không nằm trong payload. Trang ảnh chỉ hiển thị metadata và thông báo
  chưa đồng bộ; không lưu ảnh trên filesystem Render.
- Một bãi, một làn, một Pi, một Admin. Không có Redis/Celery/MQTT/microservice.
- Theo tài liệu Render kiểm tra ngày 27/10/2026, Free Web Service sleep sau 15 phút
  không có HTTP/WebSocket và cold start có thể khoảng một phút; request Pi lúc đó
  có thể timeout nhưng outbox vẫn retry. Free PostgreSQL giới hạn 1 GB, hết hạn sau
  30 ngày và không có managed backup. Luôn kiểm tra lại Dashboard trước khi deploy
  vì chính sách có thể thay đổi.
