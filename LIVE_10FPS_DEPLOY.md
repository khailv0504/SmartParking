# SmartParking — nâng cấp livestream 10 FPS (an toàn, có rollback)

Bản vá không sửa SQLite, mô hình ONNX, RFID, hay logic ENTRY/EXIT. Cloud tiếp tục hỗ trợ `POST /api/v1/edge/live-frame` và `GET /api/live/frame.jpg` cũ. Cơ chế mới: một CameraSource -> JPEG encoder -> WSS lên Render -> latest-frame RAM -> MJPEG đến browser. Không ghi video. Một Uvicorn worker vì bộ đệm frame lưu theo process.

## Kiểm thử trên laptop hoặc Pi có Git clone

1. Trong repo development: `git switch -c feature/live-ws-10fps 922fe85a29cce5051b4a8d5ffba031d59fae7bbc` (trước đó `git fetch origin`, bảo đảm working tree sạch).
2. Chạy `python3 /path/to/patch/apply_patch.py . --dry-run`, rồi `python3 /path/to/patch/apply_patch.py .`.
3. Cloud test (trên máy có Python phù hợp): `python -m pip install -r cloud/requirements-dev.txt && python -m pytest cloud/tests/test_live.py cloud/tests/test_live_stream.py -q`.
4. Edge test: `python -m pip install -r edge/requirements.txt pytest && python -m pytest edge/tests/test_live_config.py -q`.
5. Kiểm tra `git diff --check` và `git status --short` sau đó commit + push nhánh, tạo PR GitHub; **không merge nếu chưa test**.

## Triển khai an toàn

1. Render: merge PR sau review/CI; kiểm tra Render deploy `live` và `GET /api/health`. LƯU Ý Render autoDeploy theo commit ở main.
2. Raspberry Pi: `cd ~/SmartParking && mkdir -p ~/smartparking-backups && python3 -c 'import sqlite3; c=sqlite3.connect("edge/data/parking.db"); b=sqlite3.connect("/home/khai/smartparking-backups/parking-before-live10fps.db"); c.backup(b); b.close(); c.close()'`.
3. Dừng chương trình Pi đang chạy (Ctrl+C nếu thủ công). Không chạy `edge.live_preview` song song.
4. Copy **chỉ** `edge/main.py`, `edge/live_uplink.py` từ branch đã test sang `~/SmartParking/edge/`; trước khi copy hãy backup 2 file hiện tại. Không copy đè `edge/data`, secrets, `.venv`, ONNX model.
5. `cd ~/SmartParking && source .venv/bin/activate && python -m pip install 'websocket-client>=1.8,<2'`.
6. Giữ 3 biến môi trường hiện tại: `SMARTPARKING_DEVICE_ID`, `SMARTPARKING_CLOUD_URL`, `SMARTPARKING_DEVICE_TOKEN` (không in token).
7. Chạy:
   ```bash
   cd ~/SmartParking
   source .venv/bin/activate
   python -m edge.main run --real --camera 0 --live \
      --live-transport ws --live-fps 10 --live-width 640
   ```
8. Mở trang `/live` khi đã đăng nhập. Status cho biết `incoming_fps` thực tế. Test RFID, AI, SQLite outbox, nhiệt độ/RAM/CPU, mất mạng và kết nối lại. Nếu lag giảm `--live-fps 5`.

## Rollback

Dừng tiến trình với Ctrl+C. Khôi phục hai file đã backup của Pi và chạy tham số cũ:
`python -m edge.main run --real --camera 0 --live --live-fps 1 --live-width 640`.
Cloud mới vẫn tương thích HTTP cũ. Nếu cloud lỗi, redeploy commit trước khi merge (commit ban đầu `922fe85a29cce5051b4a8d5ffba031d59fae7bbc`).

**Giới hạn**: 10 FPS là chỉ tiêu thử nghiệm, chưa có benchmark trên Pi 1GB/Render thực tế. Nền tảng miễn phí có thể sleep/cold-start và ảnh hưởng độ ổn định kết nối.
