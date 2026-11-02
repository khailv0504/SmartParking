"""Edge entry point: demo, image inference, RFID production loop, local status.

Usage from repo root:
    python -m edge.main demo
    python -m edge.main register-card --uid A1B2C3D4
    python -m edge.main recognize --image path/to/plate.jpg
    python -m edge.main run --real --camera 0
    python -m edge.main sync-once
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time

from .controller import ParkingController
from .database import Database
from .frame_capture import FrameSelector, MockFrameSelector
from .recognition import MockRecognizer, YOLOPlateRecognizer
from .rfid import RC522Reader
from .sync import SyncWorker

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


def config_database(args) -> Database:
    return Database(args.db or ROOT/'data'/'parking.db',device_id=args.device_id)


def make_real_recognizer(args):
    return YOLOPlateRecognizer(args.detector or ROOT/'models'/'detector.onnx',
                               args.recognizer or ROOT/'models'/'recognizer.onnx',
                               strict_roi_quality=args.strict_roi_quality)


def worker_from_env(db: Database):
    url = os.environ.get('SMARTPARKING_CLOUD_URL', '').strip()
    token = os.environ.get('SMARTPARKING_DEVICE_TOKEN','').strip()
    if not url or not token:
        LOG.warning('Sync disabled: set SMARTPARKING_CLOUD_URL and SMARTPARKING_DEVICE_TOKEN')
        return None
    return SyncWorker(db,url,token)


def main() -> int:
    parser = argparse.ArgumentParser(description='SmartParking Edge')
    parser.add_argument('--db', default='', help='SQLite file path')
    parser.add_argument('--device-id', default=os.getenv('SMARTPARKING_DEVICE_ID','raspberrypi_01'))
    sub = parser.add_subparsers(dest='command', required=True)
    p_card = sub.add_parser('register-card')
    p_card.add_argument('--uid', required=True)
    p_card.add_argument('--disable',action='store_true')
    sub.add_parser('status')
    sub.add_parser('sync-once')
    p_demo = sub.add_parser('demo')
    p_demo.add_argument('--uid',default='A1B2C3D4')
    p_demo.add_argument('--plate',default='59A12345')
    p_img = sub.add_parser('recognize')
    p_img.add_argument('--image',required=True)
    p_img.add_argument('--output',default='',help='Optional image with plate bounding box')
    for p in (p_img,):
        p.add_argument('--detector',default='')
        p.add_argument('--recognizer',default='')
        p.add_argument('--strict-roi-quality',action='store_true')
    p_run = sub.add_parser('run')
    p_run.add_argument('--real', action='store_true', help='Required to open RC522 and camera')
    p_run.add_argument('--camera',type=int,default=0)
    p_run.add_argument('--detector',default='')
    p_run.add_argument('--recognizer',default='')
    p_run.add_argument('--strict-roi-quality',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    db=config_database(args)

    if args.command=='register-card':
        db.register_card(args.uid,not args.disable)
        print('Card updated:',args.uid.upper(),'enabled:',not args.disable)
        return 0
    if args.command=='status':
        print(json.dumps(db.summary(),indent=2))
        return 0
    if args.command=='sync-once':
        worker=worker_from_env(db)
        if worker is None:
            raise RuntimeError('Cloud URL/token not configured')
        print('synced,failed:',worker.run_once())
        return 0
    if args.command=='demo':
        db.register_card(args.uid)
        controller=ParkingController(db,MockFrameSelector(),
                                     MockRecognizer(args.plate),cooldown_seconds=0)
        first=controller.process(args.uid)
        second=controller.process(args.uid)
        print(json.dumps({'events':[vars(first),vars(second)],'summary':db.summary()},indent=2))
        return 0
    if args.command=='recognize':
        import cv2
        frame=cv2.imread(args.image)
        if frame is None:
            raise ValueError(f'Cannot read image: {args.image}')
        rec=make_real_recognizer(args).recognize(frame)
        if args.output:
            import cv2
            if rec.plate_box is not None:
                b=rec.plate_box
                cv2.rectangle(frame,(b.x1,b.y1),(b.x2,b.y2),(0,255,0),2)
                cv2.putText(frame,rec.plate or (rec.reason or 'REVIEW'),
                            (b.x1,max(25,b.y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
            output=Path(args.output)
            output.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(output),frame):
                raise OSError(f'Cannot write annotated image: {output}')
        print(json.dumps({'plate':rec.plate,'confidence':rec.confidence,
                          'detector_confidence':rec.detector_confidence,
                          'quality_score':rec.quality_score,
                          'rows':rec.row_texts,'reason':rec.reason,
                          'latency_ms':rec.latency_ms},indent=2,ensure_ascii=False))
        return 0
    if args.command=='run':
        if not args.real:
            parser.error('run requires --real: hardware access is never implicit')
        model=make_real_recognizer(args)
        selector=FrameSelector(ROOT/'data'/'captures',camera_index=args.camera)
        controller=ParkingController(db,selector,model)
        reader=RC522Reader()
        stop=threading.Event()
        def stop_signal(_sig,_frame):
            stop.set()
        signal.signal(signal.SIGINT,stop_signal)
        signal.signal(signal.SIGTERM,stop_signal)
        sync=worker_from_env(db)
        if sync:
            thread=threading.Thread(target=sync.run_forever,args=(stop,),daemon=True,name='cloud-sync')
            thread.start()
        LOG.info('Waiting for RC522 taps. Ctrl+C to stop.')
        while not stop.is_set():
            uid=reader.read_uid()
            if uid:
                result=controller.process(uid)
                LOG.info('event: %s',json.dumps(vars(result)))
            else:
                stop.wait(.1)
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
