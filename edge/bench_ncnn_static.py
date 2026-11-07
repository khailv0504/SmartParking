"""Image-only SmartParking NCNN benchmark (no RFID, camera, SQLite or Cloud).

Run once per model variant in a new Python process. Time includes letterbox,
NCNN detector, crop/quality, NCNN OCR, NMS, character ordering. Not RFID-to-event.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

import cv2
import numpy as np

from .ncnn_inference import NCNNPlateRecognizer

EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.webp'}


def _read_images(items: list[Path], limit: int) -> list[Path]:
    files=[]
    for src in items:
        if src.is_file():
            if src.suffix.lower() not in EXTENSIONS:
                raise ValueError(f'Unsupported image: {src}')
            files.append(src)
        elif src.is_dir():
            files.extend(x for x in sorted(src.rglob('*')) if x.is_file() and x.suffix.lower() in EXTENSIONS)
        else:
            raise FileNotFoundError(src)
    files=list(dict.fromkeys(x.resolve() for x in files))
    if limit:
        files=files[:limit]
    if not files:
        raise ValueError('No images found: supply --images image.jpg or images directory')
    return files


def _hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _ground_truth(file: Path | None) -> dict[str,str]:
    if file is None: return {}
    out={}
    with file.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'image','plate'} <= set(reader.fieldnames or []):
            raise ValueError('Ground truth CSV needs columns image,plate')
        for item in reader:
            key=Path(item['image'].strip()).name
            val=item['plate'].strip().upper().replace(' ','').replace('-','').replace('.','')
            if not key or not val or key in out:
                raise ValueError(f'Blank/duplicate ground truth record: {key!r}')
            out[key]=val
    return out


def _percentile(arr: list[float], pct: int) -> float:
    return round(float(np.percentile(arr,pct)),3) if arr else float('nan')


def main() -> int:
    parser=argparse.ArgumentParser(description='Static YOLO26 NCNN FP16 benchmark (no camera, RFID or database)')
    parser.add_argument('--detector',type=Path,required=True,help='NCNN detector directory')
    parser.add_argument('--recognizer',type=Path,required=True,help='NCNN OCR directory')
    parser.add_argument('--images',type=Path,nargs='+',required=True,help='Full-frame vehicle images/directories')
    parser.add_argument('--out',type=Path,required=True,help='NEW output directory; will not overwrite')
    parser.add_argument('--ground-truth',type=Path,default=None,help='Optional CSV: image,plate')
    parser.add_argument('--label',default='unspecified',help='Human-readable model/config label')
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--strict-roi-quality',action='store_true')
    args=parser.parse_args()
    if args.repeat<1 or args.warmup<0 or args.threads<1 or args.limit<0:
        parser.error('repeat>=1, warmup>=0, threads>=1, limit>=0 are required')
    if args.out.exists():
        parser.error(f'Output directory exists; refusing overwrite: {args.out}')
    files=_read_images(args.images,args.limit)
    labels=_ground_truth(args.ground_truth)
    if labels:
        missing=[str(f) for f in files if f.name not in labels]
        if missing:
            parser.error(f'Ground truth missing {len(missing)} image(s), first: {missing[0]}')
    # Load files before creating output directory. No camera is opened.
    frames=[]
    for p in files:
        frame=cv2.imread(str(p))
        if frame is None: raise ValueError(f'Failed to read image: {p}')
        frames.append((p,frame))

    tic=time.perf_counter()
    rec=NCNNPlateRecognizer(args.detector,args.recognizer,
                            strict_roi_quality=args.strict_roi_quality,threads=args.threads)
    model_load_ms=(time.perf_counter()-tic)*1000
    for _ in range(args.warmup):
        rec.recognize(frames[0][1])

    rows=[]
    for repeat in range(args.repeat):
        for path,frame in frames:
            t0=time.perf_counter()
            result=rec.recognize(frame)
            ms=(time.perf_counter()-t0)*1000
            record={
                'image':str(path), 'repeat':repeat,
                'plate':result.plate or '', 'reason':result.reason or '',
                'confidence':result.confidence,
                'detector_confidence':result.detector_confidence,
                'quality_score':result.quality_score,
                'pipeline_ms':round(result.latency_ms,3),
                'wall_ms':round(ms,3),
            }
            if labels:
                record['ground_truth']=labels[path.name]
                record['exact_match']=int((result.plate or '')==labels[path.name])
            rows.append(record)

    timings=[row['wall_ms'] for row in rows]
    elapsed=sum(timings)/1000
    peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024  # Linux ru_maxrss is KiB.
    summary={
        'backend':'ncnn', 'label':args.label,
        'detector_dir':str(args.detector.resolve()), 'ocr_dir':str(args.recognizer.resolve()),
        'detector_param_sha256':_hash(args.detector/'model.ncnn.param'),
        'detector_bin_sha256':_hash(args.detector/'model.ncnn.bin'),
        'ocr_param_sha256':_hash(args.recognizer/'model.ncnn.param'),
        'ocr_bin_sha256':_hash(args.recognizer/'model.ncnn.bin'),
        'images':len(files),'repeats':args.repeat,'warmup':args.warmup,'threads':args.threads,
        'samples':len(rows),'model_load_ms':round(model_load_ms,3),
        'p50_ms':_percentile(timings,50),'p95_ms':_percentile(timings,95),
        'mean_ms':round(statistics.fmean(timings),3),
        'inference_only_throughput_fps':round(len(rows)/elapsed,3) if elapsed else None,
        'peak_process_rss_mb':round(peak_rss_mb,2),
        'exact_plate_accuracy': round(sum(r['exact_match'] for r in rows)/len(rows),4) if labels else None,
        'note':'Image-only ALPR latency includes preprocessing + detector + OCR + postprocessing; NOT camera, IQA frame selection, RFID, DB, Cloud, or gate latency. Peak RSS is a process high-water mark, not marginal model RAM.',
    }
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/'details.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
