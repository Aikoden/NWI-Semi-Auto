"""NWI Semi-Auto Web 版のサーバ。

デスクトップ版 (main.py の run_ai_logic) と同じ処理を API として提供する:
    predict_mask -> full_postprocessing_pipeline -> detect_baseline

デスクトップ版と同様、画面に渡すのは大腿骨輪郭とベースライン (P1, P2) だけ。
ノッチ幅は利用者がブラウザ上で自分で引き、顆部幅と NWI はブラウザ側で計算する。
画像はメモリ上で処理するだけで、サーバには保存しない。
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms import (  # noqa: E402
    AIModel,
    detect_baseline,
    full_postprocessing_pipeline,
)

from laterality import LateralityModel

SAMPLES_DIR = ROOT / "samples"
STATIC_DIR = Path(__file__).resolve().parent / "static"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # Lambda のペイロード上限 (6 MB) より小さく

# 輪郭はスプライン補間で元の画素数の 8 倍の点になっている。
# 転送量を抑えるため 4 点に 1 点だけ送る (それでも 1 画素あたり 2 点)。
CONTOUR_STRIDE = 4

@lru_cache(maxsize=1)
def get_model() -> AIModel:
    return AIModel()


@lru_cache(maxsize=1)
def get_laterality_model():
    return LateralityModel()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Lambda のコールドスタート時 (初期化フェーズ) にモデルを読み込んでおく
    get_model()
    get_laterality_model()
    yield


app = FastAPI(
    title="NWI Semi-Auto",
    description="半自動 NWI 計測ツールの Web API。AI が大腿骨輪郭とベースラインを返す。",
    version="1.1.0",
    lifespan=lifespan,
)


def list_samples() -> list[str]:
    if not SAMPLES_DIR.is_dir():
        return []
    names = [p.name for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    # Normal-2 が Normal-10 より前に来るよう、番号で並べる
    return sorted(names, key=lambda n: (len(n), n))


def analyse(img_bgr: np.ndarray) -> dict:
    """AI 解析。失敗時は HTTPException を送出する。"""
    model = get_model()
    if model.seg_session is None:
        raise HTTPException(status_code=503, detail="AI モデルが読み込まれていません。")

    started = time.perf_counter()
    mask = model.predict_mask(img_bgr)
    if mask is None:
        raise HTTPException(status_code=500, detail="AI 推論に失敗しました。")

    cnt_int, cnt_float = full_postprocessing_pipeline(mask)
    if cnt_int is None:
        raise HTTPException(status_code=422, detail="AI解析に失敗しました（輪郭検出不可）。")

    ai_res = detect_baseline(cnt_float)
    if not ai_res:
        raise HTTPException(status_code=422, detail="AI解析に失敗しました（基線を算出できません）。")

    laterality = get_laterality_model().predict(img_bgr, mask)
    h, w = img_bgr.shape[:2]
    contour = np.round(cnt_float[::CONTOUR_STRIDE], 2).tolist()
    return {
        "laterality": laterality,
        "width": w,
        "height": h,
        "contour": contour,
        "baseline": {"P1": list(ai_res["P1"]), "P2": list(ai_res["P2"])},
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": "1.1.0", "model_loaded": get_model().seg_session is not None,
            "laterality_loaded": get_laterality_model().session is not None}


@app.get("/api/samples")
def samples() -> list[dict]:
    return [{"name": n, "url": f"/samples/{n}"} for n in list_samples()]


@app.post("/api/analyze/sample/{name}")
def analyze_sample(name: str) -> dict:
    if name not in list_samples():
        raise HTTPException(status_code=404, detail="サンプル画像が見つかりません。")
    img = cv2.imread(str(SAMPLES_DIR / name), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="画像の読み込みに失敗しました。")
    return analyse(img)


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空のファイルです。")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="画像サイズは 5 MB までです。")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="画像を読み込めませんでした。")
    return analyse(img)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
