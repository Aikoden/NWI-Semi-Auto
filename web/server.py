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
    find_knee_split,
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
    version="1.2.0",
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

    # 両膝が写っていれば 2 つの大腿骨の間で左右に分け、片膝ずつ処理する。
    # 推論は画像全体の 1 回だけで、そのマスクを分けて使う。
    h, w = img_bgr.shape[:2]
    split = find_knee_split(mask)
    regions = ([("single", 0, w)] if split is None
               else [("screen_left", 0, split), ("screen_right", split, w)])
    knees = [analyse_knee(img_bgr[:, x0:x1], mask[:, x0:x1], position, x0)
             for position, x0, x1 in regions]

    # 両膝の正面像では腓骨は両側とも外側にあるので、画面左の膝は right、
    # 画面右の膝は left になるはず。それ以外の組み合わせは採用しない。
    sides = [k["laterality"]["side"] for k in knees]
    if len(knees) == 2 and None not in sides and sides != ["right", "left"]:
        for k in knees:
            k["laterality"] = {**k["laterality"], "side": None, "status": "inconsistent"}

    return {
        # 旧版の画面 (ブラウザのキャッシュ) 向けに、1 膝目は従来の形でも返す
        "laterality": knees[0]["laterality"],
        "contour": knees[0]["contour"],
        "baseline": knees[0]["baseline"],
        "knees": knees,
        "width": w,
        "height": h,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


KNEE_NAMES = {"screen_left": "画面左の膝", "screen_right": "画面右の膝"}


def analyse_knee(img_bgr: np.ndarray, mask: np.ndarray, position: str, x0: int) -> dict:
    """片膝ぶんの輪郭・ベースライン・左右候補。座標は元画像の座標に戻して返す。"""
    prefix = f"{KNEE_NAMES[position]}: " if position in KNEE_NAMES else ""
    cnt_int, cnt_float = full_postprocessing_pipeline(mask)
    if cnt_int is None:
        raise HTTPException(status_code=422, detail=f"{prefix}AI解析に失敗しました（輪郭検出不可）。")

    ai_res = detect_baseline(cnt_float)
    if not ai_res:
        raise HTTPException(status_code=422, detail=f"{prefix}AI解析に失敗しました（基線を算出できません）。")

    return {
        "position": position,
        "region": [x0, x0 + img_bgr.shape[1]],
        "laterality": get_laterality_model().predict(img_bgr, mask),
        "contour": np.round(cnt_float[::CONTOUR_STRIDE] + [x0, 0], 2).tolist(),
        "baseline": {k: [float(ai_res[k][0]) + x0, float(ai_res[k][1])] for k in ("P1", "P2")},
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": "1.2.0", "model_loaded": get_model().seg_session is not None,
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
    # 更新後に古い画面が残らないよう、毎回サーバに確認させる (静的ファイルは ?v= で切り替える)
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
