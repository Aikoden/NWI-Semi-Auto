import os
import traceback

import cv2
import numpy as np
import onnxruntime as ort
from scipy import interpolate

from utils import log, resource_path

# ==================== 几何基础 ====================

def line_segment_intersection(p1, direction, q1, q2):
    p1 = np.asarray(p1, dtype=float)
    direction = np.asarray(direction, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    segment_vec = q2 - q1
    delta_p = q1 - p1
    try:
        cross_dir_seg = direction[0] * segment_vec[1] - direction[1] * segment_vec[0]
        if abs(cross_dir_seg) < 1e-9: return None
        t = (delta_p[0] * segment_vec[1] - delta_p[1] * segment_vec[0]) / cross_dir_seg
        u = (delta_p[0] * direction[1] - delta_p[1] * direction[0]) / cross_dir_seg
        if -1e-7 <= u <= 1.0 + 1e-7: return p1 + t * direction
    except: return None
    return None

def find_intersections(start_point, direction, contour_points, max_dist=float('inf')):
    if start_point is None or direction is None or contour_points is None: return []
    intersections = []
    start_point_arr = np.array(start_point, dtype=float)
    direction_arr = np.array(direction, dtype=float)
    contour_arr = np.squeeze(np.array(contour_points, dtype=float))
    if contour_arr.ndim < 2: return []
    norm_direction = np.linalg.norm(direction_arr)
    if norm_direction < 1e-9: return []
    direction_norm = direction_arr / norm_direction

    for i in range(len(contour_arr)):
        q1 = contour_arr[i]
        q2 = contour_arr[(i + 1) % len(contour_arr)]
        intersection = line_segment_intersection(start_point_arr, direction_norm, q1, q2)
        if intersection is not None:
            dist_vec = intersection - start_point_arr
            dist = np.linalg.norm(dist_vec)
            dot_product = np.dot(dist_vec, direction_norm)
            if dist >= 1.0 and dot_product >= -1e-6 and dist <= max_dist:
                intersections.append((dist, intersection))
    intersections.sort(key=lambda x: x[0])
    return intersections

# ==================== Mask 处理与拟合 ====================

def full_postprocessing_pipeline(mask_np_uint8, smoothing=0.003):
    """
    股骨 Mask 后处理:
    1. Mask 清理: 最大连通域 -> Open(11) -> Close(11)
    2. 样条拟合: 归一化 -> splprep(s=0.003) -> 8 倍上采样
    返回 (整数轮廓 (N,1,2), 浮点轮廓 (N,2))，失败时返回 (None, None)
    """
    try:
        if mask_np_uint8 is None or np.sum(mask_np_uint8) == 0:
            return None, None

        mask = mask_np_uint8.astype(np.uint8)

        # 1.1 保留最大连通域并填充
        contours_raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_raw: return None, None

        max_cnt = max(contours_raw, key=cv2.contourArea)
        if cv2.contourArea(max_cnt) < 100: return None, None

        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [max_cnt], -1, 255, thickness=cv2.FILLED)

        # 1.2 形态学处理: 先 Open (断开粘连)，后 Close (填补空洞)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        # 2.1 形态学处理后再次提取最大连通域
        num_labels, labels_im = cv2.connectedComponents(clean_mask)
        if num_labels < 2: return None, None

        largest_label = np.argmax(np.bincount(labels_im.flat)[1:]) + 1
        final_mask_for_spline = np.zeros_like(clean_mask)
        final_mask_for_spline[labels_im == largest_label] = 255

        # 2.2 提取轮廓点
        contours_final, _ = cv2.findContours(final_mask_for_spline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours_final: return None, None

        initial_contour_cv = max(contours_final, key=cv2.contourArea)
        contour_points_sq = np.squeeze(initial_contour_cv)
        if contour_points_sq.ndim == 1:
             contour_points_sq = contour_points_sq.reshape(-1, 2)

        # 2.3 点去重
        if len(contour_points_sq) < 4: return None, None
        x, y = contour_points_sq[:, 0].astype(float), contour_points_sq[:, 1].astype(float)

        dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        valid_indices = np.concatenate(([True], dist > 1e-6))
        x, y = x[valid_indices], y[valid_indices]

        if len(x) < 4: return None, None

        # 2.4 归一化
        x_min, y_min, w, h = cv2.boundingRect(initial_contour_cv)
        if w == 0 or h == 0: return None, None

        x_norm = (x - x_min) / w
        y_norm = (y - y_min) / h

        # 2.5 样条拟合
        k_val = 3 if len(x) > 3 else len(x) - 1
        tck_norm_list, _ = interpolate.splprep([x_norm, y_norm], s=smoothing, k=k_val, per=True)[:2]
        tck_norm_tuple = tuple(tck_norm_list)

        # 2.6 插值评估 (8 倍上采样)
        num_points_multiplier = 8
        num_final_points = len(x) * num_points_multiplier
        u_eval = np.linspace(0, 1, num_final_points, endpoint=True)

        x_smooth_norm, y_smooth_norm = interpolate.splev(u_eval, tck_norm_tuple, der=0)

        # 2.7 还原坐标
        x_smooth = x_smooth_norm * w + x_min
        y_smooth = y_smooth_norm * h + y_min

        final_contour_points_float = np.vstack((x_smooth, y_smooth)).T
        final_contour_cv_int = final_contour_points_float.reshape(-1, 1, 2).astype(np.int32)

        return final_contour_cv_int, final_contour_points_float

    except Exception as e:
        log(f"Pipeline Error: {e}")
        traceback.print_exc()
        return None, None

# ==================== 基线检测 ====================

def detect_baseline(spline_contour_float):
    """
    从股骨轮廓求基线 (P1, P2):
    1. 用凸包缺损找到 Notch (轮廓下方 30% 内、越靠中央权重越高)
    2. 分别取 Notch 左右两侧轮廓的最低点作为 P1, P2
    返回 {"P1": (x, y), "P2": (x, y)}，失败时返回 None
    """
    if spline_contour_float is None or len(spline_contour_float) < 20:
        return None

    try:
        spline_int = np.array(spline_contour_float, dtype=np.int32).reshape(-1, 1, 2)
        x_coords = spline_contour_float[:, 0]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        total_width = max_x - min_x

        # 1. 寻找 Notch
        hull_indices = cv2.convexHull(spline_int, returnPoints=False).astype(np.int32)
        defects = cv2.convexityDefects(spline_int, hull_indices)
        max_y = np.max(spline_contour_float[:, 1])
        bottom_thresh = max_y - (max_y - np.min(spline_contour_float[:, 1])) * 0.30

        best_defect = {'idx': -1, 'score': -1}
        if defects is not None:
            for s, e, f, d in defects.reshape(-1, 4):
                pt = spline_contour_float[f]
                if pt[1] < bottom_thresh: continue
                depth = d / 256.0
                rel_x = (pt[0] - min_x) / (total_width + 1e-9)
                center_w = 1.0 - 4.0 * ((rel_x - 0.5) ** 2)
                score = depth * max(0.1, center_w)
                if score > best_defect['score']:
                    best_defect['score'] = score; best_defect['idx'] = f

        notch_idx = best_defect['idx'] if best_defect['idx'] != -1 else np.argmax(spline_contour_float[:, 1])
        notch_pt = spline_contour_float[notch_idx]

        # 2. 寻找 P1, P2
        left_mask = spline_contour_float[:, 0] < notch_pt[0]
        right_mask = spline_contour_float[:, 0] > notch_pt[0]

        def get_lowest(mask):
            idxs = np.where(mask)[0]
            if len(idxs) == 0: return -1
            y_vals = spline_contour_float[idxs, 1]
            mx_y = np.max(y_vals)
            cands = idxs[y_vals >= (mx_y - 1.5)]
            return cands[len(cands)//2]

        p1_idx, p2_idx = get_lowest(left_mask), get_lowest(right_mask)
        if p1_idx == -1 or p2_idx == -1: return None
        if spline_contour_float[p1_idx, 0] > spline_contour_float[p2_idx, 0]: p1_idx, p2_idx = p2_idx, p1_idx
        P1, P2 = spline_contour_float[p1_idx], spline_contour_float[p2_idx]

        return {
            "P1": (float(P1[0]), float(P1[1])),
            "P2": (float(P2[0]), float(P2[1])),
        }
    except Exception:
        traceback.print_exc()
        return None

# ==================== AI 模型类 ====================

class AIModel:
    def __init__(self):
        self.seg_session = None

        # resource_path 会自动去 internal 文件夹里找模型；也可以用环境变量 NWI_MODEL_PATH 指定
        seg_p = os.environ.get("NWI_MODEL_PATH") or resource_path("model_b5_384.onnx")

        if os.path.exists(seg_p):
            try:
                self.seg_session = ort.InferenceSession(seg_p, providers=['CPUExecutionProvider'])
                log(f"Seg Model Loaded Success: {seg_p}")
            except Exception as e:
                # 加载出错（比如库版本不对），只记录日志，不弹窗干扰用户
                log(f"Seg Model Load Failed: {e}")
                self.seg_session = None
        else:
            log(f"Seg Model Not Found at: {seg_p}")
            self.seg_session = None

    def predict_mask(self, img_bgr):
        """
        股骨分割:
        1. 2x TTA (原图 + 水平翻转) -> 减少边缘模糊
        2. Argmax (竞争分类) -> 断开股骨/胫骨粘连
        3. Resize Nearest -> 保持边缘硬度
        """
        if self.seg_session is None or img_bgr is None: return None
        try:
            h_orig, w_orig = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            INPUT_SIZE = 512  # 模型输入尺寸

            # 1. 预处理 (ImageNet 均值/标准差归一化)
            def preprocess(image):
                inp = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                inp = (inp - mean) / std
                return inp.transpose(2, 0, 1)[np.newaxis, ...]

            # 2. TTA: 原图 + 水平翻转
            inp_normal = preprocess(img_rgb)
            inp_fliplr = preprocess(cv2.flip(img_rgb, 1))
            inp_batch = np.concatenate([inp_normal, inp_fliplr], axis=0)

            # 3. 推理 (ONNX)，输出 shape: (2, Classes, H, W)
            input_name = self.seg_session.get_inputs()[0].name
            preds = self.seg_session.run(None, {input_name: inp_batch})[0]

            # 4. 还原水平翻转并平均融合
            pred_normal = preds[0]
            pred_fliplr_back = cv2.flip(preds[1].transpose(1, 2, 0), 1).transpose(2, 0, 1)
            pred_avg = (pred_normal + pred_fliplr_back) / 2.0

            # 5. Argmax 竞争分类，通道顺序: [0:背景, 1:股骨, 2:胫骨]
            #    股骨和胫骨互斥，不会重叠
            mask_idx_small = np.argmax(pred_avg, axis=0).astype(np.uint8)

            # 6. 用最邻近插值还原到原图尺寸，保持边缘锐利
            full_mask = cv2.resize(mask_idx_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # 7. 提取股骨
            FEMUR_CLASS = 1
            res_mask = np.zeros_like(full_mask)
            res_mask[full_mask == FEMUR_CLASS] = 255

            return res_mask

        except Exception as e:
            log(f"Inference Error: {e}")
            traceback.print_exc()
            return None
