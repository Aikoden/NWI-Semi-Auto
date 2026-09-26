import cv2
import numpy as np
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                               QGraphicsLineItem, QWidget, QVBoxLayout, QLabel, 
                               QFrame, QMessageBox, QPushButton)
from PySide6.QtGui import QPixmap, QImage, QPen, QColor, QPolygonF, QPainter, QBrush
from PySide6.QtCore import Qt, QPointF, QLineF, Signal
from algorithms import find_intersections

# ==========================================
# 1. 结果悬浮窗 (HUD - 左上角)
# ==========================================
class ResultOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents) # 鼠标事件穿透到画布
        self.setAttribute(Qt.WA_TranslucentBackground) 
        self.setFixedWidth(360) 
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.result_frame = QFrame()
        self.result_frame.setStyleSheet("""
            QFrame { 
                background-color: rgba(30, 30, 30, 240); 
                border: 2px solid #888; 
                border-radius: 12px; 
            }
        """)
        res_layout = QVBoxLayout(self.result_frame)
        res_layout.setContentsMargins(20, 15, 20, 15)
        
        # NWI 标签
        self.lbl_nwi = QLabel("NWI: --")
        self.lbl_nwi.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 45px; background: transparent; border: none;")
        self.lbl_nwi.setAlignment(Qt.AlignCenter)
        
        res_layout.addWidget(self.lbl_nwi)
        layout.addWidget(self.result_frame)

        self.lbl_prompt = QLabel("")
        self.lbl_prompt.setWordWrap(True); self.lbl_prompt.hide()
        layout.addWidget(self.lbl_prompt)
        self.raise_()

    def update_result(self, nwi):
        val_str = nwi if isinstance(nwi, str) else f"{float(nwi):.4f}"
        self.lbl_nwi.setText(f"NWI: {val_str}")
        self.adjustSize()
        self.raise_()
    
    def update_prompt(self, text, state="normal"):
        if not text:
            self.lbl_prompt.hide()
            return
        self.lbl_prompt.setText(text)
        self.lbl_prompt.show()
        colors = {"warning": "#ff4444", "success": "#00C851", "instruction": "#ffbb33", "normal": "#33b5e5"}
        self.lbl_prompt.setStyleSheet(f"""
            color: white; background-color: {colors.get(state, '#333')}; 
            padding: 10px; border-radius: 6px; font-weight: bold; font-size: 14px; border: 1px solid white;
        """)
        self.adjustSize()
        self.raise_()
# ==========================================
# 2. 操作按钮悬浮窗 (右上角)
# ==========================================
class FloatingControlPanel(QFrame):
    save_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(140)
        self.setStyleSheet("""
            QFrame { background-color: rgba(255, 255, 255, 210); border: 1px solid #999; border-radius: 10px; }
            QPushButton { padding: 10px; font-weight: bold; border-radius: 5px; font-size: 13px; margin: 2px; }
            QPushButton#saveBtn { background-color: #28a745; color: white; }
            QPushButton#saveBtn:hover { background-color: #218838; }
            QPushButton#resetBtn { background-color: #6c757d; color: white; }
            QPushButton#resetBtn:hover { background-color: #5a6268; }
        """)
        layout = QVBoxLayout(self)
        self.btn_save = QPushButton("💾 保存 (Space)")
        self.btn_save.setObjectName("saveBtn")
        self.btn_reset = QPushButton("🛠 修正 (Reset)")
        self.btn_reset.setObjectName("resetBtn")
        
        self.btn_save.clicked.connect(self.save_requested.emit)
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_reset)

# ==========================================
# 3. 自定义测量线图形项
# ==========================================
class EditableLine(QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, color, width=2, line_type="Generic"):
        super().__init__(x1, y1, x2, y2)
        self.line_type = line_type
        pen = QPen(color, width)
        pen.setCapStyle(Qt.RoundCap) # 圆角线头更美观
        self.setPen(pen)
        self.setZValue(10)

# ==========================================
# 4. 核心画布类
# ==========================================
class ImageCanvas(QGraphicsView):
    data_updated = Signal(str, dict) # mode, data

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # 内部状态
        self.curr_path = ""
        self.current_contour_item = None
        self.active_baseline = None
        self.current_line_type = "None" # Baseline, Notch, None
        
        # 绘图辅助
        self.temp_line = None
        self.start_pt = None
        self.is_panning = False
        self.last_pan_pos = QPointF()

        # 画布 UI 设置
        self.setBackgroundBrush(QBrush(QColor(45, 45, 45)))
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 悬浮层
        self.overlay = ResultOverlay(self)
        self.controls = FloatingControlPanel(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 确保 HUD 在左上角
        self.overlay.move(15, 15)
        # 确保控制面板在右上角
        self.controls.move(self.width() - self.controls.width() - 20, 20)
        
        # 强制刷新置顶状态
        self.overlay.raise_()
        self.controls.raise_()

    # --------------------------
    # 图像与 AI 渲染逻辑
    # --------------------------
    def load_image(self, path):
        self.curr_path = path
        self.scene.clear()
        self.active_baseline = None
        self.current_contour_item = None
        
        try:
            # 支持日语/中文等特殊路径的读取方式
            file_bytes = np.fromfile(path, dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None: return None

            h, w, ch = img_bgr.shape
            q_img = QImage(img_bgr.data, w, h, ch * w, QImage.Format_BGR888).copy()
            
            # 添加图片到场景
            pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
            self.scene.addItem(pixmap_item)
            
            # 设置场景范围与缩放
            self.setSceneRect(0, 0, w, h)
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

            # 加载图片后重置 HUD
            self.overlay.update_result(0.0)
            self.overlay.update_prompt("画像読み込み完了。解析を待機中...", "normal")
            
            # 强制将悬浮层提到最前方
            self.overlay.raise_()
            self.controls.raise_()
            
            return img_bgr
        except Exception as e:
            print(f"Canvas Load Error: {e}")
            return None

    def draw_contour(self, pts):
        """ 绘制 AI 生成的绿色轮廓线 """
        if pts is None: return
        poly = QPolygonF([QPointF(p[0], p[1]) for p in pts])
        self.current_contour_item = self.scene.addPolygon(poly, QPen(QColor(0, 255, 0, 180), 2))
        self.current_contour_item.setZValue(5)

    def setup_ai_baseline(self, ai_res):
        """ 由 AI 设置基线，并准备接收用户画 Notch """
        p1 = QPointF(*ai_res['P1'])
        p2 = QPointF(*ai_res['P2'])
        self.active_baseline = self._add_line(p1, p2, "Baseline", QColor(200, 0, 200), width=3)
        self.current_line_type = "Notch" # 自动进入诺契绘制模式

    def restore_state(self, data):
        """ 渲染 CSV 历史记录 """
        self.clear_measurements() # 先清空
        
        try:
            # 1. 恢复基线 (必须赋值给 self.active_baseline，修正时要用)
            p1 = QPointF(data['P1'][0], data['P1'][1])
            p2 = QPointF(data['P2'][0], data['P2'][1])
            self.active_baseline = self._add_line(p1, p2, "Baseline", QColor(200, 0, 200), width=3)
            
            # 2. 恢复 Femur 延长线 (青色虚线)
            if 'Femur_L' in data and 'Femur_R' in data:
                fl = QPointF(data['Femur_L'][0], data['Femur_L'][1])
                fr = QPointF(data['Femur_R'][0], data['Femur_R'][1])
                # 重新画出青色虚线
                pen = QPen(QColor(0, 255, 255), 3, Qt.DashLine)
                f_item = self.scene.addLine(QLineF(fl, fr), pen)
                f_item.setZValue(8)
                # 注意：这里我们手动添加的 QGraphicsLineItem 需要标记类型，以便 Reset 时能被删掉
                f_item.line_type = "Femur" 
            
            # 3. 恢复 Notch 线 (橙色)
            nl = QPointF(data['Notch_L'][0], data['Notch_L'][1])
            nr = QPointF(data['Notch_R'][0], data['Notch_R'][1])
            self._add_line(nl, nr, "Notch", QColor(255, 140, 0), width=4)
            
            # 4. 更新 UI 状态
            self.current_line_type = "None" # 历史记录模式下禁止乱画
            self.overlay.update_result(data['NWI'])
            self.overlay.update_prompt("履歴データ。修正する場合は「Reset」を押してください。", "success")
            
        except Exception as e:
            print(f"Restore Error: {e}")

    def reset_drawing_state(self):
        # 1. 遍历场景，删除 Notch 和 Femur 类型的线
        for item in list(self.scene.items()):
            ltype = getattr(item, 'line_type', None)
            if ltype in ["Notch", "Femur"]:
                self.scene.removeItem(item)
        
        # 2. 状态重置
        self.temp_line = None
        self.start_pt = None
        
        # 3. 检查基线是否存在
        if self.active_baseline:
            # 历史记录恢复状态（有基线，无轮廓）时，需要重新运行 AI 取得轮廓
            if self.current_contour_item is None:
                print("检测到历史数据模式（缺失轮廓），请求 AI 重新推理...")
                self.overlay.update_prompt("修正のため、AI解析を再実行しています...", "normal")
                
                # 通知 main.py 重新运行 AI
                self.data_updated.emit("REQUEST_AI_RERUN", {}) 
                return  # 退出函数，等待 AI 跑完自动会调用 setup_ai_baseline

            # 确保轮廓是可见的
            self.current_contour_item.setVisible(True)

            self.current_line_type = "Notch"
            
            # HUD 复位
            self.overlay.lbl_nwi.setText("NWI: --")
            self.overlay.update_prompt("修正モード：ノッチ幅（橙色）を引き直してください。", "instruction")
            
            # 通知主窗口清空数据
            self.data_updated.emit("RESET", {}) 
            self.setCursor(Qt.CrossCursor)
        else:
            # 没有基线
            self.overlay.update_prompt("AI基線データがありません。画像を再読み込みしてください。", "warning")
    # --------------------------
    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.is_panning = True
            self.last_pan_pos = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if e.button() == Qt.LeftButton:
            if self.current_line_type == "None": return
            
            pos = self.mapToScene(e.position().toPoint())
            if not self.temp_line:
                self.start_pt = pos
                self.temp_line = QGraphicsLineItem(QLineF(pos, pos))
                self.temp_line.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
                self.scene.addItem(self.temp_line)
            else:
                end_p = self.temp_line.line().p2()
                self.scene.removeItem(self.temp_line)
                self.temp_line = None
                self.handle_draw_finish(end_p)
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.is_panning:
            delta = e.position() - self.last_pan_pos
            self.last_pan_pos = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self.temp_line:
            try:
                curr = self.mapToScene(e.position().toPoint())
                
                # Notch 模式下自动锁定平行基线方向
                if self.current_line_type == "Notch" and self.active_baseline:
                    b = self.active_baseline.line()
                    vec = np.array([b.dx(), b.dy()])
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        u = vec / norm
                        v = np.array([curr.x() - self.start_pt.x(), curr.y() - self.start_pt.y()])
                        proj_len = np.dot(v, u)
                        proj_pt = QPointF(self.start_pt.x() + proj_len * u[0], self.start_pt.y() + proj_len * u[1])
                        
                        self.temp_line.setLine(QLineF(self.start_pt, proj_pt))
                        return

                # 普通模式更新线段
                self.temp_line.setLine(QLineF(self.start_pt, curr))
                
            except RuntimeError:
                # 如果捕获到 "Internal C++ object already deleted" 错误
                # 说明线段已经被销毁了，我们将 Python 引用也置空，防止再次触发
                self.temp_line = None

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(e)

    # --------------------------
    # 数据计算与更新
    # --------------------------
    def handle_draw_finish(self, end_pt): 
        if self.current_line_type == "Notch":
            # 1. 获取基线方向 (由 AI 自动生成的紫色线提供角度)
            if not self.active_baseline:
                QMessageBox.warning(self, "Error", "AI基線が存在しません。")
                return
                
            b_line = self.active_baseline.line()
            direction = np.array([b_line.dx(), b_line.dy()])
            unit_dir = direction / np.linalg.norm(direction)
            
            # 2. 获取人工画的线端点
            n_p1 = np.array([self.start_pt.x(), self.start_pt.y()])
            n_p2 = np.array([end_pt.x(), end_pt.y()])
            n_len = np.linalg.norm(n_p1 - n_p2)
            mid_p = (n_p1 + n_p2) / 2.0
            
            # 3. 寻找对应的 Femur 宽度 (延长线交点)
            if self.current_contour_item:
                poly = self.current_contour_item.polygon()
                cnt_pts = np.array([[p.x(), p.y()] for p in poly])
                
                # 向两侧寻找最外侧交点
                ints_p = find_intersections(mid_p, unit_dir, cnt_pts)
                ints_n = find_intersections(mid_p, -unit_dir, cnt_pts)
                
                if ints_p and ints_n:
                    # 获取最外圈的交点
                    fp_left = ints_n[-1][1]
                    fp_right = ints_p[-1][1]
                    f_len = np.linalg.norm(fp_right - fp_left)
                    
                    # 绘制显眼的延长线 (分母)
                    self._add_line(QPointF(*fp_left), QPointF(*fp_right), "Femur", QColor(0, 255, 255), width=3)
                else:
                    f_len = n_len # 回退方案
                    fp_left, fp_right = n_p1, n_p2
            else:
                f_len = n_len
                fp_left, fp_right = n_p1, n_p2

            # 4. 计算 NWI
            nwi = n_len / f_len if f_len > 0 else 0
            
            # 绘制人工确认的 Notch 线 (橙色)
            self._add_line(self.start_pt, end_pt, "Notch", QColor(255, 140, 0), width=4)
            
            # 5. 更新数据
            data = {
                "NWI": nwi, "Femur_Px": f_len, "Notch_Px": n_len,
                "P1": (b_line.p1().x(), b_line.p1().y()), 
                "P2": (b_line.p2().x(), b_line.p2().y()),
                "Femur_L": (float(fp_left[0]), float(fp_left[1])),
                "Femur_R": (float(fp_right[0]), float(fp_right[1])),
                "Notch_L": (float(n_p1[0]), float(n_p1[1])),
                "Notch_R": (float(n_p2[0]), float(n_p2[1])),
            }
            
            self.overlay.update_result(nwi)
            self.overlay.update_prompt("計算完了。保存してください。", "success")
            self.data_updated.emit("AI_SEMI", data)
            self.current_line_type = "None"

    def _add_line(self, p1, p2, ltype, color, width=2):
        item = EditableLine(p1.x(), p1.y(), p2.x(), p2.y(), color, width, ltype)
        self.scene.addItem(item)
        return item

    def clear_measurements(self, keep_baseline=False):
        """ 只清理测量用的线条 (EditableLine)，绝对不要删除 current_contour_item """
        for item in list(self.scene.items()):
            # 仅删除我们自定义的线条类
            if isinstance(item, EditableLine):
                if keep_baseline and item.line_type == "Baseline":
                    continue
                self.scene.removeItem(item)
        
        if not keep_baseline:
            self.active_baseline = None