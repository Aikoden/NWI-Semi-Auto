import cv2
import numpy as np
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                               QGraphicsLineItem, QWidget, QVBoxLayout, QLabel, 
                               QFrame, QMessageBox, QPushButton)
from PySide6.QtGui import QPixmap, QImage, QPen, QColor, QPolygonF, QPainter, QBrush
from PySide6.QtCore import Qt, QPointF, QLineF, Signal
from algorithms import find_intersections

# ==========================================
# 1. 結果の表示 (HUD・左上)
# ==========================================
class ResultOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents) # マウス操作は下のキャンバスに通す
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
        
        # NWI の値
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
# 2. 操作ボタン (右上)
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
# 3. 計測線の図形
# ==========================================
class EditableLine(QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, color, width=2, line_type="Generic"):
        super().__init__(x1, y1, x2, y2)
        self.line_type = line_type
        pen = QPen(color, width)
        pen.setCapStyle(Qt.RoundCap) # 線の端を丸くする
        self.setPen(pen)
        self.setZValue(10)

# ==========================================
# 4. 画像キャンバス
# ==========================================
class ImageCanvas(QGraphicsView):
    data_updated = Signal(str, dict) # mode, data

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # 内部の状態
        self.curr_path = ""
        self.current_contour_item = None
        self.active_baseline = None
        self.current_line_type = "None" # Baseline, Notch, None
        
        # 描画中の線と画面移動
        self.temp_line = None
        self.start_pt = None
        self.is_panning = False
        self.last_pan_pos = QPointF()

        # キャンバスの表示設定
        self.setBackgroundBrush(QBrush(QColor(45, 45, 45)))
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # キャンバスの上に重ねる表示
        self.overlay = ResultOverlay(self)
        self.controls = FloatingControlPanel(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # HUD を左上に置く
        self.overlay.move(15, 15)
        # 操作パネルを右上に置く
        self.controls.move(self.width() - self.controls.width() - 20, 20)

        # 重ねた表示を最前面に戻す
        self.overlay.raise_()
        self.controls.raise_()

    # --------------------------
    # 画像と AI 解析結果の描画
    # --------------------------
    def load_image(self, path):
        self.curr_path = path
        self.scene.clear()
        self.active_baseline = None
        self.current_contour_item = None
        
        try:
            # 日本語などを含むパスでも読める読み込み方
            file_bytes = np.fromfile(path, dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None: return None

            h, w, ch = img_bgr.shape
            q_img = QImage(img_bgr.data, w, h, ch * w, QImage.Format_BGR888).copy()
            
            # 画像をシーンに追加する
            pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
            self.scene.addItem(pixmap_item)
            
            # シーンの範囲と表示倍率を設定する
            self.setSceneRect(0, 0, w, h)
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

            # 画像を読み込んだら HUD を初期状態に戻す
            self.overlay.update_result(0.0)
            self.overlay.update_prompt("画像読み込み完了。解析を待機中...", "normal")
            
            # 重ねた表示を最前面に出す
            self.overlay.raise_()
            self.controls.raise_()
            
            return img_bgr
        except Exception as e:
            print(f"Canvas Load Error: {e}")
            return None

    def draw_contour(self, pts):
        """ AI が求めた輪郭を緑の線で描く """
        if pts is None: return
        poly = QPolygonF([QPointF(p[0], p[1]) for p in pts])
        self.current_contour_item = self.scene.addPolygon(poly, QPen(QColor(0, 255, 0, 180), 2))
        self.current_contour_item.setZValue(5)

    def setup_ai_baseline(self, ai_res):
        """ AI のベースラインを描き、ユーザーが Notch を引ける状態にする """
        p1 = QPointF(*ai_res['P1'])
        p2 = QPointF(*ai_res['P2'])
        self.active_baseline = self._add_line(p1, p2, "Baseline", QColor(200, 0, 200), width=3)
        self.current_line_type = "Notch" # そのまま Notch を引くモードにする

    def restore_state(self, data):
        """ CSV に保存済みの計測を描き直す """
        self.clear_measurements() # 先に消す

        try:
            # 1. ベースラインを戻す (修正のときに使うので、必ず self.active_baseline に入れる)
            p1 = QPointF(data['P1'][0], data['P1'][1])
            p2 = QPointF(data['P2'][0], data['P2'][1])
            self.active_baseline = self._add_line(p1, p2, "Baseline", QColor(200, 0, 200), width=3)
            
            # 2. Femur の延長線 (水色の破線) を戻す
            if 'Femur_L' in data and 'Femur_R' in data:
                fl = QPointF(data['Femur_L'][0], data['Femur_L'][1])
                fr = QPointF(data['Femur_R'][0], data['Femur_R'][1])
                # 水色の破線を描き直す
                pen = QPen(QColor(0, 255, 255), 3, Qt.DashLine)
                f_item = self.scene.addLine(QLineF(fl, fr), pen)
                f_item.setZValue(8)
                # 注意：ここで直接追加した QGraphicsLineItem にも種類を付けておく。Reset のときに消せるようにするため
                f_item.line_type = "Femur" 
            
            # 3. Notch の線 (橙色) を戻す
            nl = QPointF(data['Notch_L'][0], data['Notch_L'][1])
            nr = QPointF(data['Notch_R'][0], data['Notch_R'][1])
            self._add_line(nl, nr, "Notch", QColor(255, 140, 0), width=4)
            
            # 4. UI の状態を更新する
            self.current_line_type = "None" # 保存済みの表示中は線を引けないようにする
            self.overlay.update_result(data['NWI'])
            self.overlay.update_prompt("履歴データ。修正する場合は「Reset」を押してください。", "success")
            
        except Exception as e:
            print(f"Restore Error: {e}")

    def reset_drawing_state(self):
        # 1. シーンから Notch と Femur の線を消す
        for item in list(self.scene.items()):
            ltype = getattr(item, 'line_type', None)
            if ltype in ["Notch", "Femur"]:
                self.scene.removeItem(item)
        
        # 2. 描画中の状態を戻す
        self.temp_line = None
        self.start_pt = None
        
        # 3. ベースラインがあるか確かめる
        if self.active_baseline:
            # 保存済みの計測を表示している（ベースラインはあるが輪郭はない）ときは、AI を再実行して輪郭を求める
            if self.current_contour_item is None:
                print("保存済みデータの表示中（輪郭なし）のため、AI の再実行を依頼します...")
                self.overlay.update_prompt("修正のため、AI解析を再実行しています...", "normal")

                # main.py に AI の再実行を依頼する
                self.data_updated.emit("REQUEST_AI_RERUN", {})
                return  # ここで抜ける。AI 解析が終わると setup_ai_baseline が呼ばれる

            # 輪郭を表示する
            self.current_contour_item.setVisible(True)

            self.current_line_type = "Notch"
            
            # HUD を初期状態に戻す
            self.overlay.lbl_nwi.setText("NWI: --")
            self.overlay.update_prompt("修正モード：ノッチ幅（橙色）を引き直してください。", "instruction")
            
            # メインウィンドウに計測データを消すよう知らせる
            self.data_updated.emit("RESET", {}) 
            self.setCursor(Qt.CrossCursor)
        else:
            # ベースラインがない
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
                
                # Notch モードでは、線をベースラインと平行に固定する
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

                # それ以外のモードでは、マウスの位置まで線を伸ばす
                self.temp_line.setLine(QLineF(self.start_pt, curr))
                
            except RuntimeError:
                # "Internal C++ object already deleted" のエラーは、線がすでに消されたことを示す。
                # Python 側の参照も消して、同じエラーが繰り返し起きないようにする
                self.temp_line = None

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(e)

    # --------------------------
    # 計算とデータの更新
    # --------------------------
    def handle_draw_finish(self, end_pt): 
        if self.current_line_type == "Notch":
            # 1. ベースラインの向き (AI が描いた紫の線の角度) を求める
            if not self.active_baseline:
                QMessageBox.warning(self, "Error", "AI基線が存在しません。")
                return
                
            b_line = self.active_baseline.line()
            direction = np.array([b_line.dx(), b_line.dy()])
            unit_dir = direction / np.linalg.norm(direction)
            
            # 2. ユーザーが引いた線の両端
            n_p1 = np.array([self.start_pt.x(), self.start_pt.y()])
            n_p2 = np.array([end_pt.x(), end_pt.y()])
            n_len = np.linalg.norm(n_p1 - n_p2)
            mid_p = (n_p1 + n_p2) / 2.0
            
            # 3. 対応する Femur の幅 (延長線と輪郭の交点) を求める
            if self.current_contour_item:
                poly = self.current_contour_item.polygon()
                cnt_pts = np.array([[p.x(), p.y()] for p in poly])
                
                # 両側へ伸ばして、いちばん外側の交点を探す
                ints_p = find_intersections(mid_p, unit_dir, cnt_pts)
                ints_n = find_intersections(mid_p, -unit_dir, cnt_pts)
                
                if ints_p and ints_n:
                    # いちばん外側の交点
                    fp_left = ints_n[-1][1]
                    fp_right = ints_p[-1][1]
                    f_len = np.linalg.norm(fp_right - fp_left)
                    
                    # 延長線 (NWI の分母) を目立つ色で描く
                    self._add_line(QPointF(*fp_left), QPointF(*fp_right), "Femur", QColor(0, 255, 255), width=3)
                else:
                    f_len = n_len # 交点が見つからないときの代わり
                    fp_left, fp_right = n_p1, n_p2
            else:
                f_len = n_len
                fp_left, fp_right = n_p1, n_p2

            # 4. NWI を計算する
            nwi = n_len / f_len if f_len > 0 else 0

            # ユーザーが引いた Notch の線 (橙色) を描く
            self._add_line(self.start_pt, end_pt, "Notch", QColor(255, 140, 0), width=4)
            
            # 5. データを更新する
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
        """ 計測用の線 (EditableLine) だけを消す。current_contour_item は決して消さない """
        for item in list(self.scene.items()):
            # 自作の線のクラスだけを消す
            if isinstance(item, EditableLine):
                if keep_baseline and item.line_type == "Baseline":
                    continue
                self.scene.removeItem(item)
        
        if not keep_baseline:
            self.active_baseline = None