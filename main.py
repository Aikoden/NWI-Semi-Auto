import sys
import os
import csv
import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QListWidget, QListWidgetItem, QTableWidget, 
                               QTableWidgetItem, QHeaderView, QMessageBox, QLabel, QDialog, 
                               QFrame, QAbstractItemView)
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, QTimer

# 自作のコンポーネントとユーティリティ
from ui_components import ImageCanvas
from utils import resource_path, VersioningManager, CSV_HEADERS
from algorithms import AIModel, full_postprocessing_pipeline, detect_baseline

# ==========================================
# 起動時の案内画面
# ==========================================
class WelcomeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NWI解析ツール")
        self.setFixedWidth(500)        # 幅だけ固定し、高さは内容に合わせる
        self.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(self)
        
        title = QLabel("NWI計測支援システム")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #007bff; margin-top: 20px;")
        
        info = QLabel(
            "<h3>操作ガイド:</h3>"
            "<ul>"
            "<li><b>プレビュー:</b> 画像を選択するとまずプレビューが表示されます。</li>"
            "<li><b>解析開始:</b> 画面中央のボタンまたは Enter キーで AI 解析を開始します。</li>"
            "<li><b>計測:</b> 自動生成された基線を元に、ノッチ幅を引いてください。</li>"
            "<li><b>保存:</b> 結果を保存し、次へ進みます。</li>"
            "</ul>"
        )
        info.setWordWrap(True) 
        info.setStyleSheet("font-size: 14px; margin: 10px;")
        
        btn = QPushButton("開始する")
        btn.setStyleSheet("background-color: #28a745; color: white; padding: 12px; font-weight: bold; border-radius: 5px;")
        btn.clicked.connect(self.accept)
        
        layout.addWidget(title)
        layout.addWidget(info)
        layout.addWidget(btn)

# ==========================================
# 保存完了ダイアログ
# ==========================================
class PostSaveDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("保存完了")
        self.setFixedSize(300, 150)
        layout = QVBoxLayout(self)
        msg = QLabel("✅ 保存しました")
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet("font-size: 18px; font-weight: bold; color: #28a745;")
        
        btn = QPushButton("次へ (Next)")
        btn.setStyleSheet("background-color: #007bff; color: white; padding: 10px; border-radius: 5px;")
        btn.clicked.connect(self.accept)
        
        layout.addWidget(msg)
        layout.addWidget(btn)

# ==========================================
# メインウィンドウ
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1400, 900)
        self.setWindowTitle("NWI Analysis Workstation v2025 (AI Support Only)")
        
        # 1. データとモデルの初期化
        self.vm = VersioningManager()
        self.session_data = self.vm.load_session_data()
        self.ai = AIModel() 
        self.app_state = "IDLE" # 状態: IDLE, PREVIEW, MEASURED

        # 2. UI のレイアウト
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # --- 左：画像リスト ---
        left_panel = QFrame()
        left_panel.setFixedWidth(280)
        l_layout = QVBoxLayout(left_panel)
        
        self.btn_retest = QPushButton("🔄 全件再計測(New Round)")
        self.btn_retest.setStyleSheet("""
            QPushButton { background-color: #dc3545; color: white; padding: 12px; font-weight: bold; border-radius: 5px; }
            QPushButton:hover { background-color: #c82333; }
        """)
        self.btn_retest.clicked.connect(self.on_retest_all)
        
        self.flist = QListWidget()
        self.flist.itemClicked.connect(self.on_file_click)
        
        l_layout.addWidget(QLabel("<b>計測コントロール</b>"))
        l_layout.addWidget(self.btn_retest)
        l_layout.addWidget(QLabel("<b>画像リスト (samples)</b>"))
        l_layout.addWidget(self.flist)
        
        # --- 中央：画像キャンバス ---
        self.canvas = ImageCanvas()

        # 計測開始の大きいボタン (キャンバスの上に重ねる)
        self.btn_start = QPushButton("▶ 計測開始 (Enter)", self.canvas)
        self.btn_start.setGeometry(0, 0, 220, 60) # 位置は on_file_click で計算する
        self.btn_start.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; font-size: 20px; font-weight: bold; border-radius: 10px; border: 2px solid white; }
            QPushButton:hover { background-color: #218838; }
        """)
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_start.hide()

        # 【重要：シグナルの接続】
        self.canvas.data_updated.connect(self.on_data_updated)
        self.canvas.controls.save_requested.connect(self.save_current_data)
        self.canvas.controls.reset_requested.connect(self.on_reset_clicked)

        # --- 右：計測データの表 ---
        right_panel = QFrame()
        right_panel.setFixedWidth(260)
        r_layout = QVBoxLayout(right_panel)
        
        self.table = QTableWidget(0, 2)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setHorizontalHeaderLabels(["項目", "値"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers) # 表は手で編集させない
        
        r_layout.addWidget(QLabel("<b>今回の計測データ</b>"))
        r_layout.addWidget(self.table)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(right_panel)
        
        # 変数の初期化
        self.curr_file_path = ""
        self.curr_data = None
        self.current_img_bgr = None # AI 解析まで画像を保持しておく

        # 画像リストを読み込む
        self.refresh_file_list()

        # 案内ダイアログ
        QTimer.singleShot(500, lambda: WelcomeDialog(self).exec())

    # --------------------------
    # 計測の処理
    # --------------------------

    def refresh_file_list(self):
        """ samples フォルダの画像を並べ、CSV に保存済みの画像には印を付ける """
        self.flist.clear()
        samples_dir = resource_path("samples")
        # ====================================================
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        for f in sorted(os.listdir(samples_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                full_path = os.path.join(samples_dir, f)
                item = QListWidgetItem(f)
                item.setData(Qt.UserRole, full_path)
                
                if f in self.session_data:
                    item.setBackground(QColor("#d4edda"))
                    item.setText(f"✔ {f}")
                
                self.flist.addItem(item)

    def on_file_click(self, item):
        """ 段階 1：リストで選んだ画像をプレビューする（AI はまだ実行しない） """
        self.curr_file_path = item.data(Qt.UserRole)
        filename = os.path.basename(self.curr_file_path)

        # 1. 前の画像の状態を消す
        self.curr_data = None
        self.table.setRowCount(0)
        self.current_img_bgr = None
        
        # 2. 画像を読み込む
        img_bgr = self.canvas.load_image(self.curr_file_path)
        if img_bgr is None:
            QMessageBox.critical(self, "Error", "画像の読み込みに失敗しました。")
            return
        
        self.current_img_bgr = img_bgr # 画像を保持しておく

        # 3. 保存済みなら前回の線を表示し、未計測ならプレビューにする
        if filename in self.session_data:
            # 保存済み：前回の線を表示する
            data = self.session_data[filename]
            self.canvas.restore_state(data)
            self.update_table_display(data)
            self.curr_data = data
            self.app_state = "MEASURED"
            self.btn_start.hide()
            self.canvas.overlay.update_prompt("履歴データを表示中。修正可能です。", "success")
        else:
            # 未計測：プレビュー
            self.app_state = "PREVIEW"
            self.btn_start.show()
            self.btn_start.raise_()
            # ボタンを中央に置く
            w, h = self.canvas.width(), self.canvas.height()
            if w > 0 and h > 0:
                self.btn_start.move((w - 220)//2, (h - 60)//2)
            
            self.canvas.overlay.update_prompt("プレビュー中。計測を開始するにはボタンを押してください。", "instruction")

    def on_start_clicked(self):
        """ 段階 2：ボタンが押されたら AI 解析を実行する """
        if self.app_state != "PREVIEW" or self.current_img_bgr is None: return
        
        self.btn_start.hide()
        success = self.run_ai_logic(self.current_img_bgr)
        
        if success:
            self.app_state = "MEASURED"
        else:
            # 失敗したらプレビューのままにし、やり直しや次の画像へ進めるようにする
            self.app_state = "PREVIEW"

    def run_ai_logic(self, img_bgr):
        """ AI 解析: セグメンテーション -> 後処理 -> ベースライン検出 """
        self.canvas.overlay.update_prompt("AI解析を実行中...", "normal")
        QApplication.processEvents()

        # 1. 大腿骨のマスクを推論する
        mask = self.ai.predict_mask(img_bgr)
        
        if mask is not None:
            # 2. 後処理 (くっついた部分の切り離し + スプライン近似)
            cnt_int, cnt_float = full_postprocessing_pipeline(mask)
            
            if cnt_int is not None:
                # 3. 輪郭を緑で描く
                self.canvas.draw_contour(cnt_float)
                
                # 4. ベースラインを求める (幾何計算)
                ai_res = detect_baseline(cnt_float)
                
                if ai_res:
                    self.canvas.setup_ai_baseline(ai_res)
                    self.canvas.overlay.update_result(0.0)
                    self.canvas.overlay.update_prompt("AI解析完了。ノッチ幅（橙色）を引いてください。", "success")
                    return True

        # AI 解析の失敗
        self.canvas.current_line_type = "None"
        self.canvas.overlay.update_prompt("AI解析に失敗しました（輪郭検出不可）。", "warning")
        self.table.setRowCount(0)
        return False

    def on_reset_clicked(self):
        """ 「修正」ボタンの処理 """
        self.curr_data = None
        self.table.setRowCount(0)
        self.canvas.reset_drawing_state()
        self.app_state = "MEASURED" # 修正中も計測済みの状態のままにする

    def on_data_updated(self, mode, data):
        """ キャンバスから届いたデータの処理 """

        # 1. 通常のリセット (輪郭が表示されているとき。計測の直後に Reset を押した場合など)
        if mode == "RESET":
            self.curr_data = None
            self.table.setRowCount(0)
            
        # ==========================================================
        # 2. AI 解析の再実行の依頼 (保存済みデータを修正するとき)
        # ==========================================================
        elif mode == "REQUEST_AI_RERUN":
            print("保存済みデータの修正：AI 解析を再実行しています...")

            # 画像が読み込まれているか確かめる
            if self.current_img_bgr is not None:
                # on_start_clicked と同じ AI 解析の処理を使う
                self.run_ai_logic(self.current_img_bgr)
            else:
                QMessageBox.warning(self, "Warning", "画像データがありません。")

        # 3. 通常のデータ更新 (線を引いたときに届くデータ)
        else:
            self.curr_data = data
            self.update_table_display(data)

    def update_table_display(self, data):
        self.table.setRowCount(0)
        if not data: return
        display_map = [("NWI", "NWI (比率)"), ("Femur_Px", "大腿骨幅 (px)"), ("Notch_Px", "ノッチ幅 (px)")]
        for key, label in display_map:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(label))
            val = data.get(key, 0)
            fmt = "{:.4f}" if key == "NWI" else "{:.2f}"
            self.table.setItem(row, 1, QTableWidgetItem(fmt.format(val)))

    def save_current_data(self):
        """ 計測データを CSV に保存する """
        if not self.curr_data or self.curr_data.get('NWI', 0) == 0:
            QMessageBox.warning(self, "Warning", "計測データがありません。")
            return
        
        filename = os.path.basename(self.curr_file_path)
        csv_file = self.vm.current_csv
        try:
            with open(csv_file, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if f.tell() == 0: writer.writerow(CSV_HEADERS)
                
                d = self.curr_data
                writer.writerow([
                    filename, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "AI_SEMI", f"{d.get('NWI', 0):.5f}",
                    f"{d.get('Femur_Px', 0):.2f}", f"{d.get('Notch_Px', 0):.2f}",
                    d['P1'][0], d['P1'][1], d['P2'][0], d['P2'][1],
                    d['Femur_L'][0], d['Femur_L'][1], d['Femur_R'][0], d['Femur_R'][1],
                    d['Notch_L'][0], d['Notch_L'][1], d['Notch_R'][0], d['Notch_R'][1]
                ])
            
            self.session_data[filename] = self.curr_data
            it = self.flist.currentItem()
            if it:
                it.setBackground(QColor("#d4edda"))
                if "✔" not in it.text(): it.setText(f"✔ {it.text()}")
            
            if PostSaveDialog(self).exec() == QDialog.Accepted:
                self.load_next_image()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"保存失敗: {e}")

    def load_next_image(self):
        idx = self.flist.currentRow()
        if idx < self.flist.count() - 1:
            self.flist.setCurrentRow(idx + 1)
            self.on_file_click(self.flist.currentItem())

    def on_retest_all(self):
        res = QMessageBox.question(self, "確認", "新しい計測ラウンドを開始しますか？")
        if res == QMessageBox.Yes:
            self.vm.create_new_version()
            self.session_data = {}
            self.refresh_file_list()
            self.canvas.scene.clear()
            self.table.setRowCount(0)
            self.app_state = "IDLE"

    def keyPressEvent(self, e):
        # キーボードショートカット
        if e.key() == Qt.Key_Return or e.key() == Qt.Key_Enter:
            if self.app_state == "PREVIEW":
                self.on_start_clicked() # Enter で AI 解析を開始
        elif e.key() == Qt.Key_Space:
            if self.app_state == "MEASURED":
                self.save_current_data() # Space で保存

    def resizeEvent(self, event):
        # ウィンドウの大きさが変わっても、開始ボタンを中央に保つ
        super().resizeEvent(event)
        if self.btn_start.isVisible():
            w, h = self.canvas.width(), self.canvas.height()
            self.btn_start.move((w - 220)//2, (h - 60)//2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Yu Gothic UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())