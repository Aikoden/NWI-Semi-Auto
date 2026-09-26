import os
import sys
import re
import csv
import datetime

# 全局配置
CSV_HEADERS = [
    "ImageName", "Timestamp", "Mode", "NWI",
    "Femur_Px", "Notch_Px",
    "P1_x", "P1_y", "P2_x", "P2_y",
    "Femur_L_x", "Femur_L_y", "Femur_R_x", "Femur_R_y",
    "Notch_L_x", "Notch_L_y", "Notch_R_x", "Notch_R_y"
]

def log(msg):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")


def resource_path(relative_path):
    """
    智能路径查找：
    1. 优先检查 internal 目录 (专门针对 model.onnx)
    2. 只有 internal 里没有，才去根目录找 (针对 samples)
    """
    if getattr(sys, 'frozen', False):
        # 【打包后】
        base_path = os.path.dirname(sys.executable)
        
        # --- 路径分流逻辑 ---
        
        # 1. 尝试拼接 internal 路径
        internal_path = os.path.join(base_path, "internal", relative_path)
        if os.path.exists(internal_path):
            return internal_path
            
        # 2. 如果 internal 里没有，就假设在 EXE 旁边
        return os.path.join(base_path, relative_path)
        
    else:
        # 【开发环境】
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
class VersioningManager:
    def __init__(self, results_dir="Results"):
        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.current_csv = self._find_latest_csv()

    def _find_latest_csv(self):
        try:
            files = [f for f in os.listdir(self.results_dir) if f.startswith("nwi_results_v") and f.endswith(".csv")]
            if not files:
                return os.path.join(self.results_dir, "nwi_results_v1.csv")
            
            versions = []
            for f in files:
                match = re.search(r'v(\d+)', f)
                if match: versions.append(int(match.group(1)))
            
            max_v = max(versions) if versions else 1
            return os.path.join(self.results_dir, f"nwi_results_v{max_v}.csv")
        except:
            return os.path.join(self.results_dir, "nwi_results_v1.csv")

    def create_new_version(self):
        match = re.search(r'v(\d+)', os.path.basename(self.current_csv))
        curr_v = int(match.group(1)) if match else 1
        new_v = curr_v + 1
        self.current_csv = os.path.join(self.results_dir, f"nwi_results_v{new_v}.csv")
        return self.current_csv

    def load_session_data(self):
        data_map = {}
        if not os.path.exists(self.current_csv):
            return data_map
        try:
            with open(self.current_csv, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row["ImageName"]
                    data_map[name] = {
                        "NWI": float(row["NWI"]),
                        "P1": (float(row["P1_x"]), float(row["P1_y"])),
                        "P2": (float(row["P2_x"]), float(row["P2_y"])),
                        "Femur_L": (float(row["Femur_L_x"]), float(row["Femur_L_y"])),
                        "Femur_R": (float(row["Femur_R_x"]), float(row["Femur_R_y"])),
                        "Notch_L": (float(row["Notch_L_x"]), float(row["Notch_L_y"])),
                        "Notch_R": (float(row["Notch_R_x"]), float(row["Notch_R_y"])),
                        "Femur_Px": float(row["Femur_Px"]),
                        "Notch_Px": float(row["Notch_Px"])
                    }
        except Exception as e:
            print(f"Session Load Error: {e}")
        return data_map