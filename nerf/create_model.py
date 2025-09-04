# ファイル名: create_model.py

import torch

# NeRFモデルのクラス定義（calculate_volume.pyと全く同じものをここに置きます）
class NeRF(torch.nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, pts):
        sigma = self.network(pts)
        return torch.relu(sigma)

# --- ここからがメイン処理 ---
print("--- create_model.py を実行中 ---")

# 1. モデルのインスタンスを作成します
# (実際のプロジェクトでは、ここで何時間もかけてモデルを学習させます)
model = NeRF()
print("モデルのインスタンスを作成しました。")

# 2. モデルを保存し、それが正常か検証します
MODEL_SAVE_PATH = '/home/ubuntu/slocal/libero/ATM/nerf/model_20250725-033742.pth' # この名前でファイルが作られます
print(f"モデルを '{MODEL_SAVE_PATH}' として保存・検証します...")

try:
    # モデルの「重み」だけを保存します
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("  保存が完了しました。")

    # 保存したファイルをすぐに読み込んでみます
    print("  保存したファイルの検証を開始します...")
    verification_model = NeRF()
    state_dict = torch.load(MODEL_SAVE_PATH)
    verification_model.load_state_dict(state_dict)
    
    print(f"✅ 検証成功！ '{MODEL_SAVE_PATH}' は正常なファイルです。")
    print("----------------------------------------------------")

except Exception as e:
    print(f"❌ 検証失敗！エラー内容: {e}")