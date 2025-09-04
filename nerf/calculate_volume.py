import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

# (日本語フォント設定などは変更なし)
mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# (NeRFクラス定義、CSV保存関数、可視化関数は変更なし)
class NeRF(torch.nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(3, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    def forward(self, pts):
        return torch.relu(self.network(pts))

def save_points_to_csv(filepath, points_tensor, densities_tensor):
    print(f"\n💾 表面の点群データをCSVファイルに保存しています...")
    try:
        points_np = points_tensor.cpu().numpy()
        densities_np = densities_tensor.cpu().numpy().reshape(-1, 1)
        data_to_save = np.hstack((points_np, densities_np))
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'density'])
            writer.writerows(data_to_save)
        print(f"  完了しました。データを '{filepath}' に保存しました。")
    except Exception as e:
        print(f"  ❌ CSVファイルの保存中にエラーが発生しました: {e}")

def visualize_points_with_density(points_tensor, densities_tensor, max_points_to_display=50000):
    print(f"\n🎨 密度による色分けプロットの準備をしています...")
    # (この関数の中身は変更なし)
    points_np = points_tensor.cpu().numpy()
    densities_np = densities_tensor.cpu().numpy()
    
    num_points = points_np.shape[0]
    if num_points > max_points_to_display:
        indices = np.random.choice(num_points, max_points_to_display, replace=False)
        points_np = points_np[indices]
        densities_np = densities_np[indices]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        points_np[:, 0], points_np[:, 1], points_np[:, 2], 
        c=densities_np, cmap='viridis', s=1, alpha=0.6
    )
    ax.set_xlabel('X 軸'); ax.set_ylabel('Y 軸'); ax.set_zlabel('Z 軸')
    ax.set_title('物体表面の密度(σ)の3Dプロット')
    fig.colorbar(scatter, ax=ax, label='推定された密度 (σ)', shrink=0.6)
    ax.set_aspect('equal', adjustable='box')
    
    plt.savefig('shape_density_3d.png')
    print("  プロットを 'shape_density_3d.png' に保存しました。")
    plt.show()

# ----------------------------------------------------------------------------
# 2. メインの処理関数
# ----------------------------------------------------------------------------
def calculate_volume_from_nerf(
    model_path, resolution, bbox_min, bbox_max, density_threshold,
    batch_size, visualize, scene_scale_meters, save_csv, csv_output_path,
    object_physical_density # ★★★ 追加: オブジェクトの物理密度
):
    # (...(中略) モデル読み込み、グリッド生成、NeRF密度計算までは変更なし...)
    print("計算を開始します..."); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    print("モデルを読み込んでいます..."); model = NeRF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)); model.eval()
    print("モデルの読み込みが完了しました。")
    print(f"{resolution}x{resolution}x{resolution} のグリッドを生成中...")
    x, y, z = [torch.linspace(bmin, bmax, resolution) for bmin, bmax in zip(bbox_min, bbox_max)]
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    total_points = pts.shape[0]; densities = []
    start_time = time.time(); print("密度の計算を開始します...")
    with torch.no_grad():
        for i in range(0, total_points, batch_size):
            densities.append(model(pts[i:i+batch_size].to(device)).cpu())
    print(f"密度計算が完了しました。(所要時間: {time.time() - start_time:.2f}秒)")
    densities_tensor = torch.cat(densities, dim=0).squeeze()
    
    mask = densities_tensor > density_threshold
    occupied_points = pts[mask]
    occupied_densities = densities_tensor[mask]
    occupied_voxels_count = occupied_points.shape[0]
    
    if occupied_voxels_count == 0:
        print("\n❌ 密度が閾値を超えるボクセルが見つかりませんでした。"); return

    # --- 体積計算 ---
    voxel_volume = ((bbox_max[0] - bbox_min[0]) / (resolution - 1)) ** 3
    relative_total_volume = occupied_voxels_count * voxel_volume
    physical_volume_cubic_meters = relative_total_volume * (scene_scale_meters ** 3)
    volume_cm3 = physical_volume_cubic_meters * (100**3)

    # --- NeRF密度(σ)の統計情報 ---
    stats = {
        "max": torch.max(occupied_densities).item(), "min": torch.min(occupied_densities).item(),
        "mean": torch.mean(occupied_densities).item(), "median": torch.median(occupied_densities).item()
    }
    
    if save_csv:
        save_points_to_csv(csv_output_path, occupied_points, occupied_densities)
        
    # ★★★ ここからが質量を推定する処理 ★★★
    estimated_mass_g = 0
    if object_physical_density > 0 and volume_cm3 > 0:
        estimated_mass_g = object_physical_density * volume_cm3
    # ★★★ ここまでが追加箇所 ★★★
    
    print("\n✅ 計算完了！")
    print("-----------------------------------------")
    print(f"推定された物体の総体積: {volume_cm3:.4f} (cm^3)")
    if estimated_mass_g > 0:
        print(f"  -> 指定された物理密度: {object_physical_density} (g/cm^3)")
        print(f"  -> 推定された質量: {estimated_mass_g:.4f} (g)")
    print("\n表面を構成する点の数: {occupied_voxels_count}")
    print("表面のNeRF密度(σ)の統計情報:")
    print(f"  最大値: {stats['max']:.4f}, 最小値: {stats['min']:.4f}, 平均値: {stats['mean']:.4f}, 中央値: {stats['median']:.4f}")
    print("-----------------------------------------")

    if visualize:
        visualize_points_with_density(occupied_points, occupied_densities)

# ----------------------------------------------------------------------------
# 3. パラメータを設定して実行
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    MODEL_PATH = "/home/ubuntu/slocal/libero/ATM/nerf/model_20250725-033742.pth"
    VISUALIZE = True
    BOUNDING_BOX_MIN = [-1.5, -1.5, -1.5]; BOUNDING_BOX_MAX = [ 1.5,  1.5,  1.5]
    RESOLUTION = 128 
    DENSITY_THRESHOLD = 0.1
    BATCH_SIZE = 4096
    SCENE_SCALE_METERS = 0.5
    SAVE_CSV = True
    CSV_OUTPUT_PATH = 'surface_points_and_densities.csv'

    # ★★★ 追加: 解析したい物体の「物理的な密度(g/cm^3)」をここに設定 ★★★
    # 材質を調べて値を設定してください。
    # 例: 水=1.0, ABS樹脂=約1.05, アルミニウム=2.7
    OBJECT_PHYSICAL_DENSITY_G_CM3 = 1.05 # ← ここをオブジェクトの材質に合わせて変更

    # (モデル検証部分は省略)

    calculate_volume_from_nerf(
        model_path=MODEL_PATH, resolution=RESOLUTION,
        bbox_min=BOUNDING_BOX_MIN, bbox_max=BOUNDING_BOX_MAX,
        density_threshold=DENSITY_THRESHOLD, batch_size=BATCH_SIZE,
        visualize=VISUALIZE, scene_scale_meters=SCENE_SCALE_METERS,
        save_csv=SAVE_CSV, csv_output_path=CSV_OUTPUT_PATH,
        object_physical_density=OBJECT_PHYSICAL_DENSITY_G_CM3 # ★★★ 追加
    )