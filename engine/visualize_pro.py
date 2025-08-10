import cv2
import argparse
import csv
import numpy as np
import os
import sys
from tqdm import tqdm

# (load_csv_data_to_dict 関数は変更なし)
def load_csv_data_to_dict(filepath, key_column='frame'):
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}"); return None
    data_dict = {}
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or key_column not in row: continue
                processed_row = {}
                for k, v in row.items():
                    if k != key_column:
                        try: processed_row[k] = float(v) if v else None
                        except (ValueError, TypeError): processed_row[k] = None
                data_dict[int(row[key_column])] = processed_row
    except Exception as e:
        print(f"エラー: '{filepath}' の読み込み中に問題が発生しました: {e}"); return None
    return data_dict

def main():
    parser = argparse.ArgumentParser(description="動画に物体の運動ベクトル（速度、加速度成分）を描画します。")
    # --- 引数の定義 ---
    parser.add_argument("--video", required=True, help="元の動画ファイルパス")
    parser.add_argument("--coords", required=True, help="座標CSVファイルパス")
    parser.add_argument("--accels", required=True, help="加速度CSVファイルパス")
    parser.add_argument("--output", required=True, help="出力動画のファイルパス")
    
    # --- 視点操作の引数 ---
    group = parser.add_mutually_exclusive_group() # 同時に使えないオプションをグループ化
    group.add_argument("--follow", action="store_true", help="物体を追跡する視点（フォローカメラ）を有効にする")
    group.add_argument("--center-on", nargs=2, type=int, metavar=('X', 'Y'), help="指定した'x y'座標を常に画面中央にする")

    parser.add_argument("--smoothing", type=float, default=0.1, help="追跡視点の滑らかさ係数 (0.0-1.0)")
    parser.add_argument("--components", action="store_true", help="加速度を接線・法線成分に分解して表示する")
    
    # --- 描画設定の引数 ---
    parser.add_argument("--scale", type=float, default=0.01, help="ベクトル全体の表示倍率")
    parser.add_argument("--thickness", type=int, default=2, help="矢印の太さ")
    parser.add_argument("--dot-color", nargs=3, type=int, default=[0, 255, 0], help="中心点の色 (B G R)")
    args = parser.parse_args()

    # --- データと動画のセットアップ ---
    all_data = {'coords': load_csv_data_to_dict(args.coords), 'accels': load_csv_data_to_dict(args.accels)}
    if not all_data['coords']: print("座標データがありません。"); sys.exit(1)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): print(f"エラー: 動画ファイルが開けません: {args.video}"); sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened(): print("エラー: 出力動画ファイルが作成できません"); sys.exit(1)

    # --- ループ変数の初期化 ---
    camera_center = np.array([width / 2, height / 2], dtype=float)
    follow_cam_initialized = False

    for frame_number in tqdm(range(total_frames), desc="動画処理中", unit="フレーム"):
        success, frame = cap.read()
        if not success: break
        
        # --- オブジェクトの生座標を取得 ---
        coord_info = all_data['coords'].get(frame_number)
        has_valid_pos = coord_info and coord_info.get('x') is not None
        object_pos = np.array([coord_info['x'], coord_info['y']], dtype=float) if has_valid_pos else None

        # --- 1. 画面の移動量(shift)を決定 ---
        shift = np.array([0.0, 0.0])
        screen_center = np.array([width / 2, height / 2])

        if args.center_on:
            # 【新機能】指定した静的座標を中央にする
            target_pos = np.array(args.center_on, dtype=float)
            shift = screen_center - target_pos
        elif args.follow:
            # 【既存機能】動く物体を中央にする
            if has_valid_pos:
                if not follow_cam_initialized:
                    camera_center = object_pos
                    follow_cam_initialized = True
                else:
                    camera_center += (object_pos - camera_center) * args.smoothing
            # 物体を見失っても、最後のカメラ位置を維持
            shift = screen_center - camera_center

        # --- 2. 画面を移動させる ---
        if np.any(shift != 0):
            M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            frame = cv2.warpAffine(frame, M, (width, height))

        # --- 3. 描画処理 ---
        if has_valid_pos:
            # 移動後の画面におけるオブジェクトの描画位置を計算
            draw_pos = (object_pos + shift).astype(int)

            # 中心の点を描画
            cv2.circle(frame, tuple(draw_pos), radius=5, color=tuple(args.dot_color), thickness=-1)
            
            # ベクトル計算（変更なし）
            prev_coord = all_data['coords'].get(frame_number - 1)
            next_coord = all_data['coords'].get(frame_number + 1)
            accel_info = all_data['accels'].get(frame_number)
            
            velocity_vector = np.array([0.0, 0.0])
            if prev_coord and next_coord and prev_coord.get('x') is not None and next_coord.get('x') is not None:
                if next_coord.get('y') is not None and prev_coord.get('y') is not None:
                    velocity_vector = np.array([next_coord['x'] - prev_coord['x'], next_coord['y'] - prev_coord['y']])

            accel_vector = np.array([0.0, 0.0])
            if accel_info and accel_info.get('acceleration_x') is not None and accel_info.get('acceleration_y') is not None:
                accel_vector = np.array([accel_info['acceleration_x'], accel_info['acceleration_y']])
            
            # ベクトル描画
            if args.components:
                norm_v = np.linalg.norm(velocity_vector)
                if norm_v > 1e-6:
                    # ... (成分分解の描画ロジック、描画基準点を draw_pos に変更)
                    unit_v = velocity_vector / norm_v
                    tangential_accel_scalar = np.dot(accel_vector, unit_v)
                    tangential_accel_vector = tangential_accel_scalar * unit_v
                    normal_accel_vector = accel_vector - tangential_accel_vector
                    
                    tangential_color = [0, 255, 0] if tangential_accel_scalar > 0 else [0, 0, 255]
                    t_end_point = tuple((draw_pos + tangential_accel_vector * args.scale).astype(int))
                    cv2.arrowedLine(frame, tuple(draw_pos), t_end_point, tangential_color, args.thickness, tipLength=0.3)

                    normal_color = [255, 0, 255]
                    n_end_point = tuple((draw_pos + normal_accel_vector * args.scale).astype(int))
                    cv2.arrowedLine(frame, tuple(draw_pos), n_end_point, normal_color, args.thickness, tipLength=0.3)
            else:
                # ... (通常ベクトルの描画ロジック、描画基準点を draw_pos に変更)
                vel_end_point = tuple((draw_pos + velocity_vector * 1.0).astype(int))
                cv2.arrowedLine(frame, tuple(draw_pos), vel_end_point, [255, 0, 0], args.thickness, tipLength=0.3)
                acc_end_point = tuple((draw_pos + accel_vector * args.scale).astype(int))
                cv2.arrowedLine(frame, tuple(draw_pos), acc_end_point, [0, 0, 255], args.thickness, tipLength=0.3)

        writer.write(frame)

    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"\n処理完了\n出力動画を '{args.output}' に保存しました。")

if __name__ == '__main__':
    main()