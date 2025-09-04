import cv2
import argparse
import csv
import numpy as np
import os
import sys
from tqdm import tqdm # tqdmライブラリをインポート

def load_csv_data_to_dict(filepath, key_column='frame'):
    """CSVファイルを読み込み、指定されたキー列を基準にした辞書として返す。"""
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

def process_frame(frame, frame_number, data, args):
    """各フレームの描画処理を行う関数"""
    height, width, _ = frame.shape
    
    # 各種データを取得
    coord_info = data['coords'].get(frame_number)
    prev_coord = data['coords'].get(frame_number - 1)
    next_coord = data['coords'].get(frame_number + 1)
    accel_info = data['accels'].get(frame_number)
    force_info = data['forces'].get(frame_number) # ★★★ 追加 ★★★

    # 中心点を描画
    if coord_info and coord_info.get('x') is not None:
        start_point = (int(coord_info['x']), int(coord_info['y']))
        if not (0 <= start_point[0] < width and 0 <= start_point[1] < height):
            return # 座標が画面外なら何もしない
            
        cv2.circle(frame, start_point, radius=5, color=tuple(args.dot_color), thickness=-1)

        # 速度ベクトルを描画
        if prev_coord and next_coord and prev_coord.get('x') is not None and next_coord.get('x') is not None:
            velocity_vector = np.array([next_coord['x'] - prev_coord['x'], next_coord['y'] - prev_coord['y']])
            vel_end_point = tuple((np.array(start_point) + velocity_vector * args.vel_scale).astype(int))
            cv2.arrowedLine(frame, start_point, vel_end_point, tuple(args.vel_color), args.thickness, tipLength=0.3)

        # 加速度ベクトルを描画
        if accel_info and accel_info.get('ax') is not None:
            accel_vector = np.array([accel_info['ax'], accel_info['ay']])
            acc_end_point = tuple((np.array(start_point) + accel_vector * args.acc_scale).astype(int))
            cv2.arrowedLine(frame, start_point, acc_end_point, tuple(args.acc_color), args.thickness, tipLength=0.3)

        # ★★★ ここから力のベクトル描画機能を追加 ★★★
        if force_info:
            fx = force_info.get('force_x')
            fy = force_info.get('force_y')
            magnitude = force_info.get('force_magnitude')

            if fx is not None and fy is not None and magnitude is not None:
                # 矢印（力ベクトル）の描画
                force_vector = np.array([fx, fy])
                force_end_point = tuple((np.array(start_point) + force_vector * args.force_scale).astype(int))
                cv2.arrowedLine(frame, start_point, force_end_point, tuple(args.force_color), args.thickness, tipLength=0.3)
                
                # 数値（力の大きさ）の描画
                text = f"{magnitude:.2f}"
                text_position = (force_end_point[0] + 10, force_end_point[1] - 10)
                cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                            args.font_scale, tuple(args.font_color), 2, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="動画に座標点、速度・加速度・力のベクトルを描画します。")
    # --- 引数の定義 ---
    parser.add_argument("--video", required=True, help="元の動画ファイルパス")
    parser.add_argument("--coords", required=True, help="座標CSVファイルパス")
    parser.add_argument("--accels", required=True, help="加速度CSVファイルパス")
    parser.add_argument("--forces", required=True, help="力CSVファイルパス") # ★★★ 追加 ★★★
    parser.add_argument("--output", required=True, help="出力動画のファイルパス")
    
    # 描画設定
    parser.add_argument("--acc-scale", type=float, default=0.01, help="加速度ベクトルの表示倍率")
    parser.add_argument("--acc-color", nargs=3, type=int, default=[0, 0, 255], help="加速度ベクトルの色 (B G R) - 赤")
    parser.add_argument("--vel-scale", type=float, default=1.0, help="速度ベクトルの表示倍率")
    parser.add_argument("--vel-color", nargs=3, type=int, default=[255, 0, 0], help="速度ベクトルの色 (B G R) - 青")
    parser.add_argument("--force-scale", type=float, default=0.5, help="力ベクトルの表示倍率") # ★★★ 追加 ★★★
    parser.add_argument("--force-color", nargs=3, type=int, default=[0, 165, 255], help="力ベクトルの色 (B G R) - オレンジ") # ★★★ 追加 ★★★
    parser.add_argument("--dot-color", nargs=3, type=int, default=[0, 255, 0], help="中心点の色 (B G R) - 緑")
    parser.add_argument("--thickness", type=int, default=2, help="矢印の太さ")
    parser.add_argument("--font-scale", type=float, default=0.6, help="文字の大きさ") # ★★★ 追加 ★★★
    parser.add_argument("--font-color", nargs=3, type=int, default=[255, 255, 255], help="文字の色 (B G R) - 白") # ★★★ 追加 ★★★
    args = parser.parse_args()

    # --- データの読み込み ---
    all_data = {
        'coords': load_csv_data_to_dict(args.coords),
        'accels': load_csv_data_to_dict(args.accels),
        'forces': load_csv_data_to_dict(args.forces), # ★★★ 追加 ★★★
    }
    if not all_data['coords'] or not all_data['accels'] or not all_data['forces']: sys.exit(1)

    # --- 動画のセットアップ ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません: {args.video}"); sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if width <= 0 or height <= 0:
        print("エラー: 無効なフレームサイズです"); sys.exit(1)
    if fps <= 0: fps = 30.0

    # --- VideoWriterのセットアップ ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # .mp4形式に適したコーデック
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        # 代替コーデックで再試行
        print("警告: mp4vコーデックでの初期化に失敗。MJPGで再試行します。")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not writer.isOpened():
            print("エラー: 出力動画ファイルが作成できません"); sys.exit(1)

    # --- メインループ (プログレスバー付き) ---
    for frame_number in tqdm(range(total_frames), desc="動画処理中", unit="フレーム"):
        success, frame = cap.read()
        if not success: break
        
        process_frame(frame, frame_number, all_data, args)
        
        writer.write(frame)

    # --- 終了処理 ---
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n処理完了！\n出力動画を '{args.output}' に保存しました。")

if __name__ == '__main__':
    main()