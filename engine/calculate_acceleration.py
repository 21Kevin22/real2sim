# calculate_acceleration.py (修正版)
import csv
import math
import argparse
import sys
import os
import cv2

def calculate_physics_data(coords, time_interval):
    """座標データから速度と加速度を計算する"""
    if len(coords) < 3:
        return []

    # 1. 速度を計算 (v = Δp / Δt)
    velocities = []
    for i in range(1, len(coords)):
        # 座標データがないフレームはスキップ
        if coords[i]['x'] is None or coords[i-1]['x'] is None:
            velocities.append({'vx': None, 'vy': None})
            continue

        vx = (coords[i]['x'] - coords[i-1]['x']) / time_interval
        vy = (coords[i]['y'] - coords[i-1]['y']) / time_interval
        velocities.append({'vx': vx, 'vy': vy})

    # 2. 加速度を計算 (a = Δv / Δt)
    accelerations = []
    # 加速度は3フレーム目から計算可能 (位置データが3つ必要)
    for i in range(1, len(velocities)):
        # 速度データがないフレームはスキップ
        if velocities[i]['vx'] is None or velocities[i-1]['vx'] is None:
            accelerations.append({'frame': coords[i+1]['frame'], 'ax': None, 'ay': None, 'magnitude': None})
            continue

        ax = (velocities[i]['vx'] - velocities[i-1]['vx']) / time_interval
        ay = (velocities[i]['vy'] - velocities[i-1]['vy']) / time_interval
        magnitude = math.sqrt(ax**2 + ay**2)
        
        # 加速度は、3点 (t-2, t-1, t) の情報から t の値を計算するため、フレーム番号は i+1 に対応
        accelerations.append({
            'frame': coords[i+1]['frame'], 
            'ax': ax, 
            'ay': ay, 
            'magnitude': magnitude
        })
        
    return accelerations

def main():
    parser = argparse.ArgumentParser(description="座標CSVファイルから速度と加速度を計算します。")
    parser.add_argument("-i", "--input-csv", required=True, help="入力座標CSVファイルのパス")
    parser.add_argument("-v", "--video", required=True, help="元の動画ファイルのパス (FPS取得用)")
    parser.add_argument("-o", "--output-csv", required=True, help="出力加速度CSVファイルのパス")
    args = parser.parse_args()

    # --- 1. 入力ファイルの存在確認 ---
    if not os.path.exists(args.input_csv):
        print(f"エラー: 入力CSVファイルが見つかりません: {args.input_csv}")
        sys.exit()
    if not os.path.exists(args.video):
        print(f"エラー: 動画ファイルが見つかりません: {args.video}")
        sys.exit()
        
    # --- 2. FPSの取得と時間間隔の計算 ---
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0:
        print("エラー: 動画から有効なFPSを取得できませんでした。")
        sys.exit()
    time_interval = 1.0 / fps
    print(f"動画のFPS: {fps:.2f}, フレーム間隔 (Δt): {time_interval:.4f} 秒")

    # --- 3. 座標CSVファイルの読み込み ---
    coords = []
    with open(args.input_csv, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader) # ヘッダーをスキップ
        for row in reader:
            # ★★★ ここを修正しました ★★★
            # int() を float() に変更して、小数点を含む数値を読み込めるようにします。
            x = float(row[1]) if row[1] else None
            y = float(row[2]) if row[2] else None
            coords.append({'frame': int(row[0]), 'x': x, 'y': y})
    print(f"{len(coords)} フレーム分の座標データを読み込みました。")

    # --- 4. 加速度の計算 ---
    print("加速度を計算中...")
    accelerations = calculate_physics_data(coords, time_interval)
    print("計算が完了しました。")

    # --- 5. 結果をCSVファイルに書き出し ---
    with open(args.output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['frame', 'acceleration_x', 'acceleration_y', 'acceleration_magnitude'])
        for acc in accelerations:
            # Noneの場合は空欄で出力
            ax = f"{acc['ax']:.2f}" if acc['ax'] is not None else ""
            ay = f"{acc['ay']:.2f}" if acc['ay'] is not None else ""
            mag = f"{acc['magnitude']:.2f}" if acc['magnitude'] is not None else ""
            writer.writerow([acc['frame'], ax, ay, mag])
            
    print(f"加速度データを '{args.output_csv}' に保存しました。")

if __name__ == '__main__':
    main()