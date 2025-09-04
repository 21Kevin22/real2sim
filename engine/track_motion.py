import cv2
import numpy as np
import csv
import argparse
import os
from tqdm import tqdm

def main():
    """
    動画内の動体を検出し、その面積を物理単位で計算して時系列でCSVファイルに記録します。
    キャリブレーションのためのスケール値が必要です。
    """
    # --- コマンドライン引数の設定 ---
    parser = argparse.ArgumentParser(description="動画内の動いている物体の面積を検出し、物理単位で追跡します。")
    parser.add_argument("-v", "--video", required=True, help="入力動画ファイルのパス (.mp4)")
    parser.add_argument("-o", "--output", required=True, help="出力CSVファイルのパス")
    # ★★★ 追加した引数 ★★★
    parser.add_argument("-s", "--scale", type=float, required=True, help="キャリブレーションで求めたスケール値 (例: cm/pixel)")
    parser.add_argument("--min-area", type=int, default=500, help="検出する物体の最小面積（ノイズ除去用）")
    parser.add_argument("--history", type=int, default=500, help="背景モデル学習に使うフレーム数")
    args = parser.parse_args()

    # --- 動画ファイルの読み込み ---
    if not os.path.exists(args.video):
        print(f"エラー: 動画ファイルが見つかりません: {args.video}")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした。")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"動画を読み込みました: {args.video} ({total_frames} フレーム)")
    print(f"使用するスケール値: {args.scale} (cm/pixel)")

    # --- 背景差分モデルの初期化 ---
    backSub = cv2.createBackgroundSubtractorMOG2(history=args.history, varThreshold=16, detectShadows=True)
    
    # --- CSVファイルの準備 ---
    try:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ★★★ ヘッダーを面積用に変更 ★★★
            # 物理単位の面積のヘッダーを動的に設定 (例: area_cm2)
            unit_name = "cm" # ここを 'mm' や 'm' に変更可能
            physical_area_header = f"area_{unit_name}2"
            writer.writerow(['frame', 'area_pixels', physical_area_header])
            
            # --- フレームごとの処理 ---
            for frame_number in tqdm(range(total_frames), desc="処理中", unit="フレーム"):
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. 前景マスクの生成
                fg_mask = backSub.apply(frame)

                # 2. マスクのノイズ除去と平滑化
                kernel = np.ones((3, 3), np.uint8)
                fg_mask_opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                fg_mask_closed = cv2.morphologyEx(fg_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=3)

                # 3. 輪郭の検出
                contours, _ = cv2.findContours(fg_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 見つかった輪郭の中から最大のものを探す
                largest_contour = None
                max_area_pixels = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > args.min_area and area > max_area_pixels:
                        max_area_pixels = area
                        largest_contour = contour
                
                # ★★★ 最大の輪郭の面積を計算して記録 ★★★
                if largest_contour is not None:
                    # ピクセル単位の面積
                    area_pixels = max_area_pixels
                    
                    # スケール値を使って物理的な面積を計算
                    # 面積なので、スケール値は2乗する
                    physical_area = area_pixels * (args.scale ** 2)
                    
                    # CSVファイルに書き込み
                    writer.writerow([frame_number, round(area_pixels, 2), round(physical_area, 4)])
                else:
                    # 追跡対象が見つからなかった
                    writer.writerow([frame_number, 0, 0])

    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        cap.release()
        print(f"\n処理が完了しました。追跡データを '{args.output}' に保存しました。")

if __name__ == '__main__':
    main()