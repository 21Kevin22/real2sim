import cv2
import argparse
import sys
import csv
import os

def track_object():
    parser = argparse.ArgumentParser(description="動画内の指定された物体を追跡し、中心座標をCSVに保存します。")
    parser.add_argument("-v", "--video", required=True, help="入力動画ファイルのパス (.mp4)")
    parser.add_argument("-o", "--output", help="出力CSVファイルのパス（自動追跡モードでのみ使用）")
    parser.add_argument("--tracker", default="csrt", help="使用するトラッカーの種類 (例: csrt, kcf)")
    parser.add_argument(
        "--roi", nargs=4, type=int, 
        help="追跡対象のROI (Region of Interest) を 'x y w h' の形式で指定（自動追跡モード用）"
    )
    args = parser.parse_args()

    # OpenCVトラッカーのディクショナリ
    trackers = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
    }
    tracker = trackers[args.tracker]()

    # 動画ファイルを読み込む
    if not os.path.exists(args.video):
        print(f"エラー: 動画ファイルが見つかりません: {args.video}")
        sys.exit()
    cap = cv2.VideoCapture(args.video)

    # 最初のフレームを読み込む
    success, frame = cap.read()
    if not success:
        print("エラー: 動画の最初のフレームを読み込めませんでした。")
        sys.exit()

    roi = tuple(args.roi) if args.roi else None

    # --- モード分岐 ---
    # ROIが指定されていない場合 -> ROI選択モード
    if roi is None:
        print("--- ROI選択モード ---")
        print("1. 追跡したい物体をマウスでドラッグして四角で囲ってください。")
        print("2. 選択したら、'スペースキー' または 'Enterキー' を押して確定します。")
        print("3. やり直す場合は 'c' キーを押してください。")
        roi = cv2.selectROI("ROI Selector", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("ROI Selector")
        if not roi or roi[2] == 0 or roi[3] == 0:
            print("ROIが選択されませんでした。プログラムを終了します。")
            sys.exit()
        
        print("\nROI選択完了")
        print(f"取得したROI座標 (x y w h): {roi[0]} {roi[1]} {roi[2]} {roi[3]}")
        print("この座標をコピーして、次の自動追跡モードで --roi オプションとして使用してください。")
        sys.exit() # 座標を表示して終了

    # ROIが指定されている場合 -> 自動追跡モード
    else:
        if not args.output:
            print("エラー: 自動追跡モードでは --output ファイルパスが必要です。")
            sys.exit()
        print("--- 自動追跡モード ---")
        print(f"指定されたROI {roi} で追跡を開始します。")

    # トラッカーを初期化
    tracker.init(frame, roi)

    # CSVファイルの準備
    with open(args.output, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['frame', 'x', 'y'])

        frame_number = 0
        while True:
            success, frame = cap.read()
            if not success:
                break # 動画の終わり

            # トラッカーを更新
            track_success, bbox = tracker.update(frame)

            if track_success:
                # バウンディングボックスの中心座標を計算
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                center_x = int(p1[0] + bbox[2] / 2)
                center_y = int(p1[1] + bbox[3] / 2)
                
                # CSVに書き込み
                writer.writerow([frame_number, center_x, center_y])
            else:
                # 追跡に失敗した場合（データは欠損となる）
                pass
            
            frame_number += 1
            if frame_number % 100 == 0:
                print(f"  ... {frame_number} フレームを処理中 ...")

    cap.release()
    print(f"\n処理完了 {frame_number} フレーム分の追跡データを '{args.output}' に保存しました。")
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
    print(f"この動画のFPSは {fps:.2f} です。次の加速度計算でこの値を使用してください。")


if __name__ == '__main__':
    track_object()