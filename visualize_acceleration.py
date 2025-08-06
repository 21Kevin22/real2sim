import cv2
import argparse
import csv
import numpy as np
import os
import sys

def load_csv_data_to_dict(filepath, key_column='frame'):
    """CSVファイルを読み込み、指定されたキー列を基準にした辞書として返す。"""
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}")
        return None
    data_dict = {}
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or key_column not in row: continue
                processed_row = {}
                for k, v in row.items():
                    if k != key_column:
                        try:
                            processed_row[k] = float(v) if v else None
                        except (ValueError, TypeError):
                            processed_row[k] = None
                data_dict[int(row[key_column])] = processed_row
    except Exception as e:
        print(f"エラー: '{filepath}' の読み込み中に問題が発生しました: {e}")
        return None
    return data_dict

def get_best_fourcc():
    """利用可能な最適なコーデックを取得する"""
    # 優先順位順にコーデックを試す
    codecs = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Xvid
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion JPEG
        ('X264', cv2.VideoWriter_fourcc(*'X264')),  # H.264
    ]
    
    for name, fourcc in codecs:
        # 小さなテスト動画で各コーデックをテスト
        test_writer = cv2.VideoWriter('test_temp.mp4', fourcc, 1, (100, 100))
        if test_writer.isOpened():
            test_writer.release()
            # テストファイルを削除
            if os.path.exists('test_temp.mp4'):
                os.remove('test_temp.mp4')
            print(f"使用するコーデック: {name}")
            return fourcc
        test_writer.release()
    
    # すべて失敗した場合はデフォルト
    print("警告: 適切なコーデックが見つかりません。デフォルトを使用します。")
    return cv2.VideoWriter_fourcc(*'mp4v')

def visualize_vectors():
    """動画に中心点と加速度ベクトルを描画し、新しい動画として保存する。"""
    parser = argparse.ArgumentParser(description="動画に座標点と加速度ベクトルを描画します。")
    parser.add_argument("--video", required=True, help="元の動画ファイルパス")
    parser.add_argument("--coords", required=True, help="中心座標が記録されたCSVファイルパス (coordinates.csv)")
    parser.add_argument("--accels", required=True, help="加速度ベクトルが記録されたCSVファイルパス (accelerations.csv)")
    parser.add_argument("--output", required=True, help="ベクトルを描画した出力動画のファイルパス")
    parser.add_argument("--scale", type=float, default=0.01, help="加速度ベクトルの表示倍率（矢印の長さ調整用）")
    parser.add_argument("--color", nargs=3, type=int, default=[0, 0, 255], help="矢印の色 (B G R)。例: 0 0 255 は赤")
    parser.add_argument("--thickness", type=int, default=2, help="矢印の太さ")
    parser.add_argument("--dot-color", nargs=3, type=int, default=[0, 255, 0], help="中心点の色 (B G R)。例: 0 255 0 は緑")
    args = parser.parse_args()

    print("CSVデータを読み込んでいます...")
    coords_data = load_csv_data_to_dict(args.coords)
    accels_data = load_csv_data_to_dict(args.accels)
    if coords_data is None or accels_data is None:
        sys.exit(1)

    # --- 動画のセットアップ ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません: {args.video}")
        return

    # 動画のプロパティを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"元動画の情報:")
    print(f"  解像度: {width} x {height}")
    print(f"  FPS: {fps}")
    print(f"  総フレーム数: {total_frames}")
    
    # フレームサイズが有効かチェック
    if width <= 0 or height <= 0:
        print("エラー: 無効なフレームサイズです")
        cap.release()
        return
        
    if fps <= 0:
        print("警告: 無効なFPSです。デフォルト値30を使用します")
        fps = 30.0

    # 最適なコーデックを取得
    fourcc = get_best_fourcc()
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # VideoWriterを初期化
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"エラー: 出力動画ファイルが作成できません: {args.output}")
        cap.release()
        return

    print(f"動画の処理を開始します... 出力先: {args.output}")
    frame_number = 0
    processed_frames = 0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # フレームサイズを確認・調整
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            # 座標データと加速度データを取得
            coord_info = coords_data.get(frame_number)
            accel_info = accels_data.get(frame_number)
            
            # 中心点を描画
            if coord_info and coord_info.get('x') is not None and coord_info.get('y') is not None:
                start_point = (int(coord_info['x']), int(coord_info['y']))
                
                # 座標が画面内に収まるかチェック
                if 0 <= start_point[0] < width and 0 <= start_point[1] < height:
                    cv2.circle(frame, start_point, radius=5, color=tuple(args.dot_color), thickness=-1)
                    
                    # 加速度ベクトルを描画
                    if accel_info:
                        ax = accel_info.get('ax')
                        ay = accel_info.get('ay')
                        if ax is not None and ay is not None:
                            accel_vector = np.array([ax, ay])
                            end_point_vector = np.array(start_point) + accel_vector * args.scale
                            end_point = tuple(end_point_vector.astype(int))
                            
                            # 矢印の終点も画面内に収まるかチェック
                            if 0 <= end_point[0] < width and 0 <= end_point[1] < height:
                                cv2.arrowedLine(frame, start_point, end_point, 
                                              tuple(args.color), args.thickness, tipLength=0.3)
            
            # フレームを書き込み
            writer.write(frame)
            processed_frames += 1
            frame_number += 1
            
            if frame_number % 100 == 0:
                print(f"  ... {frame_number} フレームを処理済み ...")
                
    except Exception as e:
        print(f"エラー: フレーム処理中に問題が発生しました: {e}")
    finally:
        # リソースを確実に解放
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"\n処理完了!")
    print(f"処理されたフレーム数: {processed_frames}")
    print(f"加速度ベクトルを描画した動画を '{args.output}' に保存しました。")
    
    # 出力ファイルが正常に作成されたかチェック
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        print("✓ 出力動画ファイルが正常に作成されました。")
    else:
        print("✗ 警告: 出力動画ファイルに問題がある可能性があります。")

if __name__ == '__main__':
    visualize_vectors()