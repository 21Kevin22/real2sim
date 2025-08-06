import cv2
import argparse
import csv
import numpy as np
import os
import sys

def ensure_video_extension(filepath):
    """ビデオファイルの拡張子が適切かチェック・修正"""
    valid_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    _, ext = os.path.splitext(filepath)
    if ext.lower() not in valid_extensions:
        return filepath + '.avi'
    return filepath

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

def get_best_fourcc(width, height, fps):
    """利用可能な最適なコーデックを取得する"""
    # 優先順位順にコーデックを試す
    codecs = [
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion JPEG (最も安定)
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Xvid
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264
        ('H264', cv2.VideoWriter_fourcc(*'H264')),  # H.264
        ('X264', cv2.VideoWriter_fourcc(*'X264')),  # H.264
        # 数値指定も試す
        (-1, -1),  # デフォルト
    ]
    
    for name, fourcc in codecs:
        try:
            print(f"コーデック {name} をテスト中...")
            # 実際のサイズでテスト動画を作成
            test_writer = cv2.VideoWriter('test_temp.avi', fourcc, fps, (width, height))
            if test_writer.isOpened():
                # 複数フレームを書き込んでテスト
                for i in range(5):
                    test_frame = np.full((height, width, 3), i * 50, dtype=np.uint8)
                    success = test_writer.write(test_frame)
                    if not success:
                        print(f"  フレーム書き込み失敗: {name}")
                        break
                else:
                    test_writer.release()
                    
                    # ファイルが適切に作成されたかチェック
                    if os.path.exists('test_temp.avi') and os.path.getsize('test_temp.avi') > 1000:
                        # 作成したファイルが読み込めるかテスト
                        test_cap = cv2.VideoCapture('test_temp.avi')
                        if test_cap.isOpened():
                            ret, frame = test_cap.read()
                            test_cap.release()
                            os.remove('test_temp.avi')
                            if ret:
                                print(f"✓ 使用するコーデック: {name}")
                                return fourcc
                        else:
                            os.remove('test_temp.avi')
                    else:
                        if os.path.exists('test_temp.avi'):
                            os.remove('test_temp.avi')
                
                test_writer.release()
            else:
                test_writer.release()
                print(f"  コーデック {name} の初期化失敗")
        except Exception as e:
            print(f"  コーデック {name} のテスト中にエラー: {e}")
            if os.path.exists('test_temp.avi'):
                try:
                    os.remove('test_temp.avi')
                except:
                    pass
            continue
    
    # すべて失敗した場合は最もシンプルなコーデック
    print("警告: すべてのコーデックテストが失敗しました。MJPGを強制使用します。")
    return cv2.VideoWriter_fourcc(*'MJPG')

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
    parser.add_argument("--debug", action="store_true", help="詳細なデバッグ情報を表示")
    args = parser.parse_args()

    print("CSVデータを読み込んでいます...")
    coords_data = load_csv_data_to_dict(args.coords)
    accels_data = load_csv_data_to_dict(args.accels)
    if coords_data is None or accels_data is None:
        sys.exit(1)
    
    # CSVデータのサンプル表示
    print(f"座標データ: {len(coords_data)} フレーム")
    if coords_data:
        sample_frame = list(coords_data.keys())[0]
        print(f"  サンプル (フレーム {sample_frame}): {coords_data[sample_frame]}")
    
    print(f"加速度データ: {len(accels_data)} フレーム")
    if accels_data:
        sample_frame = list(accels_data.keys())[0]
        print(f"  サンプル (フレーム {sample_frame}): {accels_data[sample_frame]}")

    # --- 動画のセットアップ ---
    print(f"入力動画ファイルをチェック中: {args.video}")
    
    # ファイルの存在確認
    if not os.path.exists(args.video):
        print(f"エラー: 動画ファイルが存在しません: {args.video}")
        return
    
    # ファイルサイズの確認
    file_size = os.path.getsize(args.video)
    print(f"ファイルサイズ: {file_size} bytes")
    if file_size == 0:
        print("エラー: 動画ファイルが空です")
        return
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません: {args.video}")
        print("可能な原因:")
        print("  - サポートされていない動画形式")
        print("  - 破損した動画ファイル")
        print("  - OpenCVのコーデックサポート不足")
        
        # ffprobeがあれば詳細情報を取得
        try:
            import subprocess
            result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', args.video], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("ffprobeによる動画情報:")
                print(result.stdout)
            else:
                print("ffprobeでも動画情報を取得できませんでした")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("ffprobeが利用できません")
        
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

    # 出力ファイル形式を確認・調整
    output_path = args.output
    if not output_path.lower().endswith(('.avi', '.mp4', '.mov')):
        print(f"警告: 出力ファイル形式を .avi に変更します")
        output_path = os.path.splitext(args.output)[0] + '.avi'
    
    # 最適なコーデックを取得
    fourcc = get_best_fourcc(width, height, fps)
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # VideoWriterを初期化
    print(f"VideoWriterを初期化中...")
    print(f"  - 出力ファイル: {output_path}")
    print(f"  - コーデック: {fourcc}")
    print(f"  - FPS: {fps}")
    print(f"  - 解像度: {width} x {height}")
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"エラー: 出力動画ファイルが作成できません: {output_path}")
        print("代替コーデックを試します...")
        
        # より確実なAVI形式で再試行
        if not output_path.lower().endswith('.avi'):
            output_path = os.path.splitext(output_path)[0] + '.avi'
            print(f"出力形式を.aviに変更: {output_path}")
        
        # 代替コーデックで再試行
        alt_codecs = [
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ('default', -1),  # デフォルト
        ]
        
        for alt_name, alt_fourcc in alt_codecs:
            print(f"代替コーデック {alt_name} を試行中...")
            writer = cv2.VideoWriter(output_path, alt_fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"代替コーデック {alt_name} で初期化成功")
                break
            writer.release()
        else:
            print("すべてのコーデックで初期化に失敗しました")
            cap.release()
            return

    print(f"動画の処理を開始します... 出力先: {output_path}")
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
            success = writer.write(frame)
            if not success:
                print(f"警告: フレーム {frame_number} の書き込みに失敗しました")
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
    print(f"加速度ベクトルを描画した動画を '{output_path}' に保存しました。")
    
    # 出力ファイルが正常に作成されたかチェック
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("✓ 出力動画ファイルが正常に作成されました。")
        
        # 出力ビデオの検証
        print("\n出力ビデオの検証中...")
        test_cap = cv2.VideoCapture(output_path)
        if test_cap.isOpened():
            test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_fps = test_cap.get(cv2.CAP_PROP_FPS)
            test_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            test_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✓ 出力ビデオ情報:")
            print(f"  - 解像度: {test_width} x {test_height}")
            print(f"  - FPS: {test_fps}")
            print(f"  - フレーム数: {test_frames}")
            print(f"  - ファイルサイズ: {os.path.getsize(output_path)} bytes")
            
            # 最初のフレームを読み込んでテスト
            ret, test_frame = test_cap.read()
            if ret:
                print("✓ 最初のフレームの読み込み成功")
            else:
                print("✗ 最初のフレームの読み込み失敗")
            
            test_cap.release()
        else:
            print("✗ 警告: 出力ビデオが開けません")
    else:
        print("✗ 警告: 出力動画ファイルに問題がある可能性があります。")
        if os.path.exists(output_path):
            print(f"   ファイルサイズ: {os.path.getsize(output_path)} bytes")
        else:
            print("   ファイルが存在しません")

if __name__ == '__main__':
    visualize_vectors()