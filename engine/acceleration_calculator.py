import argparse
import numpy as np
import csv

def calculate_accelerations(coordinates, fps):
    """
    座標の時系列データとフレームレートから加速度ベクトルのリストを計算します。
    """
    coords_np = np.array(coordinates, dtype=float)

    # ▼▼▼▼▼ 修正点1: データが3フレーム未満の場合は計算不可とする ▼▼▼▼▼
    # 加速度の計算には最低3つの点(過去, 現在, 未来)が必要なため。
    if len(coords_np) < 3:
        return [None] * len(coords_np)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    if fps <= 0:
        raise ValueError("フレームレート(fps)は正の値である必要があります。")
    delta_t = 1.0 / fps
    accelerations = [None]
    for i in range(1, len(coords_np) - 1):
        p_prev = coords_np[i-1]
        p_curr = coords_np[i]
        p_next = coords_np[i+1]
        acceleration_vector = (p_next - 2 * p_curr + p_prev) / (delta_t ** 2)
        accelerations.append(acceleration_vector)
    accelerations.append(None)
    return accelerations

def main():
    """
    メインの処理。コマンドライン引数を受け取り、加速度を計算してファイルに保存します。
    """
    parser = argparse.ArgumentParser(
        description="座標データCSVから加速度を計算し、別のCSVファイルに保存します。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # ( ... 引数の定義は変更なし ... )
    parser.add_argument("-i", "--input", required=True, help="入力CSVファイルのパス。\n形式: 1行目はヘッダー(例: frame,x,y)、2行目以降がデータ。")
    parser.add_argument("-o", "--output", required=True, help="計算結果の加速度を保存する出力CSVファイルのパス。")
    parser.add_argument("--fps", type=float, required=True, help="動画のフレームレート (例: 30.0)")
    args = parser.parse_args()

    # --- 入力CSVファイルから座標データを読み込み ---
    coordinates = []
    frame_ids = []
    print(f"入力ファイル '{args.input}' を読み込んでいます...")
    try:
        with open(args.input, 'r', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            coord_indices = [i for i, col in enumerate(header) if col.lower() in ['x', 'y', 'z']]
            frame_index = header.index('frame')
            for row in reader:
                frame_ids.append(row[frame_index])
                coords = tuple(float(row[i]) for i in coord_indices)
                coordinates.append(coords)
    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません: {args.input}")
        return
    except (ValueError, StopIteration): # データ行がない場合にStopIterationが発生する
        # このブロックは、ファイルは存在するがヘッダーのみの場合に通過する
        pass
    except Exception as e:
        print(f"エラー: ファイル読み込み中に問題が発生しました: {e}")
        return

    # ▼▼▼▼▼ 修正点2: 座標データが空の場合のチェックを追加 ▼▼▼▼▼
    if not coordinates:
        print("エラー: 座標データが見つかりませんでした。入力ファイルが空か、動画内で手が検出されなかった可能性があります。")
        # 空の出力ファイル（ヘッダーのみ）を作成して、処理を終了します。
        with open(args.output, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['frame', 'ax', 'ay']) # 2D座標を想定したヘッダー
        return
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # --- 加速度を計算 ---
    print("加速度を計算中...")
    accel_vectors = calculate_accelerations(coordinates, args.fps)

    # --- 結果を出力CSVファイルに書き込み ---
    try:
        with open(args.output, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            coord_dims = ['x', 'y', 'z'][:len(coordinates[0])]
            accel_header = [f"a{dim}" for dim in coord_dims]
            writer.writerow(['frame'] + accel_header)
            for i, acc in enumerate(accel_vectors):
                row_data = [frame_ids[i]]
                if acc is not None:
                    row_data.extend(list(acc))
                else:
                    row_data.extend([''] * len(accel_header))
                writer.writerow(row_data)
        print(f"計算完了！ 加速度データを '{args.output}' に保存しました。")
    except Exception as e:
        print(f"エラー: ファイル書き込み中に問題が発生しました: {e}")

if __name__ == '__main__':
    main()