import cv2
import argparse
import csv
import numpy as np
import os
import sys
from tqdm import tqdm

def load_csv_data_to_dict(filepath, key_column='frame'):
    if not os.path.exists(filepath): print(f"エラー: ファイルが見つかりません: {filepath}"); return None
    data_dict = {}
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f);
            for row in reader:
                if not row or key_column not in row or not row[key_column]: continue
                processed_row = {k: float(v) if v and v.strip() else None for k, v in row.items() if k != key_column}
                data_dict[int(row[key_column])] = processed_row
    except Exception as e: print(f"エラー: '{filepath}' の読み込み中に問題が発生しました: {e}"); return None
    return data_dict

def get_best_fourcc(width, height, fps):
    codecs = [('mp4v', cv2.VideoWriter_fourcc(*'mp4v')), ('XVID', cv2.VideoWriter_fourcc(*'XVID')), ('MJPG', cv2.VideoWriter_fourcc(*'MJPG'))]
    temp_filename = 'test_codec.mp4';
    for name, fourcc in codecs:
        writer = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))
        if writer.isOpened():
            writer.release();
            if os.path.exists(temp_filename): os.remove(temp_filename)
            print(f"✓ 使用するコーデック: {name}"); return name, fourcc
    return 'mp4v', cv2.VideoWriter_fourcc(*'mp4v')

# ★★★ 全ての描画ロジックを統合した完全版の関数 ★★★
def draw_vectors_on_frame(frame, frame_number, all_data, args, last_known_position, shift):
    height, width, _ = frame.shape
    
    # 描画すべき基準点があるか確認
    if last_known_position is None:
        return # 追跡が一度も始まっていない場合は何もしない

    # カメラの移動量を反映した描画位置を計算
    draw_pos = tuple((np.array(last_known_position) + shift).astype(int))

    # 現在フレームのデータがあるか確認
    coord_info = all_data['coords'].get(frame_number)
    
    if coord_info and coord_info.get('x') is not None:
        # --- データが正常に見つかった場合の描画 ---
        # 描画位置が画面外ならスキップ
        if not (0 <= draw_pos[0] < width and 0 <= draw_pos[1] < height):
            return

        # 1. 中心点を描画
        cv2.circle(frame, draw_pos, radius=5, color=tuple(args.dot_color), thickness=-1)

        # 2. 各種データを取得
        accel_info = all_data['accels'].get(frame_number)
        force_info = all_data['forces'].get(frame_number)
        prev_coord = all_data['coords'].get(frame_number - 1)
        next_coord = all_data['coords'].get(frame_number + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 3. 右下に合計の力の大きさを描画
        force_text = f"Total Force: {force_info.get('force_magnitude', 0):.2f} N" if force_info else "Total Force: N/A"
        text_size, _ = cv2.getTextSize(force_text, font, args.font_scale, 2)
        text_position = (width - text_size[0] - 20, height - 20)
        cv2.putText(frame, force_text, text_position, font, args.font_scale, tuple(args.font_color), 2, cv2.LINE_AA)

        # 4. ベクトル描画
        if args.components:
            # --- 成分分解モード ---
            if accel_info and prev_coord and next_coord and prev_coord.get('x') is not None and next_coord.get('x') is not None:
                velocity_vector = np.array([next_coord['x'] - prev_coord['x'], next_coord['y'] - prev_coord['y']])
                norm_v = np.linalg.norm(velocity_vector)
                if norm_v > 1e-6 and accel_info.get('ax') is not None:
                    unit_v = velocity_vector / norm_v
                    accel_vector = np.array([accel_info['ax'], accel_info['ay']])
                    # 加速度成分
                    tangential_scalar_a = np.dot(accel_vector, unit_v)
                    tangential_vector_a = tangential_scalar_a * unit_v
                    normal_vector_a = accel_vector - tangential_vector_a
                    tangential_color = (0, 255, 0) if tangential_scalar_a >= 0 else (0, 0, 255)
                    cv2.arrowedLine(frame, draw_pos, tuple((draw_pos + tangential_vector_a * args.acc_scale).astype(int)), tangential_color, args.thickness, tipLength=0.3)
                    cv2.arrowedLine(frame, draw_pos, tuple((draw_pos + normal_vector_a * args.acc_scale).astype(int)), (255, 0, 255), args.thickness, tipLength=0.3)
                    # 力の成分
                    if force_info and force_info.get('force_x') is not None:
                        force_vector = np.array([force_info['force_x'], force_info['force_y']])
                        tangential_force_scalar = np.dot(force_vector, unit_v)
                        normal_force_scalar = np.linalg.norm(force_vector - (tangential_force_scalar * unit_v))
                        cv2.putText(frame, f"Ft: {tangential_force_scalar:.2f} N", (20, height - 50), font, args.font_scale, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Fn: {normal_force_scalar:.2f} N", (20, height - 20), font, args.font_scale, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            # --- 通常モード ---
            if prev_coord and next_coord and prev_coord.get('x') is not None and next_coord.get('x') is not None:
                velocity_vector = np.array([next_coord['x'] - prev_coord['x'], next_coord['y'] - prev_coord['y']])
                vel_end_point = tuple((draw_pos + velocity_vector * args.vel_scale).astype(int))
                cv2.arrowedLine(frame, draw_pos, vel_end_point, tuple(args.vel_color), args.thickness, tipLength=0.3)
            if accel_info and accel_info.get('ax') is not None:
                accel_vector = np.array([accel_info['ax'], accel_info['ay']])
                acc_end_point = tuple((draw_pos + accel_vector * args.acc_scale).astype(int))
                cv2.arrowedLine(frame, draw_pos, acc_end_point, tuple(args.acc_color), args.thickness, tipLength=0.3)
    else:
        # --- データが見つからなかった場合の描画 (TRACKING LOST) ---
        text = "TRACKING LOST"; font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_pos_x = draw_pos[0] - text_size[0] // 2
        text_pos_y = draw_pos[1] + text_size[1] // 2
        text_pos_x = max(0, min(text_pos_x, width - text_size[0]))
        text_pos_y = max(text_size[1], min(text_pos_y, height))
        cv2.putText(frame, (text), (text_pos_x, text_pos_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="動画にベクトルと力の情報を描画します。")
    # (引数定義は変更なし)
    parser.add_argument("--video", required=True)
    parser.add_argument("--coords", required=True)
    parser.add_argument("--accels", required=True)
    parser.add_argument("--forces", required=True)
    parser.add_argument("--output", required=True)
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument("--follow", action="store_true")
    view_group.add_argument("--center-on", nargs=2, type=int, metavar=('X', 'Y'))
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--acc-scale", type=float, default=100.0)
    parser.add_argument("--vel-scale", type=float, default=3.0)
    parser.add_argument("--acc-color", nargs=3, type=int, default=[0, 0, 255])
    parser.add_argument("--vel-color", nargs=3, type=int, default=[255, 0, 0])
    parser.add_argument("--dot-color", nargs=3, type=int, default=[0, 255, 0])
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.8)
    parser.add_argument("--font-color", nargs=3, type=int, default=[255, 255, 255])
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--x-offset", type=int, default=0)
    parser.add_argument("--y-offset", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # (以降のmain関数ロジックは変更なし)
    all_data = {'coords': load_csv_data_to_dict(args.coords), 'accels': load_csv_data_to_dict(args.accels), 'forces': load_csv_data_to_dict(args.forces)}
    if not all_data['coords']: sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        output_avi = os.path.splitext(args.output)[0] + '.avi'
        writer = cv2.VideoWriter(output_avi, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
        if not writer.isOpened(): print(f"❌ エラー: AVI形式でも出力動画ファイルが作成できません。"); sys.exit(1)
        args.output = output_avi

    camera_center = np.array([width / 2, height / 2], dtype=float)
    follow_cam_initialized = False
    last_known_position = None
    
    print("\n--- 動画処理開始 ---")
    try:
        for frame_number in tqdm(range(total_frames), desc="動画処理中", unit="フレーム"):
            success, frame = cap.read()
            if not success: break
            
            coord_info = all_data['coords'].get(frame_number)
            object_pos = None
            if coord_info and coord_info.get('x') is not None:
                corrected_x = coord_info['x'] + args.x_offset
                corrected_y = coord_info['y'] + args.y_offset
                object_pos = np.array([corrected_x, corrected_y], dtype=float)
                last_known_position = tuple(object_pos.astype(int))

            shift = np.array([0.0, 0.0])
            if args.follow:
                screen_center = np.array([width / 2, height / 2])
                if object_pos is not None:
                    if not follow_cam_initialized:
                        camera_center = object_pos; follow_cam_initialized = True
                    else:
                        camera_center += (object_pos - camera_center) * args.smoothing
                shift = screen_center - camera_center
            elif args.center_on:
                target_pos = np.array(args.center_on, dtype=float)
                shift = np.array([width / 2, height / 2]) - target_pos

            if np.any(shift != 0):
                M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
                frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

            draw_vectors_on_frame(frame, frame_number, all_data, args, last_known_position, shift)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
        print(f"\n処理完了！\n出力動画を '{args.output}' に保存しました。")

if __name__ == '__main__':
    main()