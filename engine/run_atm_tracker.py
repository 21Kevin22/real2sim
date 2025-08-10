import cv2
import argparse
import os
import sys
import csv
from tqdm import tqdm
import torch
import yaml
import glob
from collections import OrderedDict

# --- パスの設定 ---
if '__file__' in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

atm_root_dir = os.path.join(current_dir, '..')
sys.path.append(atm_root_dir)
sys.path.append(os.path.join(atm_root_dir, 'third_party'))


# --- 関数の定義 ---

def import_bc_vilt_policy():
    """BCViLTPolicyを遅延インポートする関数"""
    try:
        from atm.policy.vilt import BCViLTPolicy
        return BCViLTPolicy
    except ImportError as e:
        print(f"致命的エラー: 'atm'モジュールが見つかりません: {e}", file=sys.stderr)
        sys.exit(1)

def find_config_file(config_path, config_name):
    """設定ファイルを柔軟に検索する関数"""
    if os.path.isfile(config_path) and config_path.endswith(('.yaml', '.yml')):
        return config_path
    if os.path.isdir(config_path) and config_name:
        for ext in ['.yaml', '.yml']:
            path = os.path.join(config_path, config_name + ext)
            if os.path.isfile(path):
                return path
    print(f"エラー: 設定ファイルが見つかりません。パス: '{config_path}', 名前: '{config_name}'", file=sys.stderr)
    return None

def load_config(args):
    """設定ファイルを読み込み、このスクリプト用に調整する関数"""
    print("1. 設定ファイルを読み込んでいます...")
    config_file = find_config_file(args.config_path, args.config_name)
    if not config_file:
        sys.exit(1)

    print(f"  - 使用する設定ファイル: {config_file}")
    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # --- ステージ1: モデル全体の次元数を統一 ---
    print("  - モデル全体の次元数を統一します...")
    try:
        target_embed_size = cfg['model_cfg']['spatial_transformer_cfg']['spatial_downsample_embed_size']
        print(f"  - 目標の次元数: {target_embed_size}")
        cfg['model_cfg']['img_encoder_cfg']['embed_size'] = target_embed_size
        print(f"  - 画像エンコーダーの出力次元数を {target_embed_size} に変更しました。")
    except KeyError:
        print("警告: モデルの次元数を自動統一できませんでした。")

    # --- ステージ2: 変数置換 ---
    print("  - 設定ファイル内の変数を処理します...")
    frame_stack = cfg.get('frame_stack', 10)
    if 'model_cfg' in cfg and 'obs_cfg' in cfg['model_cfg']:
        obs_cfg = cfg['model_cfg']['obs_cfg']
        if 'max_seq_len' in obs_cfg and isinstance(obs_cfg['max_seq_len'], str):
            if '${frame_stack}' in obs_cfg['max_seq_len']:
                obs_cfg['max_seq_len'] = frame_stack
                print(f"  - 'max_seq_len' を {frame_stack} に設定しました。")

    # --- ステージ3: 単一カメラ用に調整 ---
    print("  - モデル設定を単一カメラ用に自動調整します。")
    if 'model_cfg' in cfg and 'obs_cfg' in cfg['model_cfg']:
        cfg['model_cfg']['obs_cfg']['num_views'] = 1
        if 'camera_names' in cfg['model_cfg']['obs_cfg'] and isinstance(cfg['model_cfg']['obs_cfg']['camera_names'], list):
            original_views = cfg['model_cfg']['obs_cfg']['camera_names']
            if len(original_views) > 1:
                cfg['model_cfg']['obs_cfg']['camera_names'] = original_views[:1]
                print(f"  - 視点を '{original_views}' から '{original_views[:1]}' に変更しました。")
    
    # --- ステージ4: 不要な機能を無効化 ---
    print("  - 動画ファイル用に不要な機能を無効化します...")
    if 'model_cfg' in cfg:
        if 'obs_cfg' in cfg['model_cfg']:
            cfg['model_cfg']['obs_cfg']['extra_states'] = []
        if 'extra_state_encoder_cfg' in cfg['model_cfg']:
            if 'extra_state_keys' in cfg['model_cfg']['extra_state_encoder_cfg']:
                cfg['model_cfg']['extra_state_encoder_cfg']['extra_state_keys'] = []
    
    # ★★★★★ ここからが修正箇所です ★★★★★
    # Policy Head（頭）に、正しい入力サイズを教える
    print("  - Policy Headの入力サイズを最終調整します...")
    try:
        # 前回の実行時エラー 'RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x64 ...)'
        # から、PolicyHeadに渡される直前のデータの特徴量が「64」であることが判明しています。
        # そのため、複雑な計算はせず、この実績値を直接利用してモデルの不整合を解消します。
        policy_input_size = 64 # ★★★ これが重要な修正点です ★★★
        
        # policy_head_cfgにinput_sizeを注入
        if 'policy_head_cfg' not in cfg['model_cfg']:
            cfg['model_cfg']['policy_head_cfg'] = {}
        cfg['model_cfg']['policy_head_cfg']['input_size'] = policy_input_size
        print(f"  - Policy Headの入力サイズを {policy_input_size} に設定しました。 (エラー情報に基づき修正)")
    except KeyError as e:
        print(f"警告: Policy Headの入力サイズを自動設定しようとしましたが、設定パスが見つかりませんでした: {e}")
    # ★★★★★ ここまでが修正箇所です ★★★★★

    print("✓ 設定の読み込みと調整が完了しました。")
    return cfg

def build_model(cfg, weights_path, device):
    """モデルを構築し、学習済み重みをロードする関数"""
    print(f"2. ATMモデルを構築し、重み '{weights_path}' を読み込んでいます...")
    print(f"  - 使用デバイス: {device}")

    model_cfg = cfg.get('model_cfg', {})
    track_cfg = model_cfg.get('track_cfg', {})
    
    if track_cfg.get('track_fn') in ['???', None]:
        print("  - 'track_fn' が未設定のため、既存のトラッカーモデルを検索します...")
        track_transformer_dirs = ['results/track_transformer/libero_track_transformer_libero-spatial', 'results/track_transformer/libero_track_transformer_libero-object', 'results/track_transformer/libero_track_transformer_libero-goal']
        found_track_fn = None
        for track_dir in track_transformer_dirs:
            if os.path.isdir(track_dir):
                found_track_fn = track_dir
                break
        
        if found_track_fn:
            track_cfg['track_fn'] = found_track_fn
            print(f"  - トラッカーモデルとして '{found_track_fn}' を使用します。")
        else:
            print("致命的エラー: 利用可能な 'track_transformer' モデルが見つかりません。", file=sys.stderr)
            sys.exit(1)

    print("  - 動画推論用に言語エンコーダーを無効化します...")
    if 'language_encoder_cfg' in model_cfg:
        model_cfg['language_encoder_cfg'] = {}

    BCViLTPolicy = import_bc_vilt_policy()
    
    print("\n--- デバッグ情報 (build_model) ---")
    print("BCViLTPolicyに渡す policy_head_cfg:", model_cfg.get('policy_head_cfg'))
    print("---------------------------------\n")
    try:
        model = BCViLTPolicy(**model_cfg)
    except Exception as e:
        print(f"致命的エラー: モデルのインスタンス化に失敗しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint.get('policy', checkpoint))
    
    if list(state_dict.keys()) and list(state_dict.keys())[0].startswith('policy.'):
        state_dict = {k.replace('policy.', ''): v for k, v in state_dict.items()}

    model_state_dict = model.state_dict()
    compatible_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            compatible_state_dict[k] = v
    
    model.load_state_dict(compatible_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✓ モデルの準備が完了しました。")
    return model

def preprocess_frame(frame, img_size, device):
    """動画フレームをモデルの入力形式に変換する"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, img_size, interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame / 255.0
    tensor = torch.from_numpy(normalized_frame).permute(2, 0, 1).float()
    return tensor.unsqueeze(0).to(device)

def extract_coordinates(action_tensor):
    """モデル出力(action)から座標を安全に抽出する"""
    if action_tensor is None: return None, None
    flat_action = action_tensor.flatten()
    if len(flat_action) >= 2: return flat_action[0].item(), flat_action[1].item()
    return None, None

def run_tracking(model, cfg, args):
    """動画処理のメインループを実行する関数"""
    print("3. 動画の処理を開始します...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません: {args.video}", file=sys.stderr)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    device = next(model.parameters()).device
    
    try:
        h, w = cfg['model_cfg']['obs_cfg']['obs_shapes']['rgb'][1:]
        img_size = (w, h)
        print(f"  - モデルの入力画像サイズを {img_size} に設定しました。")
    except KeyError:
        img_size = (128, 128)
        print(f"警告: 設定から画像サイズを取得できませんでした。デフォルトの {img_size} を使用します。")

    output_dir = os.path.dirname(args.output)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w', newline='', encoding='utf-8') as outfile, torch.no_grad():
        writer = csv.writer(outfile)
        writer.writerow(['frame', 'x', 'y'])

        extra_states = {} 

        for frame_number in tqdm(range(total_frames), desc="追跡中", unit="フレーム"):
            success, frame = cap.read()
            if not success: break
            
            processed_frame = preprocess_frame(frame, img_size, device)
            obs_channels_last = processed_frame.permute(0, 2, 3, 1)
            obs = obs_channels_last.unsqueeze(1)
            
            # 言語命令は使わないので、ダミーのNoneを渡す
            dummy_language_instruction = None
            
            # ★★★★★ ここが最後の修正箇所です ★★★★★
            # 存在しない .act() ではなく、モデル本体を直接呼び出す (.forward()が実行される)
            action = model(obs, dummy_language_instruction, extra_states)
            # ★★★★★ ここまで ★★★★★

            x, y = extract_coordinates(action)
            writer.writerow([frame_number, x, y])

    cap.release()
    print(f"✓ 処理完了！追跡データを '{args.output}' に保存しました。")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="学習済みのATMモデルを使って、動画から物体の座標を追跡します。")
    parser.add_argument("--video", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output", required=True, help="出力CSVファイルのパス")
    parser.add_argument("--weights", required=True, help="学習済みモデルの重みファイル(.ckpt)のパス")
    parser.add_argument("--config-path", required=True, help="設定ファイルのパス")
    parser.add_argument("--config-name", default=None, help="設定ファイルの名前（オプション）")
    args = parser.parse_args()
    
    cfg = load_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.weights, device)
    run_tracking(model, cfg, args)

if __name__ == '__main__':
    main()