import pandas as pd
import numpy as np
import argparse
import os

def load_and_validate_data(filepath, required_cols):
    """
    CSVファイルをDataFrameとして読み込み、必要な列が存在するか検証する。
    """
    if not os.path.exists(filepath):
        print(f"❌ エラー: ファイルが見つかりません: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        # 必須列の存在チェック
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ エラー: '{filepath}'に必須列 '{col}' が見つかりません。")
                return None
        return df
    except Exception as e:
        print(f"❌ エラー: '{filepath}' の読み込み中に問題が発生しました: {e}")
        return None

def estimate_force(accel_file, area_file, output_file, density):
    """
    加速度と面積のデータから力を推定し、結果をCSVに出力する。
    """
    # --- データの読み込みと検証 ---
    print("🚚 CSVデータを読み込んでいます...")
    accel_df = load_and_validate_data(accel_file, ['frame', 'ax', 'ay'])
    area_df = load_and_validate_data(area_file, ['frame', 'area_cm2'])

    if accel_df is None or area_df is None:
        print("🛑 データの読み込みに失敗したため、処理を中断します。")
        return

    print(f"✅ 加速度データ: {len(accel_df)} フレーム")
    print(f"✅ 面積データ: {len(area_df)} フレーム")
    print(f"⚙️ 使用する面密度: {density} g/cm^2")

    # ★★★【診断機能 1】★★★
    # 有効な面積データ（0より大きい）が存在するかを事前にチェック
    if area_df['area_cm2'].max() <= 0:
        print("\n" + "="*50)
        print("⚠️ 警告: 入力された 'areas.csv' には、面積が0より大きい有効なデータが1件も含まれていません。")
        print("      これにより、計算される力はすべて0になります。")
        print("      面積を検出する前のスクリプトの設定（--min-areaなど）を見直してください。")
        print("="*50 + "\n")

    # --- データの結合と計算 ---
    print("🔬 力の計算を開始します...")
    # 'frame'をキーとして2つのデータを結合
    merged_df = pd.merge(accel_df, area_df, on='frame', how='inner')
    
    total_processed_frames = len(merged_df)
    
    # 面積が0以下のフレームでも計算自体は行う（結果は0になる）
    merged_df['mass_g'] = merged_df['area_cm2'] * density
    merged_df['force_x'] = merged_df['mass_g'] * merged_df['ax']
    merged_df['force_y'] = merged_df['mass_g'] * merged_df['ay']
    merged_df['force_magnitude'] = np.sqrt(merged_df['force_x']**2 + merged_df['force_y']**2)

    # 実際に力が計算された（面積が0より大きかった）フレーム数をカウント
    calculated_frames_count = len(merged_df[merged_df['area_cm2'] > 0])

    # --- 結果の出力 ---
    try:
        output_df = merged_df[['frame', 'mass_g', 'force_x', 'force_y', 'force_magnitude']]
        # 結果を小数点以下4桁に丸める
        output_df = output_df.round(4)
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n🎉 処理が完了しました。力の推定データを '{output_file}' に保存しました。")
    except Exception as e:
        print(f"❌ エラー: 出力ファイル '{output_file}' の書き込み中にエラーが発生しました: {e}")

    # ★★★【診断機能 3】★★★
    # 最終的なサマリーを表示
    print("\n" + "="*20 + " 処理結果サマリー " + "="*20)
    print(f"🤝 処理対象となった総フレーム数 (両ファイルに共通): {total_processed_frames}")
    print(f"💪 実際に力が計算されたフレーム数 (面積 > 0): {calculated_frames_count}")
    if total_processed_frames > 0 and calculated_frames_count == 0:
        print("▶︎▶︎▶︎ ❗️問題の可能性: 力が計算されたフレームが0でした。入力の 'areas.csv' の 'area_cm2' 列の値を確認してください。")
    elif total_processed_frames > calculated_frames_count > 0:
        print("▶︎▶︎▶︎ ℹ️ 情報: 一部のフレームでは面積が0または負の値だったため、力の計算結果が0になっています。")
    elif total_processed_frames == 0:
         print("▶︎▶︎▶︎ ❗️問題の可能性: 加速度と面積のファイルで共通するフレーム番号が1件もありませんでした。")
    else:
        print("▶︎▶︎▶︎ ✅ 正常: 多くのフレームで力が計算されました。")
    print("="*58)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="加速度と面積のデータから力を推定します。")
    parser.add_argument("--accel", required=True, help="加速度データが記録されたCSVファイル (accelerations.csv)")
    parser.add_argument("--area", required=True, help="面積データが記録されたCSVファイル (areas.csv)")
    parser.add_argument("--output", required=True, help="推定した力を保存する出力CSVファイル")
    parser.add_argument("--density", type=float, required=True, help="物体の面密度 (例: g/cm^2)")
    args = parser.parse_args()

    estimate_force(args.accel, args.area, args.output, args.density)