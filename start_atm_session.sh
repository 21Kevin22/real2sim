#!/bin/bash

# ATMセッションを開始するスクリプト
# 使用方法: ./start_atm_session.sh [suite_name] [mode]

SESSION_NAME="ATM"
SUITE_NAME=${1:-"libero_spatial"}
MODE=${2:-"interactive"}

echo "=== ATMセッション開始スクリプト ==="
echo "セッション名: $SESSION_NAME"
echo "スイート名: $SUITE_NAME"
echo "モード: $MODE"
echo ""

# 現在のディレクトリをATMディレクトリに変更
cd "$(dirname "$0")"

# 既存のセッションが存在する場合は削除
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "既存のセッション '$SESSION_NAME' を終了しています..."
    tmux kill-session -t $SESSION_NAME
fi

# 新しいセッションを作成
echo "新しいtmuxセッション '$SESSION_NAME' を作成しています..."
tmux new-session -d -s $SESSION_NAME

# セッション内でATMの環境をセットアップ
tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
tmux send-keys -t $SESSION_NAME "conda activate atm" C-m
tmux send-keys -t $SESSION_NAME "clear" C-m

# 使用可能なコマンドのヘルプを表示
tmux send-keys -t $SESSION_NAME "echo '=== ATM環境が準備されました ==='" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '使用可能なコマンド:'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '1. トレーニング (Track-guided Policy):'" C-m
tmux send-keys -t $SESSION_NAME "echo '   python -m scripts.train_libero_policy_atm --suite $SUITE_NAME --tt results/track_transformer/libero_track_transformer_${SUITE_NAME//_/-}/'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '2. トレーニング (Vanilla BC):'" C-m
tmux send-keys -t $SESSION_NAME "echo '   python -m scripts.train_libero_policy_bc --suite $SUITE_NAME'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '3. 評価:'" C-m
tmux send-keys -t $SESSION_NAME "echo '   python -m scripts.eval_libero_policy --suite $SUITE_NAME --exp results/policy/[experiment_name]'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '4. トラッキング実行:'" C-m
tmux send-keys -t $SESSION_NAME "echo '   python engine/run_atm_tracker.py --video [video_path] --output [output.csv] --weights [model.pth] --config-path conf/train_bc --config-name libero_vilt'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '=== セッション管理 ==='" C-m
tmux send-keys -t $SESSION_NAME "echo 'セッションからデタッチ: Ctrl+B, D'" C-m
tmux send-keys -t $SESSION_NAME "echo 'セッション終了: tmux kill-session -t $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME "echo 'セッション再アタッチ: tmux attach-session -t $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# モードに応じて追加のセットアップ
if [ "$MODE" = "training" ]; then
    echo "トレーニングモードでセッションを開始します..."
    tmux send-keys -t $SESSION_NAME "echo 'トレーニングモード: Track-guided Policyのトレーニングを開始します...'" C-m
    tmux send-keys -t $SESSION_NAME "python -m scripts.train_libero_policy_atm --suite $SUITE_NAME --tt results/track_transformer/libero_track_transformer_${SUITE_NAME//_/-}/" C-m
elif [ "$MODE" = "evaluation" ]; then
    echo "評価モードでセッションを開始します..."
    tmux send-keys -t $SESSION_NAME "echo '評価モード: 既存のモデルを評価します...'" C-m
    tmux send-keys -t $SESSION_NAME "echo '評価する実験名を指定してください:'" C-m
    tmux send-keys -t $SESSION_NAME "ls results/policy/" C-m
else
    echo "インタラクティブモードでセッションを開始します..."
    tmux send-keys -t $SESSION_NAME "echo 'インタラクティブモード: コマンドを手動で実行してください。'" C-m
fi

# セッションにアタッチ
echo ""
echo "tmuxセッション '$SESSION_NAME' にアタッチしています..."
echo "セッションを終了するには: tmux kill-session -t $SESSION_NAME"
echo "セッションからデタッチするには: Ctrl+B, D"
echo ""
tmux attach-session -t $SESSION_NAME 