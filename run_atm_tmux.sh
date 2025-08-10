#!/bin/bash

# ATMをtmuxセッションで実行するスクリプト

# セッション名
SESSION_NAME="ATM"

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
tmux send-keys -t $SESSION_NAME "echo 'ATM環境が準備されました。'" C-m
tmux send-keys -t $SESSION_NAME "echo '使用可能なコマンド:'" C-m
tmux send-keys -t $SESSION_NAME "echo '  - python -m scripts.train_libero_policy_atm --suite libero_spatial --tt results/track_transformer/libero_track_transformer_libero-spatial/'" C-m
tmux send-keys -t $SESSION_NAME "echo '  - python -m scripts.train_libero_policy_bc --suite libero_spatial'" C-m
tmux send-keys -t $SESSION_NAME "echo '  - python -m scripts.eval_libero_policy --suite libero_spatial --exp results/policy/...'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# セッションにアタッチ
echo "tmuxセッション '$SESSION_NAME' にアタッチしています..."
echo "セッションを終了するには: tmux kill-session -t $SESSION_NAME"
echo "セッションからデタッチするには: Ctrl+B, D"
echo ""
tmux attach-session -t $SESSION_NAME 