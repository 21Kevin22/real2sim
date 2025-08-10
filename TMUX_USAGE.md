# ATM tmuxセッション使用方法

このドキュメントでは、tmuxを使用してATM（Any-point Trajectory Modeling）を実行する方法を説明します。

## 前提条件

1. ATMがインストールされていること
2. conda環境`atm`が作成されていること
3. tmuxがインストールされていること

## セッション開始方法

### 1. 基本的なセッション開始

```bash
cd ATM
./start_atm_session.sh
```

これにより、インタラクティブモードでATMセッションが開始されます。

### 2. 特定のスイートでセッション開始

```bash
# libero_spatialスイートでセッション開始
./start_atm_session.sh libero_spatial

# libero_objectスイートでセッション開始
./start_atm_session.sh libero_object

# libero_goalスイートでセッション開始
./start_atm_session.sh libero_goal
```

### 3. トレーニングモードでセッション開始

```bash
# libero_spatialスイートでトレーニングモード開始
./start_atm_session.sh libero_spatial training
```

### 4. 評価モードでセッション開始

```bash
# libero_spatialスイートで評価モード開始
./start_atm_session.sh libero_spatial evaluation
```

## セッション管理

### セッションにアタッチ

```bash
tmux attach-session -t ATM
```

### セッションからデタッチ

セッション内で `Ctrl+B, D` を押す

### セッションを終了

```bash
tmux kill-session -t ATM
```

### セッション一覧表示

```bash
tmux list-sessions
```

## 使用可能なコマンド

セッション内で以下のコマンドが使用できます：

### 1. Track-guided Policyトレーニング

```bash
python -m scripts.train_libero_policy_atm --suite libero_spatial --tt results/track_transformer/libero_track_transformer_libero-spatial/
```

### 2. Vanilla BCトレーニング

```bash
python -m scripts.train_libero_policy_bc --suite libero_spatial
```

### 3. モデル評価

```bash
python -m scripts.eval_libero_policy --suite libero_spatial --exp results/policy/[experiment_name]
```

### 4. トラッキング実行

```bash
python engine/run_atm_tracker.py --video [video_path] --output [output.csv] --weights [model.pth] --config-path conf/train_bc --config-name libero_vilt
```

## トラブルシューティング

### セッションが既に存在する場合

スクリプトは自動的に既存のセッションを終了して新しいセッションを作成します。

### conda環境がアクティブにならない場合

セッション内で手動でconda環境をアクティブにしてください：

```bash
conda activate atm
```

### パスが正しくない場合

セッション内でATMディレクトリに移動してください：

```bash
cd /path/to/ATM
```

## 注意事項

1. 長時間のトレーニングを実行する場合は、セッションからデタッチしてバックグラウンドで実行することをお勧めします
2. セッションを終了する前に、実行中のプロセスが完了していることを確認してください
3. 重要なデータは定期的にバックアップしてください

## サポート

問題が発生した場合は、以下を確認してください：

1. ATMのインストールが正しく完了しているか
2. conda環境`atm`が存在するか
3. 必要な依存関係がインストールされているか
4. tmuxがインストールされているか 