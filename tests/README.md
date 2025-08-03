# テストドキュメント

このディレクトリには、MIP-MCPプロジェクトのテストスイートが含まれています。

## テスト構造

```
tests/
├── __init__.py              # テストパッケージ
├── conftest.py              # pytest設定と共通フィクスチャ
├── README.md                # このファイル
├── unit/                    # ユニットテスト
│   ├── test_basic.py        # 基本的なインポートとモデルテスト
│   ├── test_models.py       # データモデルのテスト
│   ├── test_handlers.py     # MCPハンドラーのテスト
│   ├── test_pyodide_executor.py  # Pyodide実行エンジンのテスト
│   ├── test_scip_solver.py  # SCIPソルバーのテスト
│   └── test_utils.py        # ユーティリティ関数のテスト
├── integration/             # 統合テスト
│   ├── test_mcp_server.py   # MCPサーバーの統合テスト
│   └── test_end_to_end.py   # エンドツーエンドテスト
└── fixtures/                # テスト用サンプルデータ
    └── sample_problems.py   # 最適化問題のサンプル
```

## テスト実行

### 基本テスト（推奨）
```bash
# 動作確認済みの基本テストのみ実行
make test-basic
```

### 全テスト実行
```bash
# 全ユニットテスト実行
make test-unit

# 全テスト実行（統合テストを含む）
make test
```

### カバレッジレポート
```bash
# 基本テストのカバレッジ
make coverage

# 全テストのカバレッジ
make coverage-all
```

## テストの種類

### ユニットテスト

- **test_basic.py**: 基本的な機能とモジュールインポートのテスト
- **test_models.py**: Pydanticデータモデルのテスト（設定、解、レスポンス）
- **test_handlers.py**: MCPツールハンドラーのテスト
- **test_pyodide_executor.py**: Pyodideセキュア実行エンジンのテスト
- **test_scip_solver.py**: SCIP最適化ソルバーのテスト
- **test_utils.py**: ユーティリティ関数のテスト

### 統合テスト

- **test_mcp_server.py**: MCPサーバー全体の統合テスト
- **test_end_to_end.py**: 完全なワークフローのエンドツーエンドテスト

## テスト設定

### 環境変数
- `TESTING=1`: テストモードを有効化
- `MCP_MODE=0`: テスト中はMCPモードを無効化

### フィクスチャ
- `temp_config_dir`: 一時設定ディレクトリ
- `mock_config`: モック設定オブジェクト
- `sample_pulp_code`: サンプルPuLPコード
- `mock_mcp_context`: モックMCPコンテキスト

## 現在の状況

### 動作中のテスト ✅
- 基本機能テスト（test_basic.py）
- データモデルテスト（test_models.py）

### 開発中のテスト 🚧
- ハンドラーテスト（実際のAPIとの整合性調整中）
- Pyodide実行エンジンテスト（Node.js依存関係の解決中）
- SCIPソルバーテスト（pyscipoptモックの調整中）
- ユーティリティテスト（YAML依存関係の解決中）

### 統合テスト 📋
- FastMCP依存関係のため、統合テストは基本インフラ構築後に実装予定

## テスト実行のベストプラクティス

1. **開発時**: `make test-basic`で動作確認
2. **CI/CD**: 全テストが安定したら`make test`に移行
3. **カバレッジ**: 定期的に`make coverage`でカバレッジを確認
4. **リファクタリング時**: 影響範囲に応じてテストレベルを選択

## 今後の改善点

1. モックの改善（実際のAPIシグネチャとの整合性）
2. 非同期テストのパフォーマンス最適化
3. 統合テスト環境の自動化
4. テストデータの管理改善
5. セキュリティテストの強化
