# Pyodideベース MIP-MCP サーバー アーキテクチャ設計

## 概要

既存のカスタムサンドボックスをPyodideベースの安全なWebAssembly実行環境に置き換える。ツール名は汎用的なMIP対応のまま維持し、現在はPuLPのみサポートする。

## アーキテクチャ

### 1. **Pyodide実行エンジン** (`PyodideExecutor`)

```typescript
interface PyodideExecutor {
  // Pyodide環境の初期化（遅延読み込み）
  initialize(): Promise<void>

  // MIPコード実行（PuLP対応）
  executeMIPCode(code: string, options: ExecutionOptions): Promise<ExecutionResult>

  // ライブラリ検出
  detectLibrary(code: string): MIPLibrary

  // パフォーマンス最適化
  warmup(): Promise<void>
  dispose(): void
}
```

### 2. **ライブラリ検出システム** (`LibraryDetector`)

```python
class LibraryDetector:
    def detect_library(self, code: str) -> MIPLibrary:
        """コードからMIPライブラリを自動検出"""
        if 'import pulp' in code or 'from pulp' in code:
            return MIPLibrary.PULP
        elif 'import mip' in code or 'from mip' in code:
            return MIPLibrary.PYTHON_MIP  # 将来対応
        else:
            return MIPLibrary.UNKNOWN
```

### 3. **問題情報抽出** (`ProblemExtractor`)

```python
class ProblemExtractor:
    def extract_problem_info(self, code: str, pyodide_globals: dict) -> ProblemInfo:
        """実行後のグローバル変数から問題情報を抽出"""
        # 1. 変数方式（推奨）
        if '__mps_content__' in pyodide_globals:
            return ProblemInfo(format='mps', content=pyodide_globals['__mps_content__'])
        elif '__lp_content__' in pyodide_globals:
            return ProblemInfo(format='lp', content=pyodide_globals['__lp_content__'])

        # 2. 自動検出方式
        problems = self._detect_problems(pyodide_globals)
        if problems:
            return self._generate_format(problems[0])

        return ProblemInfo(format=None, content=None)
```

### 4. **MCPツール統合**

既存のツール名を維持しつつ、内部実装をPyodideに変更：

```python
class MIPMCPTools:
    # ツール名: execute_mip_code (汎用的)
    # 実装: PuLPサポート、将来的に他ライブラリ対応可能
    async def execute_mip_code(self, code: str) -> ExecutionResult

    # ツール名: validate_mip_code (汎用的)
    # 実装: 現在はPuLP構文検証
    async def validate_mip_code(self, code: str) -> ValidationResult

    # その他既存ツールも同様に汎用的な名前を維持
```

## 実装詳細

### Phase 1: Pyodide実行エンジン

1. **Node.js + Pyodide統合**
   ```typescript
   class PyodideExecutor {
     private pyodide: PyodideInterface | null = null

     async initialize() {
       if (!this.pyodide) {
         this.pyodide = await loadPyodide()
         await this.pyodide.loadPackage("micropip")
         await this.installMIPLibraries()
       }
     }

     private async installMIPLibraries() {
       await this.pyodide.runPythonAsync(`
         import micropip
         await micropip.install('pulp')
       `)
     }
   }
   ```

2. **セキュアな実行環境**
   ```python
   # Pyodide内部で実行される安全なコード実行ラッパー
   def secure_execute_mip_code(user_code: str) -> dict:
       # グローバル名前空間を制限
       safe_globals = {
           '__builtins__': {},
           'pulp': pulp,
           # その他必要最小限のみ
       }

       try:
           exec(user_code, safe_globals)
           return {
               'success': True,
               'globals': safe_globals,
               'error': None
           }
       except Exception as e:
           return {
               'success': False,
               'globals': {},
               'error': str(e)
           }
   ```

### Phase 2: 問題情報抽出

1. **変数ベース抽出（推奨）**
   ```python
   # ユーザーコード例
   import pulp
   prob = pulp.LpProblem("example", pulp.LpMaximize)
   # ... 問題定義 ...

   # LP形式を変数に設定（推奨方式）
   prob.writeLP("/tmp/problem.lp")
   with open("/tmp/problem.lp", "r") as f:
       __lp_content__ = f.read()
   ```

2. **自動検出（フォールバック）**
   ```python
   def auto_detect_problems(globals_dict: dict) -> List[ProblemInfo]:
       problems = []
       for name, obj in globals_dict.items():
           if hasattr(obj, 'writeLP'):  # PuLP問題オブジェクト
               problems.append(extract_from_pulp_problem(obj))
       return problems
   ```

### Phase 3: パフォーマンス最適化

1. **Pyodide事前ウォームアップ**
   ```typescript
   class PyodidePool {
     private instances: PyodideExecutor[] = []

     async warmup(count: number = 3) {
       for (let i = 0; i < count; i++) {
         const executor = new PyodideExecutor()
         await executor.initialize()
         this.instances.push(executor)
       }
     }

     async getExecutor(): Promise<PyodideExecutor> {
       return this.instances.pop() || new PyodideExecutor()
     }
   }
   ```

2. **メモリ効率化**
   - 不要なライブラリの遅延読み込み
   - 実行後のクリーンアップ
   - インスタンス再利用

## 移行計画

### Step 1: 新しいPyodide実行エンジン実装
- `src/mip_mcp/executor/pyodide_executor.py`
- Node.js統合とPyodide初期化

### Step 2: 既存ハンドラーの更新
- `src/mip_mcp/handlers/execute_code.py`の内部実装変更
- MCPツール名とインターフェースは維持

### Step 3: ライブラリ検出の統合
- `src/mip_mcp/utils/library_detector.py`をPyodide対応に更新

### Step 4: テストとパフォーマンス調整
- セキュリティテスト
- パフォーマンステスト
- エラーハンドリング強化

## 期待される効果

### セキュリティ
- ✅ **完全分離**: WebAssembly環境での実行
- ✅ **ゼロ脆弱性**: ホストシステムアクセス不可
- ✅ **検証済み**: Pyodideは本番環境で広く使用

### 機能性
- ✅ **PuLP完全サポート**: LP/MPS生成確認済み
- ✅ **拡張性**: 将来的にpython-mip等追加可能
- ✅ **互換性**: 既存MCPツール名・インターフェース維持

### パフォーマンス
- ✅ **高速起動**: 数百ms程度
- ✅ **軽量**: メモリ使用量最適化
- ✅ **スケーラブル**: Cloud Run環境での運用

## リスク軽減

### 潜在的リスク
1. **Pyodide起動時間**: ウォームアップで解決
2. **メモリ使用量**: プールとクリーンアップで管理
3. **ライブラリ制限**: PuLPサポート確認済み

### 対策
- インスタンスプール
- メモリ監視
- グレースフルフォールバック（RestrictedPython）
