# Solver Selection Design

## Problem
現在、ユーザーはソルバーパラメータ（solver_params）を設定できるが、どのソルバーを使用するかを指定できない。execute_mip_codeツールはハードコードでSCIPソルバーのみを使用している。

## Solution Design

### 1. APIパラメータの追加
`execute_mip_code`ツールに`solver`パラメータを追加：
- Type: Optional[str] = None
- Default: None (設定ファイルのdefaultを使用)
- Options: "scip" (現在は1つのみサポート)

### 2. SolverFactoryパターンの実装
ソルバー選択のためのファクトリークラスを作成：
```python
class SolverFactory:
    @staticmethod
    def create_solver(solver_name: str, config: Dict[str, Any]) -> BaseSolver:
        if solver_name.lower() == "scip":
            return SCIPSolver(config)
        else:
            raise ValueError(f"Unsupported solver: {solver_name}")
```

### 3. 設定からのデフォルトソルバー取得
設定ファイル（default.yaml）の`solvers.default`を使用してデフォルトソルバーを決定

### 4. バックワード互換性
- 既存のコードは変更なしで動作し続ける
- solver パラメータが未指定の場合はSCIPを使用

## Implementation Steps
1. SolverFactoryクラスの実装
2. execute_mip_code_handlerの更新（solverパラメータ追加）
3. MCP tool定義の更新
4. テストの追加
