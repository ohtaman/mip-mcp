# 技術設計書 - MIP MCP Server

## 1. システムアーキテクチャ

### 1.1 全体構成

```
┌─────────────────┐    streamable HTTP    ┌─────────────────┐
│   LLM Client    │ ◄──────────────────► │   MIP MCP       │
│  (Claude/GPT)   │                       │    Server       │
└─────────────────┘                       └─────────────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────┐
                                          │ Python Executor │
                                          │ (PuLP/pyomo/    │
                                          │  pyscipopt)     │
                                          └─────────────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────┐
                                          │  Solver Layer   │
                                          │ (SCIP/Gurobi/   │
                                          │  CPLEX/CBC)     │
                                          └─────────────────┘
```

### 1.2 モジュール構成

```
src/mip_mcp/
├── __init__.py
├── server.py              # MCPサーバーメイン
├── handlers/              # MCPリクエストハンドラー
│   ├── __init__.py
│   ├── solve.py          # 最適化実行ハンドラー
│   ├── execute_code.py   # Pythonコード実行ハンドラー
│   ├── validate.py       # モデル検証ハンドラー
│   └── status.py         # ステータス確認ハンドラー
├── solvers/              # ソルバー抽象化層
│   ├── __init__.py
│   ├── base.py           # ベースソルバークラス
│   ├── scip_solver.py    # pyscipopt実装
│   ├── gurobi_solver.py  # Gurobi実装（オプション）
│   └── cplex_solver.py   # CPLEX実装（オプション）
├── models/               # データモデル定義
│   ├── __init__.py
│   ├── problem.py        # 最適化問題定義
│   ├── solution.py       # ソリューション定義
│   └── config.py         # 設定データモデル
├── parsers/              # 問題形式パーサー
│   ├── __init__.py
│   ├── json_parser.py    # JSON形式パーサー
│   ├── lp_parser.py      # LP形式パーサー
│   └── mps_parser.py     # MPS形式パーサー
├── executor/             # Pythonコード実行エンジン
│   ├── __init__.py
│   ├── code_executor.py  # Pythonコード実行器
│   ├── sandbox.py        # サンドボックス環境
│   └── libraries.py      # 許可されたライブラリ管理
├── utils/                # ユーティリティ
│   ├── __init__.py
│   ├── logger.py         # ログ設定
│   ├── config_manager.py # 設定管理
│   └── validators.py     # 入力検証
└── config/               # 設定ファイル
    ├── default.yaml      # デフォルト設定
    └── solvers.yaml      # ソルバー設定
```

## 2. コアコンポーネント設計

### 2.1 MCPサーバー (server.py)

```python
from fastmcp import FastMCP
from typing import Dict, Any, Optional

class MIPMCPServer:
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigManager(config_path)
        self.solver_factory = SolverFactory(self.config)
        self.app = FastMCP("mip-mcp")
        self._register_handlers()

    def _register_handlers(self):
        # MCPツール登録
        self.app.add_tool("solve_optimization", solve_handler)
        self.app.add_tool("execute_python_code", execute_code_handler)
        self.app.add_tool("get_library_examples", get_library_examples_handler)
        self.app.add_tool("validate_model", validate_handler)
        self.app.add_tool("get_solver_status", status_handler)

    async def run(self):
        await self.app.run()
```

### 2.2 ソルバー抽象化層

#### ベースソルバークラス (solvers/base.py)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.problem import OptimizationProblem
from ..models.solution import OptimizationSolution

class BaseSolver(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    @abstractmethod
    async def solve(self, problem: OptimizationProblem) -> OptimizationSolution:
        """最適化問題を解決する"""
        pass

    @abstractmethod
    def validate_problem(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """問題の妥当性を検証する"""
        pass

    @abstractmethod
    def get_solver_info(self) -> Dict[str, Any]:
        """ソルバー情報を取得する"""
        pass

    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """ソルバーパラメータを設定する"""
        pass
```

#### SCIPソルバー実装 (solvers/scip_solver.py)

```python
import pyscipopt
from .base import BaseSolver
from ..models.problem import OptimizationProblem
from ..models.solution import OptimizationSolution

class SCIPSolver(BaseSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = pyscipopt.Model()

    async def solve(self, problem: OptimizationProblem) -> OptimizationSolution:
        try:
            # 変数定義
            variables = {}
            for var_def in problem.variables:
                variables[var_def.name] = self.model.addVar(
                    name=var_def.name,
                    vtype=var_def.type,
                    lb=var_def.lower_bound,
                    ub=var_def.upper_bound
                )

            # 制約条件追加
            for constraint in problem.constraints:
                expr = self._build_expression(constraint.expression, variables)
                self.model.addCons(expr <= constraint.rhs, name=constraint.name)

            # 目的関数設定
            obj_expr = self._build_expression(problem.objective.expression, variables)
            self.model.setObjective(obj_expr, sense=problem.objective.sense)

            # 最適化実行
            self.model.optimize()

            return self._extract_solution(variables)

        except Exception as e:
            return OptimizationSolution(
                status="error",
                message=str(e),
                objective_value=None,
                variables={}
            )

### 2.3 Pythonコード実行エンジン

#### コード実行器 (executor/code_executor.py)

```python
import ast
import sys
import io
import contextlib
from typing import Dict, Any, Optional, List
from ..models.solution import OptimizationSolution
from .sandbox import SecurityChecker
from .libraries import get_allowed_imports

class PythonCodeExecutor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.security_checker = SecurityChecker()
        self.allowed_imports = get_allowed_imports()

    async def execute_optimization_code(self, code: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Pythonコードを実行して最適化問題を解決する

        Args:
            code: 実行するPythonコード（PuLP/pyomo/pyscipopt等を使用）
            data: コードで使用するデータ

        Returns:
            最適化結果
        """
        try:
            # セキュリティチェック
            self.security_checker.validate_code(code)

            # 実行環境の準備
            namespace = self._prepare_namespace(data)

            # コード実行
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, namespace)

            # 結果の抽出
            result = self._extract_result(namespace, output.getvalue())

            return result

        except Exception as e:
            return {
                "status": "error",
                "message": f"Code execution failed: {str(e)}",
                "objective_value": None,
                "variables": {}
            }

    def _prepare_namespace(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """実行環境の名前空間を準備"""
        namespace = {
            '__builtins__': self._get_safe_builtins(),
            'data': data or {},
        }

        # 許可されたライブラリをインポート
        for lib_name, lib_module in self.allowed_imports.items():
            namespace[lib_name] = lib_module

        return namespace

    def _get_safe_builtins(self) -> Dict[str, Any]:
        """安全なbuiltins関数のみを提供"""
        safe_builtins = {
            'len', 'range', 'enumerate', 'zip', 'sum', 'min', 'max',
            'abs', 'round', 'int', 'float', 'str', 'list', 'dict',
            'tuple', 'set', 'bool', 'print'
        }

        return {name: getattr(__builtins__, name) for name in safe_builtins if hasattr(__builtins__, name)}

    def _extract_result(self, namespace: Dict[str, Any], output: str) -> Dict[str, Any]:
        """実行結果から最適化結果を抽出"""
        # PuLPの場合
        if 'pulp' in output.lower() or any('pulp' in str(v) for v in namespace.values()):
            return self._extract_pulp_result(namespace)

        # pyomoの場合
        if 'pyomo' in output.lower() or any('pyomo' in str(type(v)) for v in namespace.values()):
            return self._extract_pyomo_result(namespace)

        # pyscipoptの場合
        if 'scip' in output.lower() or any('scip' in str(type(v)) for v in namespace.values()):
            return self._extract_scip_result(namespace)

        # 汎用的な結果抽出
        return self._extract_generic_result(namespace, output)
```

#### セキュリティチェッカー (executor/sandbox.py)

```python
import ast
from typing import Set, List

class SecurityChecker:
    """コードのセキュリティチェックを行う"""

    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
        'dir', 'getattr', 'setattr', 'delattr', 'hasattr'
    }

    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'glob', 'socket',
        'urllib', 'http', 'ftplib', 'smtplib', 'multiprocessing',
        'threading', 'pickle', 'marshal', 'shelve'
    }

    def validate_code(self, code: str) -> bool:
        """コードの安全性を検証"""
        try:
            tree = ast.parse(code)
            checker = DangerousNodeVisitor()
            checker.visit(tree)

            if checker.dangerous_nodes:
                raise SecurityError(f"Dangerous operations detected: {checker.dangerous_nodes}")

            return True

        except SyntaxError as e:
            raise SecurityError(f"Syntax error in code: {str(e)}")

class DangerousNodeVisitor(ast.NodeVisitor):
    """危険なASTノードを検出する"""

    def __init__(self):
        self.dangerous_nodes = []

    def visit_Call(self, node):
        # 危険な関数呼び出しをチェック
        if isinstance(node.func, ast.Name) and node.func.id in SecurityChecker.DANGEROUS_FUNCTIONS:
            self.dangerous_nodes.append(f"Function call: {node.func.id}")
        self.generic_visit(node)

    def visit_Import(self, node):
        # 危険なモジュールのインポートをチェック
        for alias in node.names:
            if alias.name in SecurityChecker.DANGEROUS_MODULES:
                self.dangerous_nodes.append(f"Import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # from文での危険なインポートをチェック
        if node.module in SecurityChecker.DANGEROUS_MODULES:
            self.dangerous_nodes.append(f"Import from: {node.module}")
        self.generic_visit(node)

class SecurityError(Exception):
    """セキュリティ関連のエラー"""
    pass
```

#### 許可ライブラリ管理 (executor/libraries.py)

```python
from typing import Dict, Any
import importlib

def get_allowed_imports() -> Dict[str, Any]:
    """許可されたライブラリのマッピングを返す"""
    allowed_libs = {}

    # 最適化ライブラリ
    optimization_libs = [
        'pulp',
        'pyomo',
        'pyscipopt',
        'cvxpy',
        'ortools'
    ]

    # 数値計算ライブラリ
    numeric_libs = [
        'numpy',
        'pandas',
        'scipy',
        'math',
        'statistics'
    ]

    # 全てのライブラリを試行してインポート
    for lib_name in optimization_libs + numeric_libs:
        try:
            lib_module = importlib.import_module(lib_name)
            allowed_libs[lib_name] = lib_module
        except ImportError:
            # ライブラリが利用できない場合はスキップ
            continue

    return allowed_libs

def get_library_info() -> Dict[str, Dict[str, Any]]:
    """利用可能なライブラリの情報を返す"""
    return {
        'pulp': {
            'description': 'Linear Programming library for Python',
            'example': '''
import pulp

# 問題定義
prob = pulp.LpProblem("Example", pulp.LpMaximize)

# 変数定義
x = pulp.LpVariable("x", 0, None)
y = pulp.LpVariable("y", 0, None)

# 目的関数
prob += 3*x + 2*y

# 制約条件
prob += 2*x + y <= 100
prob += x + y <= 80

# 求解
prob.solve()

# 結果出力
result = {
    "status": pulp.LpStatus[prob.status],
    "objective": pulp.value(prob.objective),
    "variables": {v.name: v.varValue for v in prob.variables()}
}
'''
        },
        'pyomo': {
            'description': 'Python Optimization Modeling Objects',
            'example': '''
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# モデル作成
model = pyo.ConcreteModel()

# 変数定義
model.x = pyo.Var(bounds=(0, None))
model.y = pyo.Var(bounds=(0, None))

# 目的関数
model.obj = pyo.Objective(expr=3*model.x + 2*model.y, sense=pyo.maximize)

# 制約条件
model.constraint1 = pyo.Constraint(expr=2*model.x + model.y <= 100)
model.constraint2 = pyo.Constraint(expr=model.x + model.y <= 80)

# 求解
solver = SolverFactory('cbc')
results = solver.solve(model)

# 結果出力
result = {
    "status": str(results.solver.termination_condition),
    "objective": pyo.value(model.obj),
    "variables": {
        "x": pyo.value(model.x),
        "y": pyo.value(model.y)
    }
}
'''
        }
    }
```

### 2.4 Pythonコード実行ハンドラー (handlers/execute_code.py)

```python
from fastmcp import Context
from typing import Dict, Any, Optional
from ..executor.code_executor import PythonCodeExecutor
from ..utils.logger import get_logger

logger = get_logger(__name__)

async def execute_code_handler(
    context: Context,
    code: str,
    data: Optional[Dict[str, Any]] = None,
    library: str = "pulp"
) -> Dict[str, Any]:
    """
    Pythonコードを実行して最適化問題を解決する

    Args:
        code: 実行するPythonコード
        data: コードで使用するデータ
        library: 使用する最適化ライブラリのヒント

    Returns:
        最適化結果
    """
    try:
        executor = PythonCodeExecutor(context.config)
        result = await executor.execute_optimization_code(code, data)

        logger.info(f"Code execution completed with library {library}")
        return result

    except Exception as e:
        logger.error(f"Code execution failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "objective_value": None,
            "variables": {}
        }

async def get_library_examples_handler(context: Context) -> Dict[str, Any]:
    """
    利用可能なライブラリとサンプルコードを返す
    """
    from ..executor.libraries import get_library_info

    return {
        "libraries": get_library_info(),
        "supported_formats": ["pulp", "pyomo", "pyscipopt", "cvxpy", "ortools"]
    }
```

### 2.5 データモデル定義

#### 最適化問題モデル (models/problem.py)

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum

class VariableType(str, Enum):
    CONTINUOUS = "C"
    INTEGER = "I"
    BINARY = "B"

class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class Variable(BaseModel):
    name: str
    type: VariableType = VariableType.CONTINUOUS
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

class Constraint(BaseModel):
    name: str
    expression: Dict[str, float]  # {variable_name: coefficient}
    sense: Literal["<=", ">=", "="] = "<="
    rhs: float

class Objective(BaseModel):
    sense: ObjectiveSense
    expression: Dict[str, float]  # {variable_name: coefficient}

class OptimizationProblem(BaseModel):
    name: str
    variables: List[Variable]
    constraints: List[Constraint]
    objective: Objective
    parameters: Optional[Dict[str, Any]] = None
```

#### ソリューションモデル (models/solution.py)

```python
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class SolutionVariable(BaseModel):
    name: str
    value: float
    reduced_cost: Optional[float] = None

class SolutionConstraint(BaseModel):
    name: str
    slack: Optional[float] = None
    dual_value: Optional[float] = None

class OptimizationSolution(BaseModel):
    status: str  # optimal, infeasible, unbounded, error, etc.
    objective_value: Optional[float] = None
    variables: Dict[str, float] = {}
    constraints: List[SolutionConstraint] = []
    solve_time: Optional[float] = None
    iterations: Optional[int] = None
    message: Optional[str] = None
    solver_info: Optional[Dict[str, Any]] = None
```

### 2.6 MCPハンドラー実装

#### 最適化実行ハンドラー (handlers/solve.py)

```python
from fastmcp import Context
from typing import Dict, Any
from ..models.problem import OptimizationProblem
from ..parsers import get_parser
from ..utils.logger import get_logger

logger = get_logger(__name__)

async def solve_handler(
    context: Context,
    problem_data: Dict[str, Any],
    solver_name: str = "scip",
    solver_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    最適化問題を解決する

    Args:
        problem_data: 最適化問題データ（JSON/LP/MPS形式）
        solver_name: 使用するソルバー名
        solver_params: ソルバーパラメータ

    Returns:
        最適化結果
    """
    try:
        # 問題データのパース
        parser = get_parser(problem_data)
        problem = parser.parse(problem_data)

        # ソルバー取得
        solver = context.solver_factory.get_solver(solver_name)

        # パラメータ設定
        if solver_params:
            solver.set_parameters(solver_params)

        # 最適化実行
        solution = await solver.solve(problem)

        logger.info(f"Optimization completed: {solution.status}")

        return solution.dict()

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "objective_value": None,
            "variables": {}
        }
```

## 3. 設定管理システム

### 3.1 設定ファイル構造

#### default.yaml
```yaml
server:
  name: "mip-mcp"
  version: "0.1.0"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

solvers:
  default: "scip"
  timeout: 3600  # seconds

executor:
  enabled: true
  timeout: 300  # seconds for code execution
  memory_limit: "1GB"

parsers:
  supported_formats: ["json", "lp", "mps", "python"]

validation:
  max_variables: 100000
  max_constraints: 100000
  max_code_length: 10000  # characters
```

#### solvers.yaml
```yaml
scip:
  class: "SCIPSolver"
  enabled: true
  parameters:
    limits/time: 3600
    display/verblevel: 1

gurobi:
  class: "GurobiSolver"
  enabled: false
  parameters:
    TimeLimit: 3600
    OutputFlag: 1

cplex:
  class: "CPLEXSolver"
  enabled: false
  parameters:
    timelimit: 3600
    output.clonelog: 1
```

### 3.2 設定管理クラス (utils/config_manager.py)

```python
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_dir = Path(config_path) if config_path else Path(__file__).parent.parent / "config"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        default_config = self._load_yaml("default.yaml")
        solver_config = self._load_yaml("solvers.yaml")

        default_config["solvers_config"] = solver_config
        return default_config

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """YAMLファイルを読み込む"""
        config_path = self.config_dir / filename
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得する"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
```

## 4. 入力形式対応

### 4.1 入力形式の拡張

MCPサーバーは以下の入力形式に対応:

1. **Pythonコード実行** (推奨)
   - PuLP, pyomo, pyscipopt等のライブラリを使用
   - 複雑なデータ処理と最適化を組み合わせ可能
   - 柔軟なモデリングが可能

2. **構造化データ形式**
   - JSON: プログラマティックな問題定義
   - LP: 標準線形計画形式
   - MPS: 業界標準形式

### 4.2 パーサーシステム

### 4.3 パーサーファクトリー (parsers/__init__.py)

```python
from typing import Dict, Any, Union
from .json_parser import JSONParser
from .lp_parser import LPParser
from .mps_parser import MPSParser

def get_parser(problem_data: Union[Dict[str, Any], str]):
    """
    問題データの形式に応じて適切なパーサーを返す
    """
    if isinstance(problem_data, dict):
        return JSONParser()
    elif isinstance(problem_data, str):
        if problem_data.strip().startswith("Minimize") or problem_data.strip().startswith("Maximize"):
            return LPParser()
        elif "ROWS" in problem_data and "COLUMNS" in problem_data:
            return MPSParser()
        else:
            # JSON文字列として試行
            try:
                import json
                json.loads(problem_data)
                return JSONParser()
            except:
                raise ValueError("Unsupported problem format")
    else:
        raise ValueError("Invalid problem data type")
```

### 4.4 JSONパーサー (parsers/json_parser.py)

```python
from typing import Dict, Any, Union
from ..models.problem import OptimizationProblem
import json

class JSONParser:
    def parse(self, data: Union[Dict[str, Any], str]) -> OptimizationProblem:
        """
        JSON形式の最適化問題をパースする

        Expected format:
        {
            "name": "problem_name",
            "variables": [
                {"name": "x1", "type": "C", "lower_bound": 0, "upper_bound": 10}
            ],
            "constraints": [
                {"name": "c1", "expression": {"x1": 2, "x2": 3}, "sense": "<=", "rhs": 10}
            ],
            "objective": {
                "sense": "minimize",
                "expression": {"x1": 1, "x2": 2}
            }
        }
        """
        if isinstance(data, str):
            data = json.loads(data)

        return OptimizationProblem.parse_obj(data)
```

## 5. エラーハンドリング戦略

### 5.1 エラー分類

```python
class MIPMCPError(Exception):
    """MIP MCP基底例外クラス"""
    pass

class SolverError(MIPMCPError):
    """ソルバー関連エラー"""
    pass

class ParseError(MIPMCPError):
    """パース関連エラー"""
    pass

class ValidationError(MIPMCPError):
    """バリデーション関連エラー"""
    pass

class ConfigError(MIPMCPError):
    """設定関連エラー"""
    pass
```

### 5.2 エラーハンドリング戦略

1. **入力検証段階**: `ValidationError`でクライアントに詳細な修正指示
2. **パース段階**: `ParseError`で形式エラーを報告
3. **ソルバー実行段階**: `SolverError`で実行エラーを処理
4. **内部エラー**: ログ出力と一般的なエラーメッセージ

## 6. テスト戦略

### 6.1 テスト構造

```
tests/
├── unit/
│   ├── test_solvers/
│   ├── test_parsers/
│   ├── test_handlers/
│   └── test_models/
├── integration/
│   ├── test_mcp_communication/
│   └── test_solver_integration/
└── fixtures/
    ├── sample_problems/
    └── expected_solutions/
```

### 6.2 テストデータ

- 小規模線形計画問題
- 整数計画問題
- 実行不可能問題
- 非有界問題
- エラーケース用の不正データ

## 7. パフォーマンス考慮事項

### 7.1 メモリ管理

- 大規模問題でのストリーミング処理
- ソルバーインスタンスの適切な破棄
- 中間結果のメモリ効率

### 7.2 非同期処理

- 長時間実行される最適化の非同期対応
- プログレス報告機能
- キャンセル機能

## 8. セキュリティ考慮事項

### 8.1 入力検証

- 問題サイズ制限
- ファイル形式検証
- SQLインジェクション対策（外部DB使用時）

### 8.2 リソース制限

- CPU使用時間制限
- メモリ使用量制限
- 同時実行数制限

## 9. 依存関係とインストール

### 9.1 pyproject.tomlの更新

```toml
[project]
name = "mip-mcp"
version = "0.1.0"
description = "MIP optimization server using Model Context Protocol"
readme = "README.md"
authors = [
    { name = "ohtaman", email = "ohtamans@gmail.com" }
]
requires-python = ">=3.12"
dependencies = [
    "fastmcp>=2.10.6",
    "pyscipopt>=5.5.0",
    "pulp>=2.7.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
extra = [
    "pyomo>=6.0.0",
    "cvxpy>=1.3.0",
    "gurobipy>=10.0.0",  # 商用ライセンス必要
    "cplex>=22.1.0",     # 商用ライセンス必要
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "ortools>=9.0.0",
]

[project.scripts]
mip-mcp = "mip_mcp:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 9.2 使用例

#### Pythonコード実行による最適化

```python
# LLMクライアントから送信されるリクエスト例
{
    "tool": "execute_python_code",
    "arguments": {
        "code": """
import pulp

# 生産計画問題
prob = pulp.LpProblem("Production", pulp.LpMaximize)

# 変数定義
x1 = pulp.LpVariable("Product_A", 0, None)
x2 = pulp.LpVariable("Product_B", 0, None)

# 目的関数（利益最大化）
prob += 40*x1 + 30*x2

# 制約条件
prob += 2*x1 + x2 <= 100  # 労働時間制約
prob += x1 + 2*x2 <= 80   # 原材料制約

# 求解
prob.solve()

# 結果をグローバル変数として設定
result = {
    "status": pulp.LpStatus[prob.status],
    "objective": pulp.value(prob.objective),
    "variables": {v.name: v.varValue for v in prob.variables()}
}
""",
        "data": {
            "max_labor_hours": 100,
            "max_materials": 80
        }
    }
}
```

## 10. 将来拡張性

### 10.1 プラグインアーキテクチャ

- 新しいソルバーの動的追加
- カスタムパーサーの追加
- カスタムバリデーターの追加
- 新しい最適化ライブラリのサポート

### 10.2 スケーラビリティ

- 分散処理対応（将来）
- キューイングシステム統合
- 結果キャッシュ機能
- Docker化とKubernetes対応
