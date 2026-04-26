# craigslistbargain 包导入统一改造方案

## 一、现状分析

### 1.1 发现的问题
项目中混用了三种导入风格：
1. **相对导入**：`from .core.scenario import Scenario`
2. **隐式顶层导入**：`from core.scenario import Scenario`（依赖 PYTHONPATH）
3. **完整包名绝对导入**：`from craigslistbargain.core.scenario import Scenario`

### 1.2 关键发现
- **缺少 `__init__.py`**：`craigslistbargain/__init__.py` 不存在，需要创建
- **运行方式**：所有 shell 脚本使用 `PYTHONPATH=. python multi_rl.py ...`
- **特殊情况**：`scripts/chat_to_scenarios.py` 使用 `sys.path.append('craigslistbargain/')`

## 二、改造目标

将 `craigslistbargain` 包内的所有导入统一为**完整包名绝对导入**：
- ✅ `from craigslistbargain.core.scenario import Scenario`
- ✅ `from craigslistbargain.neural.preprocess import Preprocessor`
- ❌ `from .core.scenario import Scenario`
- ❌ `from core.scenario import Scenario`

**保持现有运行方式**：继续使用 `PYTHONPATH=. python multi_rl.py ...`

## 三、改造范围

### 3.1 需要改造的文件（主代码）

#### 核心模块
- [`craigslistbargain/multi_rl.py`](craigslistbargain/multi_rl.py) - 主入口文件
- [`craigslistbargain/multi_manager.py`](craigslistbargain/multi_manager.py)
- [`craigslistbargain/multi_manager_debug.py`](craigslistbargain/multi_manager_debug.py)
- [`craigslistbargain/multi_trainer.py`](craigslistbargain/multi_trainer.py)
- [`craigslistbargain/main.py`](craigslistbargain/main.py)
- [`craigslistbargain/train_sl.py`](craigslistbargain/train_sl.py)
- [`craigslistbargain/evaluate.py`](craigslistbargain/evaluate.py)
- [`craigslistbargain/reinforce.py`](craigslistbargain/reinforce.py)
- [`craigslistbargain/parse_dialogue.py`](craigslistbargain/parse_dialogue.py)
- [`craigslistbargain/options.py`](craigslistbargain/options.py)

#### core 子包
- [`craigslistbargain/core/scenario.py`](craigslistbargain/core/scenario.py)
- [`craigslistbargain/core/price_tracker.py`](craigslistbargain/core/price_tracker.py)
- [`craigslistbargain/core/event.py`](craigslistbargain/core/event.py)
- [`craigslistbargain/core/kb.py`](craigslistbargain/core/kb.py)
- [`craigslistbargain/core/controller.py`](craigslistbargain/core/controller.py)

#### neural 子包
- [`craigslistbargain/neural/__init__.py`](craigslistbargain/neural/__init__.py)
- [`craigslistbargain/neural/preprocess.py`](craigslistbargain/neural/preprocess.py)
- [`craigslistbargain/neural/utterance.py`](craigslistbargain/neural/utterance.py)
- [`craigslistbargain/neural/batcher_rl.py`](craigslistbargain/neural/batcher_rl.py)
- [`craigslistbargain/neural/batcher.py`](craigslistbargain/neural/batcher.py)
- [`craigslistbargain/neural/vocab_builder.py`](craigslistbargain/neural/vocab_builder.py)
- [`craigslistbargain/neural/symbols.py`](craigslistbargain/neural/symbols.py)
- [`craigslistbargain/neural/generator.py`](craigslistbargain/neural/generator.py)
- [`craigslistbargain/neural/model_builder.py`](craigslistbargain/neural/model_builder.py)
- [`craigslistbargain/neural/rl_model_builder.py`](craigslistbargain/neural/rl_model_builder.py)
- [`craigslistbargain/neural/models.py`](craigslistbargain/neural/models.py)
- [`craigslistbargain/neural/trainer.py`](craigslistbargain/neural/trainer.py)
- [`craigslistbargain/neural/sl_trainer.py`](craigslistbargain/neural/sl_trainer.py)
- [`craigslistbargain/neural/rl_trainer.py`](craigslistbargain/neural/rl_trainer.py)
- [`craigslistbargain/neural/a2c_trainer.py`](craigslistbargain/neural/a2c_trainer.py)
- [`craigslistbargain/neural/evaluator.py`](craigslistbargain/neural/evaluator.py)
- [`craigslistbargain/neural/nlg.py`](craigslistbargain/neural/nlg.py)

#### sessions 子包
- [`craigslistbargain/sessions/session.py`](craigslistbargain/sessions/session.py)
- [`craigslistbargain/sessions/neural_session.py`](craigslistbargain/sessions/neural_session.py)
- [`craigslistbargain/sessions/tom_session.py`](craigslistbargain/sessions/tom_session.py)

#### systems 子包
- [`craigslistbargain/systems/__init__.py`](craigslistbargain/systems/__init__.py)
- [`craigslistbargain/systems/neural_system.py`](craigslistbargain/systems/neural_system.py)

#### model 子包
- [`craigslistbargain/model/dialogue_state.py`](craigslistbargain/model/dialogue_state.py)
- [`craigslistbargain/model/parser.py`](craigslistbargain/model/parser.py)
- [`craigslistbargain/model/generator.py`](craigslistbargain/model/generator.py)
- [`craigslistbargain/model/manager.py`](craigslistbargain/model/manager.py)
- [`craigslistbargain/model/templates.py`](craigslistbargain/model/templates.py)

#### analysis 子包
- [`craigslistbargain/analysis/visualizer.py`](craigslistbargain/analysis/visualizer.py)
- [`craigslistbargain/analysis/html_visualizer.py`](craigslistbargain/analysis/html_visualizer.py)
- [`craigslistbargain/analysis/utils.py`](craigslistbargain/analysis/utils.py)
- [`craigslistbargain/analysis/dialogue.py`](craigslistbargain/analysis/dialogue.py)

#### buffer 子包
- [`craigslistbargain/buffer/__init__.py`](craigslistbargain/buffer/__init__.py)
- [`craigslistbargain/buffer/buffer.py`](craigslistbargain/buffer/buffer.py)

### 3.2 测试文件（需要改造）
- [`craigslistbargain/test_options.py`](craigslistbargain/test_options.py)
- [`craigslistbargain/neural/test_trainer_tom_smoke.py`](craigslistbargain/neural/test_trainer_tom_smoke.py)
- [`craigslistbargain/neural/test_rl_builder.py`](craigslistbargain/neural/test_rl_builder.py)
- [`craigslistbargain/neural/test_rl_builder_forward.py`](craigslistbargain/neural/test_rl_builder_forward.py)

### 3.3 脚本文件（需要特殊处理）
- [`craigslistbargain/scripts/chat_to_scenarios.py`](craigslistbargain/scripts/chat_to_scenarios.py) - 移除 `sys.path.append`

### 3.4 暂不改造的文件
- `craigslistbargain/scripts/old/*` - 历史脚本，可能已废弃
- `craigslistbargain/exp_scripts/**/*.sh` - Shell 脚本，保持 `PYTHONPATH=.`

## 四、改造步骤

### 步骤 1：创建包标识文件
创建 `craigslistbargain/__init__.py`（空文件即可）

### 步骤 2：导入替换规则

#### 规则 A：相对导入 → 绝对导入
```python
# 改前
from .core.scenario import Scenario
from .neural import build_optim
from ..core.price_tracker import PriceScaler

# 改后
from craigslistbargain.core.scenario import Scenario
from craigslistbargain.neural import build_optim
from craigslistbargain.core.price_tracker import PriceScaler
```

#### 规则 B：隐式顶层导入 → 绝对导入
```python
# 改前
from core.scenario import Scenario
from neural.preprocess import Preprocessor
import options

# 改后
from craigslistbargain.core.scenario import Scenario
from craigslistbargain.neural.preprocess import Preprocessor
import craigslistbargain.options
```

#### 规则 C：保持外部包导入不变
```python
# 保持不变
from cocoa.core.util import read_json
from onmt.Utils import use_gpu
import torch
```

### 步骤 3：特殊处理

#### 3.1 `craigslistbargain/scripts/chat_to_scenarios.py`
```python
# 改前
import sys
sys.path.append('craigslistbargain/')
from core.scenario import Scenario

# 改后
from craigslistbargain.core.scenario import Scenario
```

#### 3.2 `craigslistbargain/analysis/visualizer.py`
```python
# 改前
from analyze_strategy import StrategyAnalyzer

# 改后
from craigslistbargain.analysis.analyze_strategy import StrategyAnalyzer
```

### 步骤 4：验证方式

#### 4.1 语法检查
```bash
cd /hy-tmp/projects/NegotiationTOM
python -m py_compile craigslistbargain/multi_rl.py
```

#### 4.2 导入测试
```bash
cd /hy-tmp/projects/NegotiationTOM
PYTHONPATH=. python -c "from craigslistbargain.core.scenario import Scenario; print('OK')"
```

#### 4.3 运行测试
```bash
cd /hy-tmp/projects/NegotiationTOM
# 保持原有运行方式
PYTHONPATH=. python multi_rl.py --help
```

## 五、关于 PYTHONPATH 的说明

### 5.1 为什么保持 `PYTHONPATH=.`？

当前所有 shell 脚本使用：
```bash
PYTHONPATH=. python multi_rl.py ...
```

这种方式的优点：
1. **兼容性好**：同时支持 `craigslistbargain.*` 和 `cocoa.*`、`onmt.*` 的导入
2. **改动最小**：不需要修改 40+ 个 shell 脚本
3. **灵活性高**：可以直接运行 `python multi_rl.py`，也可以用 `python -m craigslistbargain.multi_rl`

### 5.2 PYTHONPATH 的作用

```bash
PYTHONPATH=. python multi_rl.py
```

等价于在 Python 中：
```python
import sys
sys.path.insert(0, '.')  # 将当前目录加入搜索路径
```

这样 Python 可以找到：
- `craigslistbargain/` → `import craigslistbargain`
- `cocoa/` → `import cocoa`
- `onmt/` → `import onmt`

### 5.3 是否需要改为其他方式？

**不需要**。原因：
1. 项目有多个顶层包（`craigslistbargain`、`cocoa`、`onmt`）
2. 改为 `python -m craigslistbargain.multi_rl` 需要修改所有脚本
3. 改为 `cd craigslistbargain && python multi_rl.py` 会破坏相对路径（如 `data/`）

**结论**：保持 `PYTHONPATH=. python multi_rl.py` 是最佳选择。

## 六、实施清单

### 阶段 1：准备工作
- [ ] 创建 `craigslistbargain/__init__.py`

### 阶段 2：核心模块（优先级最高）
- [ ] `craigslistbargain/multi_rl.py`
- [ ] `craigslistbargain/multi_manager_debug.py`
- [ ] `craigslistbargain/multi_trainer.py`
- [ ] `craigslistbargain/options.py`

### 阶段 3：core 子包
- [ ] `craigslistbargain/core/scenario.py`
- [ ] `craigslistbargain/core/price_tracker.py`
- [ ] `craigslistbargain/core/event.py`
- [ ] `craigslistbargain/core/kb.py`
- [ ] `craigslistbargain/core/controller.py`

### 阶段 4：neural 子包
- [ ] `craigslistbargain/neural/__init__.py`
- [ ] `craigslistbargain/neural/preprocess.py`
- [ ] `craigslistbargain/neural/utterance.py`
- [ ] `craigslistbargain/neural/batcher_rl.py`
- [ ] `craigslistbargain/neural/vocab_builder.py`
- [ ] `craigslistbargain/neural/symbols.py`
- [ ] `craigslistbargain/neural/generator.py`
- [ ] `craigslistbargain/neural/model_builder.py`
- [ ] `craigslistbargain/neural/rl_model_builder.py`
- [ ] `craigslistbargain/neural/models.py`
- [ ] `craigslistbargain/neural/sl_trainer.py`
- [ ] `craigslistbargain/neural/rl_trainer.py`
- [ ] `craigslistbargain/neural/a2c_trainer.py`
- [ ] `craigslistbargain/neural/evaluator.py`

### 阶段 5：sessions 子包
- [ ] `craigslistbargain/sessions/session.py`
- [ ] `craigslistbargain/sessions/neural_session.py`
- [ ] `craigslistbargain/sessions/tom_session.py`

### 阶段 6：systems 子包
- [ ] `craigslistbargain/systems/__init__.py`
- [ ] `craigslistbargain/systems/neural_system.py`

### 阶段 7：其他子包
- [ ] `craigslistbargain/model/*` (6 个文件)
- [ ] `craigslistbargain/analysis/*` (4 个文件)
- [ ] `craigslistbargain/buffer/*` (2 个文件)

### 阶段 8：测试和脚本
- [ ] `craigslistbargain/test_*.py` (测试文件)
- [ ] `craigslistbargain/neural/test_*.py` (测试文件)
- [ ] `craigslistbargain/scripts/chat_to_scenarios.py`

### 阶段 9：其他入口文件
- [ ] `craigslistbargain/main.py`
- [ ] `craigslistbargain/train_sl.py`
- [ ] `craigslistbargain/evaluate.py`
- [ ] `craigslistbargain/reinforce.py`
- [ ] `craigslistbargain/parse_dialogue.py`

### 阶段 10：验证
- [ ] 运行语法检查
- [ ] 运行导入测试
- [ ] 运行一个简单的训练脚本验证

## 七、风险与注意事项

### 7.1 潜在风险
1. **循环导入**：改为绝对导入后可能暴露循环依赖
2. **动态导入**：使用 `__import__()` 或 `importlib` 的代码需要特别注意
3. **测试覆盖**：确保改动后所有功能正常

### 7.2 回滚方案
使用 Git 版本控制，每个阶段提交一次，出问题可以回滚。

### 7.3 建议
1. **分阶段实施**：按上述阶段逐步改造，每个阶段验证
2. **保留备份**：重要文件改动前先备份
3. **测试驱动**：每改一批文件就运行测试

## 八、总结

- **改造目标**：统一为 `from craigslistbargain.xxx import yyy` 格式
- **运行方式**：保持 `PYTHONPATH=. python multi_rl.py ...`
- **改造范围**：约 60+ 个 Python 文件
- **预计工作量**：中等（大部分是机械替换）
- **风险等级**：低（改动明确，易于验证）
