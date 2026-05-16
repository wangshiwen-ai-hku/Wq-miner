# 🧬 Evolutionary Alpha Miner

Evolutionary Alpha Miner 是一款专为 WorldQuant Brain 平台设计的高性能自主量化因子（Alpha）挖掘引擎。它采用遗传算法（Evolutionary Algorithm）的思想，通过对优质“种子因子”进行交叉（Crossover）、变异（Mutation）和组合，自动化探索高 Sharpe、低相关性的 Alpha 因子空间。

## ✨ 核心特性

- **自主进化循环**：全自动运行模拟、筛选、记录和优化过程。
- **多样性保证**：基于 `family_hash` 的因子去重机制，确保进化方向的多样性，避免陷入局部最优。
- **跨域交叉**：支持 Analyst、Fundamental、Price、Volume、Option 等多维度数据的智能配对与融合。
- **弹性配置**：支持自定义模拟参数、轮次和休眠间隔。

---

## ⚙️ 快速开始

### 1. 环境准备

建议使用 Python 3.10+ 环境。

```bash
pip install -r requirements.txt
```

### 2. 配置项目

项目依赖两个关键的配置文件，请根据模板进行创建：

#### **`.env`**
用于配置 LLM（如 Gemini）的 API Key。
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 GEMINI_API_KEY
```

#### **`credentials.json`**
用于 WorldQuant Brain 平台的 API 认证。
```json
{
    "email": "your_email@example.com",
    "password": "your_password"
}
```

---

## 🚀 运行实例

项目通过 `run_loop.sh` 实现 24/7 不间断运行。为了防止 Mac 在运行期间进入休眠，推荐在终端使用 `caffeinate` 命令：

```bash
# 启动不间断挖掘循环
caffeinate -i bash run_loop.sh
```

- **`run_loop.sh`**: 核心运行脚本，默认每 120 秒开启一轮挖掘。
- **`run_agent.py`**: 单次运行的入口，可通过参数控制轮次和模拟密度。

### 记录已提交 Alpha

挖掘 Agent 在每次非 dry-run 启动时，会先刷新一份当前账号已提交 Alpha 的记录：

```bash
data/submitted_alphas.csv
```

也可以单独运行这个工具：

```bash
python -m alpha_agent.submitted_alpha_tool --credentials credentials.json
```

该 CSV 只保存平台上可见的 Alpha 字段、settings、IS/OS 表现和检查摘要，不保存接口错误信息或调试字段。

### 自然语言检索 Data Fields

可以用自然语言描述研究方向或市场假说，工具会登录 WorldQuant Brain，先解析字段需求，再检索真实可用字段：

```bash
python -m alpha_agent.datafield_tool \
  -s "Find datafields related to:
[analyst, estimate, revision, downgrade, upgrade, EPS, revenue estimate, target price, recommendation, earnings surprise, dispersion]

Exclude datafields related to:
[price, return, volume, vwap, currency, identifier, metadata, reporting currency]

Return:
field_name, dataset, description, coverage, delay, userCount, alphaCount"
```

默认保存为 JSON 到 `data/datafields/`，结构为 `request`、`plan`、`summary`、`fields`。也可以指定输出：

```bash
python -m alpha_agent.datafield_tool \
  -s "Find datafields related to:
[ownership, buyback, dividend, capex, accrual, earnings quality]

Exclude datafields related to:
[price, return, volume, currency, identifier]

Return:
field_name, dataset, description, coverage, delay, userCount, alphaCount" \
  --output data/datafields/capital_quality.json
```

推荐使用 `Find / Exclude / Return` 三段式 prompt，工具会精确解析 include terms、exclude terms 和输出列。非结构化自然语言仍可用：默认 `--intent-mode auto` 会在可用时使用 Gemini 生成检索计划；否则退回到通用短语解析。需要表格时可加 `--format csv`。

---

## 🌱 种子池管理 (Seed Pool)

种子池是进化算法的“灵魂”。你需要在 `alpha_agent/seed_pool.py` 中维护高质量的初始因子。

### 如何更换/更新种子？

当你发现新的高质量 Alpha 或希望改变研究方向时，请直接修改 `alpha_agent/seed_pool.py`：

1.  **`PROVEN_SEEDS`**: 存放你已经验证过的、表现优异的因子。这些因子将作为“精英”参与进化。
    ```python
    PROVEN_SEEDS: List[Tuple[str, str, float, str]] = [
        # (标签, 因子表达式, 历史得分, 备注)
        ("fundamental", "ts_rank(operating_income/cap, 252)", 1.42, "Operating Earnings Yield"),
    ]
    ```

2.  **`CANDIDATE_SEEDS`**: 存放待验证的灵感或第三方研究结果。Agent 会优先对这些种子进行“裸模拟”，通过验证后才会正式加入进化池。

3.  **`TAG_PAIR_TARGETS`**: 调整权重，决定 Agent 在交叉配对时更倾向于哪些领域的组合（例如增加 `analyst` 与 `price` 的交叉权重）。

> [!IMPORTANT]
> 定期更新 `seed_pool.py` 中的因子，可以引导 Agent 探索更具潜力的子空间，避免因过度挖掘导致的收益衰减。
> 很重要： `llm_hybridizer.py` 中`SYSTEM_PROMPT` !!!!!

---

## 🛠 技术架构

- **`alpha_agent/`**: 核心逻辑包。
    - `agent.py`: 控制 LLM 进行 Alpha 设计与进化的逻辑。
    - `seed_pool.py`: 因子库存管理与多样性采样。
    - `brain_client.py`: 与 WorldQuant Brain API 的交互封装。
- **`data/`**: 存放模拟结果、种子池状态等持久化数据。
- **`run_agent.py`**: 进化循环驱动程序。

---

## 📜 许可证

本项目遵循 MIT 许可证。
