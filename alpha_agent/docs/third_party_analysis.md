# 📦 Third-Party 资源分析 — 可用于种子池

## 总览

| 来源 | 价值 | 可提取内容 |
|------|------|-----------|
| **CrisperX alpha50.csv** | ⭐⭐⭐ | 50 条已验证表达式 + 完整设置 + Sharpe/Fitness |
| **WQ-Brain commands.py** | ⭐⭐⭐ | ~100 条 Alpha101 论文表达式 (已转 FastExpr) |
| **WQ-Brain database.py** | ⭐⭐ | 完整 WQ 算子库 + 数据字段清单 |
| **worldquant-miner operatorRAW.json** | ⭐⭐⭐ | 最完整的算子定义文档 (含描述/用法) |
| **worldquant-miner templateRAW.txt** | ⭐ | 2 条高级模板 (sentiment, analyst) |
| **WorldQuant_alpha101_code** | ⭐ | Alpha101 论文原始 Python 代码 (参考用) |

---

## 1. CrisperX alpha50.csv — 50 条已验证 Alpha

> [!IMPORTANT]
> 这是最高价值来源。50 条全部 Sharpe ≥ 1.25，附带完整设置和真实回测结果。

### 可直接用的高质量种子（从 CSV 提取）

| 表达式 | Sharpe | Fitness | 标签 | 关键数据字段 |
|--------|--------|---------|------|-------------|
| `rank(mdf_pva)` | 1.25 | 1.20 | fundamental | `mdf_pva` (企业估值) |
| `-rank(mdf_ite_q)` | 1.25 | 1.28 | fundamental | `mdf_ite_q` (季度税收) |
| `-rank(mdf_rnd)` | 1.25 | 1.29 | fundamental | `mdf_rnd` (研发) |
| `-rank(mdf_avi)` | 1.25 | 1.59 | fundamental | `mdf_avi` |
| `ts_mean(fnd6_newqv1300_invfgq,10)` | 1.25 | 2.18 | fundamental | `fnd6_*` 字段 |
| `rank(fnd6_dcvt)` | 1.25 | 1.28 | fundamental | `fnd6_dcvt` (债转换) |
| `ts_zscore(mdf_gry, 26)` | 1.25 | 1.18 | fundamental | `mdf_gry` (增长收益率) |
| `ts_mean(mdf_bet,10)` | 1.25 | 1.49 | price | `mdf_bet` (beta) |
| `rank(current_ratio)` | 1.25 | 1.27 | fundamental | `current_ratio` |
| `-rank(mdf_opi)` | 1.27 | 1.43 | fundamental | `mdf_opi` (经营利润率) |
| `group_rank(fam_est_eps_rank, sector)` | 1.26 | 1.22 | analyst | `fam_est_eps_rank` |
| `-ts_av_diff(mdf_eg3, 250)*ts_corr(mdf_eg3, mdf_sg3, 250)` | 1.26 | 1.77 | fundamental | `mdf_eg3`, `mdf_sg3` |

### 发现的新数据字段前缀

| 前缀 | 含义 | 示例字段 |
|------|------|---------|
| `mdf_*` | Market-derived fundamentals | `mdf_pva`, `mdf_rnd`, `mdf_avi`, `mdf_gry`, `mdf_opi`, `mdf_bet` |
| `fnd6_*` | Fundamental data (第6版) | `fnd6_dcvt`, `fnd6_optvol`, `fnd6_newa1v1300_epspi` |
| `fnd6_newqv1300_*` | Quarterly fundamentals | `fnd6_newqv1300_invfgq`, `fnd6_newqv1300_recdq` |
| `fnd6_cptmfmq_*` | Quarterly cash flow metrics | `fnd6_cptmfmq_oibdpq` |
| `fam_*` | Family/analyst fields | `fam_est_eps_rank` |
| `anl4_*` | Analyst data v4 | `anl4_afv4_eps_high` |

> [!TIP]
> 这些 `mdf_*` 和 `fnd6_*` 字段是你当前种子池中**完全没有**的新数据维度。加入后可以大幅提升跨域杂交的多样性。

---

## 2. WQ-Brain commands.py — Alpha101 论文表达式

`from_arxiv()` 函数包含 **~100 条已转 FastExpr 的 Alpha101 表达式**，全部是价量类：

### 精选可用种子（已按信号类型分类）

**反转类 (Reversal):**
- `(sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1)))` — 量价反转
- `(-1 * ts_delta((((close - low) - (high - close)) / (close - low)), 9))` — Williams %R 变化

**动量类 (Momentum):**
- `((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))` — 多维动量

**Volume-Price Correlation:**
- `(-1 * ts_corr(rank(open), rank(volume), 10))` — 开盘/成交量相关性
- `(-1 * rank(ts_covariance(rank(close), rank(volume), 5)))` — 价量协方差

**VWAP 类:**
- `(rank((vwap - close)) / rank((vwap + close)))` — VWAP 偏离
- `(((high * low)^0.5) - vwap)` — 几何均价 vs VWAP

**Group Neutralized:**
- `(-1 * Ts_Rank(ts_decay_linear(ts_corr(group_neutralize(vwap, sector), volume, 4), 8), 6))`

---

## 3. worldquant-miner operatorRAW.json — 算子文档

Gen2 版本有 **1185 行**完整算子定义，比我们 LLM prompt 中的算子列表更全。

### 我们尚未用到的高价值算子

| 算子 | 用途 | 适合场景 |
|------|------|---------|
| `hump(x, hump=0.01)` | 限制变化幅度降低换手 | 替代 `ts_decay_linear` 降 turnover |
| `ts_target_tvr_decay(x, target_tvr=0.1)` | 自动调 decay 至目标换手 | 直接控制 turnover |
| `jump_decay(x, d, sensitivity, force)` | 自动平滑跳变 | 基本面字段财报日噪声 |
| `days_from_last_change(x)` | 距上次变化天数 | 检测基本面更新周期 |
| `ts_scale(x, d)` | 时序归一化 | 替代 `ts_zscore` |
| `vector_neut(x, y)` | 向量正交化 | 去除特定因子暴露 |
| `bucket(rank(x), range=...)` | 分桶分组 | 自定义分组做 group_rank |
| `last_diff_value(x, d)` | 上次不同值 | 基本面变化检测 |

---

## 4. templateRAW.txt — 高级模板

```
sentiment = ts_backfill(ts_delay(vec_avg(SENTIMENT FROM OTHER),1),20);
vhat = ts_regression(volume, sentiment, 250);
ehat = -ts_regression(returns, vhat, 750);
alpha = group_rank(ehat, bucket(rank(cap), range='0,0.1,0.1'))
```

> [!NOTE]
> 这是一个多行表达式示例，使用了 `ts_regression` 做残差提取 + `bucket` 做自定义分组。这种残差提取的思路可以迁移到你的基本面信号上。

---

## 5. 建议：整合进种子池的内容

### 立即可用（直接加入 seed_pool.py）

```python
# From CrisperX alpha50 — 高质量 mdf/fnd6 字段种子
("fundamental_mdf", "rank(mdf_pva)", 1.20, "Enterprise value metric"),
("fundamental_mdf", "-rank(mdf_rnd)", 1.29, "R&D signal"),
("fundamental_mdf", "-rank(mdf_opi)", 1.43, "Operating profitability"),
("fundamental_mdf", "ts_zscore(mdf_gry, 26)", 1.18, "Growth yield zscore"),
("analyst", "group_rank(fam_est_eps_rank, sector)", 1.22, "Analyst EPS rank in sector"),
("fundamental_fnd6", "rank(fnd6_dcvt)", 1.28, "Debt conversion"),
("fundamental_mdf", "-ts_av_diff(mdf_eg3, 250)*ts_corr(mdf_eg3, mdf_sg3, 250)", 1.77, "Earnings-sales correlation"),
("fundamental", "rank(current_ratio)", 1.27, "Current ratio ranking"),

# From WQ-Brain/commands.py — 价量类种子
("price_volume", "(-1 * ts_corr(rank(open), rank(volume), 10))", 0.70, "Alpha101 open-vol corr"),
("price", "(rank((vwap - close)) / rank((vwap + close)))", 0.65, "VWAP deviation ratio"),
("volume", "((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))", 0.75, "Multi-dim momentum"),
```

### 算子升级（更新 LLM system prompt）

在 `SYSTEM_PROMPT` 中补充：`hump`, `ts_target_tvr_decay`, `jump_decay`, `days_from_last_change`, `bucket`, `vector_neut`
