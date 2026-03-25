# NOC-Algorithm 优化报告

基于携程订单支付监控算法PRD文档的代码审查报告

生成时间：2026-03-25

## 总体评价

代码实现质量**优秀**，核心算法模块完整，架构清晰，与PRD要求高度一致。以下是发现的优化点：

---

## ✅ 已优化项

### 1. Peak/Valley分段逻辑优化

**问题：** 原实现使用固定的 `±0.25σ` 阈值划分peak/valley，PRD要求使用P70/P30分位数

**优化：**
- 在 `feature_service.py` 添加 `rolling_p70_7d` 和 `rolling_p30_7d` 特征
- 在 `global_detector.py` 使用分位数替代固定阈值
- 在 `utils/stats.py` 添加 `safe_percentile()` 函数

**影响：** 更准确地识别业务高峰/低谷，减少误报

---

## 🔍 需要关注的优化点

### 2. 降采样聚合策略可配置化

**当前实现：** `downsample.py:58-69` 硬编码聚合逻辑

```python
if first.metric_type in {"count", "qps"}:
    aggregated_value = float(np.sum(values))
elif first.metric_type in {"rate", "ratio"}:
    aggregated_value = float(np.mean(values))
```

**PRD要求：**
- count/qps → 窗口求和 ✅
- rate/ratio → 窗口平均 ✅
- latency → 窗口p95或加权平均 ✅
- gauge → 窗口末值或均值 ✅

**建议：** 当前实现已符合PRD，但可考虑将聚合策略配置化到 `MetricMetadata`

**优先级：** P2（可选优化）

---

### 3. 秒级模型特征因子可训练化

**当前实现：** `seasonal_adjustment.py:116-126` 使用硬编码因子

```python
if features.get("isHoliday"):
    factor *= 1.02
if features.get("isHotSpot"):
    factor *= 1.05
```

**PRD要求：**
```
log(correct_rate_t) = b0 + b1*isHoliday + b2*isRCA + b3*isHotSpot + bu_bias[bu]
```

**建议：**
- 当前简化实现可满足MVP需求
- 后续可改为对数线性模型，支持离线训练参数

**优先级：** P2（Phase 2增强）

---

### 4. 模型参数持久化

**当前状态：** 模型参数存储在内存中（`_state` 字典）

**PRD要求：**
- 模型参数存入模型参数存储服务
- 支持模型灰度和回滚

**建议：**
- 扩展 `model_store.py` 支持参数持久化
- 添加版本管理和灰度切换逻辑

**优先级：** P1（Phase 3工程化）

---

### 5. 告警升级时间窗口

**当前实现：** `policy.py:112`

```python
and prediction.timestamp - incident.open_since >= self.settings.alert_upgrade_window_sec
```

**PRD要求：** P1告警超过10分钟未处理升级为P0

**当前配置：** `settings.py:30`
```python
alert_upgrade_window_sec: int = 600  # 10分钟
```

**状态：** ✅ 已正确实现

---

## 📊 PRD符合度检查表

| 模块 | PRD要求 | 实现状态 | 备注 |
|------|---------|----------|------|
| 数据质量校验 | 缺失值≤3点插值 | ✅ | `repair.py` |
| AI适配性判断 | 历史≥4周+评分≥0.65 | ✅ | `eligibility.py` |
| 降采样路由 | P0/P1/P2/P3分级 | ✅ | `router.py` |
| 秒级预测 | Seasonal-Adjustment | ✅ | MAE目标≤0.016 |
| 分钟级预测 | Huber-Regressor | ✅ | MAE目标≤0.022 |
| 全局检测 | Z-score + KDE | ✅ | P2.5/P97.5阈值 |
| 局部检测 | 4种方法组合 | ✅ | `local_detectors.py` |
| 周期过滤 | 历史同期≥60%过滤 | ✅ | `periodic_filter.py` |
| 置信度评分 | 4维度加权 | ✅ | 权重0.25/0.35/0.20/0.20 |
| 告警分级 | P0/P1/P2/P3 | ✅ | `policy.py` |
| 告警收敛 | 5分钟窗口合并 | ✅ | 300秒 |
| 告警升级 | P1→P0 10分钟 | ✅ | 600秒 |
| 恢复通知 | 连续3点正常 | ✅ | `recovery_normal_points=3` |

---

## 🎯 性能指标对比

| 指标 | PRD目标 | 当前实现 | 状态 |
|------|---------|----------|------|
| 秒级检测延迟 | ≤30秒 | 需实测 | - |
| 分钟级检测延迟 | ≤2分钟 | 需实测 | - |
| 故障发现时长 | ≤1分钟 | 需回放验证 | - |
| 召回率 | 100% | 需回放验证 | - |
| 准确率 | ≥80% | 需回放验证 | - |

**建议：** 使用历史故障数据集进行30天回放测试验证

---

## 💡 代码质量亮点

1. **架构清晰**：四层架构（数据/模型/检测/告警）职责分明
2. **状态管理**：使用Plan模式实现状态更新的原子性
3. **历史抽象**：`HistoryStore` 支持内存/SQLite双模式
4. **容错设计**：优雅降级（scipy/sklearn可选依赖）
5. **类型安全**：全面使用类型注解和Pydantic校验

---

## 📝 下一步建议

### 短期（1-2周）
1. ✅ 完成Peak/Valley分段优化（已完成）
2. 使用真实数据进行端到端测试
3. 验证MAE是否达到PRD目标

### 中期（1个月）
1. 接入真实节假日/热点/RCA数据源
2. 实现模型参数持久化
3. 30天历史故障回放验证

### 长期（2-3个月）
1. 模型灰度发布机制
2. 在线服务性能优化
3. 监控看板和效果追踪

---

## 结论

代码实现与PRD要求**高度一致**，核心算法完整，工程质量优秀。主要优化方向集中在：
1. ✅ 分段逻辑优化（已完成）
2. 模型参数持久化（Phase 3）
3. 真实数据验证和调优

**总体评分：9/10** 🌟
