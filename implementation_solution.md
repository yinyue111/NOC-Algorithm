# 订单支付监控算法实现方案

## 1. 需求理解

基于 PRD，这不是单一预测模型，而是一套完整的监控算法链路，目标是在订单支付场景下实现：

- 秒级 + 分钟级双时效异常检测
- AI 建模与规则兜底并存
- 全局异常 + 局部异常 + 周期性过滤的多阶段判定
- 告警分级、收敛、升级、恢复的闭环输出

核心验收目标：

- 故障发现时长 `<= 1 分钟`
- 秒级检测端到端延迟 `<= 30 秒`
- 核心链路召回率 `= 100%`
- 告警准确率 `>= 80%`
- 新指标接入无需人工阈值配置

## 2. 当前落地判断

`/Users/drum/Desktop/code/NOCcore` 目前为空目录，因此建议按“从零搭建算法内核”的方式实现，优先建设以下 4 个核心能力：

1. 标准化数据流和质量校验
2. 秒级/分钟级预测模型
3. 多阶段异常检测与置信度评分
4. 告警决策与回放验证框架

建议技术栈：

- Python 3.11
- `pydantic` 负责数据契约
- `numpy` / `pandas` 负责时序处理
- `scikit-learn` 负责 Huber 回归与基础统计模型
- `scipy` 负责 KDE、统计检验
- `fastapi` 作为在线推理接口
- `pytest` 负责单测和回放验证

## 3. 总体架构

建议按 PRD 的四层架构拆成可独立演进的模块：

### 3.1 数据层

职责：

- 接收秒级指标点
- 数据质量校验和修复
- AI 适配性判断
- 降采样与路由

输出：

- 标准化秒级数据流
- 标准化分钟级数据流
- 指标 AI 适配配置

### 3.2 模型层

职责：

- 特征计算
- 秒级 Seasonal-Adjustment 预测
- 分钟级 Huber-Regressor 预测
- 模型参数版本管理

输出：

- `predict_value`
- `prediction_band`
- `model_version`

### 3.3 检测层

职责：

- 全局异常判断
- 局部异常套件
- 周期性异常过滤
- 置信度评分

输出：

- `is_abnormal`
- `abnormal_type`
- `confidence_score`

### 3.4 告警层

职责：

- 告警分级
- 告警抑制与收敛
- 升级与恢复通知
- 规则兜底输出

输出：

- `alert_event`

## 4. 推荐工程目录

建议先按下面的目录组织，后续代码可以直接对应实现：

```text
NOCcore/
  implementation_solution.md
  README.md
  pyproject.toml
  src/
    noccore/
      config/
        settings.py
        metric_registry.py
      schemas/
        metric.py
        prediction.py
        anomaly.py
        alert.py
      data_layer/
        validator.py
        repair.py
        router.py
        downsample.py
        eligibility.py
      feature_layer/
        feature_service.py
        calendar_provider.py
      model_layer/
        seasonal_adjustment.py
        minute_huber.py
        model_store.py
        trainer.py
      detect_layer/
        global_detector.py
        local_detectors.py
        periodic_filter.py
        confidence.py
      alert_layer/
        policy.py
        suppressor.py
        notifier.py
      pipeline/
        online_pipeline.py
        replay_pipeline.py
      utils/
        stats.py
        time.py
  tests/
    unit/
    integration/
    replay/
  data/
    samples/
    replay_cases/
```

## 5. 数据对象设计

建议先统一 4 个核心对象：

### 5.1 标准输入点 `MetricPoint`

```python
{
  "metric_name": "payment.success_rate",
  "timestamp": 1711180800,
  "value": 0.961,
  "business_line": "AA",
  "priority": "P0",
  "tags": {
    "channel": "alipay",
    "metric_type": "rate"
  }
}
```

### 5.2 预测结果 `PredictionResult`

```python
{
  "metric_name": "...",
  "timestamp": 1711180800,
  "granularity": "1s",
  "actual_value": 0.823,
  "predict_value": 0.961,
  "lower_bound": 0.934,
  "upper_bound": 0.987,
  "model_name": "seasonal_adjustment",
  "model_version": "sec_v1"
}
```

### 5.3 异常事件 `AnomalyEvent`

```python
{
  "metric_name": "...",
  "timestamp": 1711180800,
  "global_abnormal": 1,
  "local_score": 0.75,
  "is_periodic": 0,
  "z_score": -4.2,
  "confidence_score": 0.78,
  "abnormal_labels": ["global", "boxplot", "trend"]
}
```

### 5.4 告警对象 `AlertEvent`

```python
{
  "metric_name": "...",
  "alert_level": "P0",
  "alert_type": "global+local",
  "confidence_score": 0.78,
  "duration_sec": 200,
  "message": "AA业务线支付成功率低于预测值",
  "status": "firing"
}
```

## 6. 算法实现方案

## 6.1 数据层

### 6.1.1 质量校验

严格按 PRD 落地：

- 缺失值：连续缺失 `<= 3` 个点，用线性插值补齐
- 超量跳变：单点变化超过历史 `P99 * 10` 时截断
- 时间乱序：允许 `+-5 秒` 抖动，进入小缓冲重排
- 重复点：`metric_name + timestamp` 去重取最新

建议实现为 `validator -> repair -> router` 三段式流水。

### 6.1.2 AI 适配性判断

PRD 中“历史数据 >= 4 周”属于硬门槛，其余做加权评分：

```text
eligible = history_weeks >= 4
score = 0.30 * periodicity_score
      + 0.25 * non_zero_score
      + 0.20 * business_priority_score
      + 0.25 * regularity_score
AI适配条件 = eligible and score >= 0.65
```

具体映射建议：

- `periodicity_score`：对日周期和周周期做 ACF 显著性检验，取最大值
- `non_zero_score`：`min(non_zero_ratio / 0.7, 1.0)`
- `business_priority_score`：`P0=1.0, P1=0.8, P2=0.5, P3=0.3`
- `regularity_score`：`max(0, 1 - cv / 0.8)`

输出写入 `metric_registry`，每周重算一次，支持人工覆盖。

### 6.1.3 降采样与路由

建议按照指标类型定义聚合策略，而不是统一平均值：

- `count/qps`：窗口求和
- `rate/ratio`：窗口平均
- `latency`：窗口 `p95` 或加权平均
- `gauge`：窗口末值或均值

然后按 PRD 规则路由：

- 高频 P0 非 AI -> `1m`
- P1 非 AI -> `5m`
- P2 非 AI -> `10m`
- P3 非 AI -> `30m`

## 6.2 特征层

统一生成以下在线特征：

- `isHoliday`
- `isRCA`
- `isHotSpot`
- `business_unit`
- `baseDayDiff`
- `hour_of_day`
- `day_of_week`
- `rolling_mean_7d`
- `rolling_std_7d`

建议把节假日、热点、RCA 都包装成 provider，算法核心只依赖统一接口，避免后面接外部系统时侵入主逻辑。

## 6.3 秒级模型：Seasonal-Adjustment

PRD 给的是算法思路，落地时建议明确为“对齐基线 + 乘性修正”的两阶段模型。

### 6.3.1 历史对齐日选择

从最近 4 周中选择满足以下条件的历史日：

- 同星期
- 同节假日属性
- 同业务线
- 不在已知异常窗口内

如果候选样本少于 2 天，则降级为“同小时历史中位数基线”。

### 6.3.2 基线构建

对齐时刻的历史值做稳健聚合：

```text
base_t = median(aligned_history_t)
baseDayDiff_t = EWMA(actual_{t-1} - base_{t-1}, alpha=0.3)
preprocessed_t = max(eps, base_t + baseDayDiff_t)
```

这里用 `EWMA` 而不是瞬时差值，是为了降低单点抖动对秒级预测的污染。

### 6.3.3 修正率建模

PRD 要求 `predict = preprocessHisData * f(correctRateList)`，建议定义为：

```text
log(correct_rate_t) =
    b0
  + b1 * isHoliday
  + b2 * isRCA
  + b3 * isHotSpot
  + bu_bias[bu]
  + hour_bias[hour]

predict_t = preprocessed_t * exp(log(correct_rate_t))
```

这样可以保证修正率恒为正数，也便于在线增量更新。

### 6.3.4 更新策略

- 在线：每秒根据残差做小步长更新，类似 `EWMA + SGD`
- 离线：每天 `02:00 ~ 03:00` 全量重训参数
- 样本过滤：历史异常点不进入重训数据

### 6.3.5 秒级模型输出

建议输出：

- `predict_value`
- `residual_std`
- `lower_bound = predict - 2 * residual_std`
- `upper_bound = predict + 2 * residual_std`

## 6.4 分钟级模型：Huber-Regressor

适用于非 AI 秒级指标下采样后的分钟级连续性检测。

### 6.4.1 训练窗口

按 PRD 使用：

- `N = 60`
- `epsilon = 1.35`
- `max_iter = 200`
- `alpha = 0.0001`

### 6.4.2 去异常偏移

PRD 中 `gap_i` 可以具体化为：

```text
gap_i = actual_i - predict_i(if previous point is abnormal)
cleaned_i = actual_i - gap_i
```

如果上一点异常，则优先使用预测值替代异常残差，避免异常污染滑窗回归。

### 6.4.3 拟合方式

```text
X = [1, 2, ..., N]
Y = 最近 N 个清洗后的分钟级值
model = HuberRegressor(X, Y)
predict_{N+1} = model(N + 1)
```

对于样本不足 `N` 的新指标，先使用：

- 历史均值基线
- N-Sigma 规则
- 跳零规则

三者兜底，等窗口足够后再切到 Huber。

## 6.5 全局异常检测

### 6.5.1 统计量

按 PRD：

```text
Z = (predict - actual) / sqrt(max(predict, eps))
```

需要分别维护 peak 和 valley 两套分布。

### 6.5.2 peak/valley 划分建议

建议基于 `rolling_mean_7d` 做分位数切分：

- 大于历史同时段 `P70` 记为 peak
- 小于历史同时段 `P30` 记为 valley
- 中间区间沿用最近类别

这样可以避免仅靠固定小时做划分，适配大促和节假日波动。

### 6.5.3 KDE 阈值

分别对 peak/valley 收集最近 30 天 Z 值：

- KDE 拟合分布
- 取 `P2.5` 和 `P97.5`
- 每周刷新
- 带宽支持手工覆盖

触发条件：

- 秒级连续 `3` 点超阈值
- 分钟级连续 `2` 点超阈值

## 6.6 局部异常检测

四种检测方法建议全部封装为统一接口 `detect(window) -> score, label`。

### 6.6.1 鲁棒回归残差

- 使用近 N 个点做 Huber 拟合
- 当前残差超过历史残差 `3 sigma` 判异常

### 6.6.2 Boxplot

- 对最近 `60` 点计算 `Q1 / Q3 / IQR`
- 超出 `Q1 - 1.5 * IQR` 或 `Q3 + 1.5 * IQR` 判异常

### 6.6.3 N-Sigma

- 历史同期窗口估计 `mu / sigma`
- 超出 `mu +- 3 * sigma` 判异常

### 6.6.4 趋势检验

- 对 `diff` 序列做 `Mann-Kendall`
- `p < 0.05` 视为显著趋势异常

### 6.6.5 组合得分

按 PRD：

```text
local_score = hits / 4
```

同时保留命中的方法列表，便于告警解释。

## 6.7 周期性异常过滤

建议流程：

1. 取过去 4 周同一时段 `+-10 分钟` 的历史窗口
2. 回放同样检测逻辑，统计“历史也异常”的比例
3. 若比例 `>= 60%`，直接标记为周期性异常
4. 否则对当前与历史同期做标准化差分
5. 对差分序列做趋势检验，`p < 0.05` 才保留为真实异常

这里建议把“周期性异常”定义为可解释标签，而不是直接丢弃原始检测结果，方便后续复盘。

## 6.8 置信度评分

直接按 PRD 公式实现：

```text
confidence =
    0.25 * metric_reliability
  + 0.35 * anomaly_severity
  + 0.20 * volatility_level
  + 0.20 * is_periodic
```

建议各字段具体化为：

- `metric_reliability`：历史 14 天告警命中率或回放准确率
- `anomaly_severity`：当前 Z-score 的归一化绝对值，截断到 `[0, 1]`
- `volatility_level`：`1 - normalized_cv`
- `is_periodic`：周期性异常为 `0`，非周期性为 `1`

告警门槛：

- `>= 0.6` -> P0 候选
- `>= 0.4` -> P1/P2 候选

## 6.9 告警决策

建议分两步：

### 6.9.1 级别判定

- `confidence >= 0.6` 且核心支付链路 -> `P0`
- `confidence >= 0.4` 且核心支付链路 -> `P1`
- `confidence >= 0.6` 且次核心 -> `P1`
- `confidence >= 0.4` 且次核心 -> `P2`
- 跳零规则命中 -> `P2`
- 边界/不确定 -> `P3`

### 6.9.2 抑制规则

- 同指标同类型 `5 分钟` 内收敛合并
- `P1` 超过 `10 分钟` 未处理升级为 `P0`
- 连续 `3` 个点恢复正常则恢复通知
- 维护窗口内只记录不触发通知

## 7. 在线处理主流程

建议在线流水线如下：

```text
metric point
  -> validate
  -> repair
  -> AI eligibility / registry lookup
  -> route(sec/min)
  -> feature build
  -> predict
  -> global detect
  -> local detect
  -> periodic filter
  -> confidence score
  -> alert policy
  -> notify / persist
```

对应伪代码：

```python
def process(point):
    point = validator.validate(point)
    point = repair.fix(point)
    route = router.route(point)
    features = feature_service.build(point)
    prediction = predictor.predict(point, features, route)
    anomaly = detector.detect(point, prediction, features)
    alert = alert_policy.decide(anomaly, point)
    return alert
```

## 8. 版本推进建议

为了更快落地，建议按 3 个阶段推进：

### Phase 1：MVP 算法内核

范围：

- 数据校验
- AI 适配判断
- 秒级 Seasonal-Adjustment
- 分钟级 Huber-Regressor
- 全局异常检测
- 基础告警决策

目标：

- 先跑通离线回放
- 可对历史故障集复盘
- 产出 MAE / Recall / Precision

### Phase 2：检测增强

范围：

- 四种局部检测器
- 周期性过滤
- 告警收敛、升级、恢复

目标：

- 降误报
- 提升解释性

### Phase 3：工程化

范围：

- FastAPI 在线服务
- 模型版本管理
- 定时重训
- 外部告警平台接入

目标：

- 达到 PRD 的在线 SLA 和闭环通知要求

## 9. 测试与验收设计

建议测试分三层：

### 9.1 单元测试

- 数据修复逻辑
- 特征计算
- 模型预测稳定性
- 各检测器判定逻辑
- 置信度计算

### 9.2 集成测试

- 从数据输入到告警输出的全链路
- 连续点触发和恢复逻辑
- 收敛窗口与升级规则

### 9.3 历史回放测试

- 用历史故障数据集回放 30 天
- 输出 MAE、Recall、Precision、平均发现时延
- 和跳零规则做 AB 对比

## 10. 当前最关键的实现决策

结合 PRD 和当前空仓状态，我建议先做下面几个确定项：

1. 用 Python 实现算法主干，优先保证可回放、可验证
2. 先做“离线回放 + 本地告警事件输出”，再接在线服务
3. 秒级模型采用“对齐历史中位数 + EWMA 差值 + 乘性修正率”
4. 分钟级模型严格按 Huber-Regressor 落地，并对异常点做污染隔离
5. 检测层先实现全局检测，再叠加局部检测和周期性过滤

## 11. 下一步建议

如果继续往下做，建议按下面顺序开工：

1. 初始化项目骨架与 `pyproject.toml`
2. 先完成 `schemas + data_layer + replay_pipeline`
3. 实现 `seasonal_adjustment.py` 与 `minute_huber.py`
4. 接上 `global_detector.py` 与 `confidence.py`
5. 用样例数据跑一版端到端回放

如果需要，我下一步可以直接在这个目录里继续把项目骨架和第一版算法代码搭出来。
