# NOCcore

订单支付监控算法第一版实现。

当前版本优先保证以下能力可本地直接运行：

- 秒级 AI 预测链路
- 分钟级降采样与预测链路
- 全局异常检测
- 局部异常检测
- 周期性异常过滤
- 置信度评分与告警决策
- 历史回放入口
- FastAPI 在线接入入口

## 运行方式

```bash
PYTHONPATH=src python3 -B -m noccore.pipeline.replay_pipeline
```

如果要读取外部数据文件：

```bash
PYTHONPATH=src python3 -B -m noccore.pipeline.replay_pipeline --input /path/to/data.jsonl
```

如果要启用外部历史库：

```bash
PYTHONPATH=src python3 -B -m noccore.pipeline.replay_pipeline --history-db /tmp/noccore_history.db
```

输入 JSONL 的每一行示例：

```json
{"metric_name":"payment.success_rate","timestamp":1711180800,"value":0.961,"business_line":"AA","priority":"P0","metric_type":"rate","tags":{"channel":"alipay","force_ai":true}}
```

启动在线服务：

```bash
PYTHONPATH=src uvicorn noccore.api:app --host 0.0.0.0 --port 8000
```

如果在线服务要启用 SQLite 历史存储：

```bash
NOCCORE_HISTORY_DB=/tmp/noccore_history.db PYTHONPATH=src uvicorn noccore.api:app --host 0.0.0.0 --port 8000
```

在线接口说明：

- `POST /v1/ingest`: 接收一批指标点并返回本批触发的告警
- `POST /v1/flush`: 主动冲刷尚未完成的分钟桶，适合回放或批处理收尾
- `POST /v1/reload-registry`: 重载指标注册表
- `GET /health`: 健康检查

## 设计说明

- 运行环境兼容 Python 3.10
- 核心依赖使用 `numpy` 和 `pydantic`
- 如果环境中存在 `scipy` / `scikit-learn` / `fastapi`，会自动启用完整版 KDE、Huber 回归和 HTTP 服务能力
- 历史层已抽象为 `HistoryStore`，当前提供 `InMemoryHistoryStore` 和 `SQLiteHistoryStore`
- 历史层当前可持久化 `raw_points`、`series_points`、`predictions`、`anomalies`
- `Eligibility`、秒级 `Seasonal-Adjustment`、分钟级预测历史、周期过滤和置信度回看已优先从历史层读取，不再直接依赖 pipeline 运行态内存

## 当前实现与 PRD 的对应

- 秒级模型：`src/noccore/model_layer/seasonal_adjustment.py`
- 分钟级模型：`src/noccore/model_layer/minute_huber.py`
- 历史存储抽象：`src/noccore/history_layer/store.py`
- 全局检测：`src/noccore/detect_layer/global_detector.py`
- 局部检测：`src/noccore/detect_layer/local_detectors.py`
- 周期过滤：`src/noccore/detect_layer/periodic_filter.py`
- 置信度评分：`src/noccore/detect_layer/confidence.py`
- 告警决策：`src/noccore/alert_layer/policy.py`
- 回放入口：`src/noccore/pipeline/replay_pipeline.py`

## 文档中的一个实现取舍

PRD 在 Z-score 公式和示例上存在符号歧义：

- 公式写成 `Z = (y' - y) / sqrt(y')`
- 示例中“支付成功率下滑”对应 `Z = -4.2`

为了和异常方向保持一致，当前实现统一采用：

```text
Z = (actual - predict) / sqrt(max(abs(predict), eps))
```

这样：

- `Z < 下限` 表示下跌异常
- `Z > 上限` 表示上涨异常

## 下一步可继续增强

- 接入真实节假日 / 热点 / RCA 外部数据源
- 将模型参数持久化到数据库或配置中心
- 将分钟级检测、置信度和周期过滤也逐步切到统一历史层
- 用真实历史故障集做 30 天回放验收

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests/unit -v
```
