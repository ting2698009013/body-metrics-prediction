# 身体成分软测量实验

[![quality-checks](https://github.com/ting2698009013/body-metrics-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ting2698009013/body-metrics-prediction/actions/workflows/ci.yml)

一个个人机器学习与软件工程项目：研究能否用年龄、性别、身高、体重、腰围等易获取输入，估算 DXA 身体成分指标，并通过本地 Web 界面展示模型输出及不确定性。

这里的“预测”是**横截面回归/软测量**，不是预测未来体重或身体变化。项目处于研究原型阶段，没有临床有效性结论。

公开版保留数据处理、特征工程、模型比较、分组校准、推理服务和界面代码。仓库不包含真实身体记录、下载的数据集、训练好的模型、个人笔记和历史运行结果。提供纯随机合成演示，可在没有原数据、没有模型文件的情况下走通工程流程。

## 项目内容

- Python / pandas：NHANES 多调查周期表连接、变量转换、缺失情况报告。
- scikit-learn：特征组合比较、Ridge / 树模型等回归基线。
- 评估：按调查周期划分训练和留出集；嵌套校准实验显式隔离外层周期，测试数据泄漏边界。
- 推理：三个目标分别建模，模型包带版本、输入契约、文件散列和运行库版本核验。
- FastAPI + 原生 JavaScript：本地界面、输入验证、缺模型提示、过期结果处理与异步请求竞态测试。

输出目标为体脂率、脂肪量和瘦组织量（不含骨矿物质）。瘦组织量不等于肌肉量；三个独立回归输出没有强制满足质量相加关系。

## 安装与一键合成演示

需要 Python 3.11+；Node.js 22+ 只用于前端测试。以下从仓库根目录执行：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/synthetic_demo.py
```

Linux / macOS 使用 `python3 -m venv .venv`，将后续 Python 路径换为 `.venv/bin/python`。

演示生成 900 行完全虚构的数据，按 60% / 20% / 20% 划分训练、校准和测试集；训练三个 Ridge 模型，计算残差区间，与训练均值基线比较，再临时序列化模型、核验散列、测试 API 与静态资源。只打印汇总指标；临时模型在完成后删除，不启动监听端口，也不下载数据。

这些标签由人为设定的简单方程生成，输出数字仅用于检查代码能否工作，**不代表真实人体的预测精度或适用性**。演示并不复现原 NHANES 实验。

```powershell
.\.venv\Scripts\python.exe scripts/synthetic_demo.py --rows 300 --seed 7
```

## 自动验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/frontend.test.cjs
.\.venv\Scripts\python.exe scripts/build_release.py --without-models
```

Python 测试使用合成表和临时模型，覆盖数据拼接、输入契约、模型损坏检测、校准隔离、外部验证数据合同、接口与合成演示。前端测试覆盖结果过期、响应乱序、复制摘要和缺失指标显示。CI 还会验证 wheel 中包含界面资源，以及缺模型时的明确报错。

## 查看本地界面

```powershell
.\.venv\Scripts\python.exe -m body_soft_sensor --model-dir artifacts/model_selection/models
```

打开 `http://127.0.0.1:8000`。新克隆没有模型，页面会说明模型尚未加载，预测接口返回 503；这属于预期行为。显式 `--model-dir` 可避免读取机器上其他位置的模型。请仅加载自己训练或可信来源的模型，`joblib` 反序列化不能用于不可信文件；SHA-256 校验不能证明发布者身份。

## 研究流程复现入口

历史代码研究 NHANES 2011–2018 的 DEMO、BMX、DXX 表，默认将 2011–2016 用于开发、2017–2018 用于留出评估。第三方数据由读者自行从 [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/) 获取并核对变量文档；本仓库不再分发数据。

```powershell
.\.venv\Scripts\python.exe scripts/download_nhanes.py
.\.venv\Scripts\python.exe scripts/prepare_nhanes.py
.\.venv\Scripts\python.exe scripts/run_eda.py
.\.venv\Scripts\python.exe scripts/run_feature_experiments.py
.\.venv\Scripts\python.exe scripts/train_v0_1_models.py
.\.venv\Scripts\python.exe scripts/run_v0_2_calibration_experiments.py
.\.venv\Scripts\python.exe scripts/train_v0_2_models.py
.\.venv\Scripts\python.exe scripts/run_v0_3_nested_calibration.py --preset compact --workers 2
```

上述是完整实验入口，会下载公开数据、执行较长计算，并在本地 `data/`、`artifacts/` 下写入产物；**不属于快速演示，本次公开整理未重新运行完整训练**。数据可得性和第三方链接以来源当前状态为准。旧 v0.2 校准选择结果存在流程隔离局限，研究评估应参考 v0.3 嵌套实验及其边界测试，不能只引用旧评分。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/body_soft_sensor/` | 数据转换、特征构造、推理与 API |
| `src/body_soft_sensor/static/` | 本地交互界面 |
| `scripts/synthetic_demo.py` | 独立离线合成演示 |
| `scripts/train_*.py` | 历史模型训练流程 |
| `scripts/run_v0_3_nested_calibration.py` | 按周期隔离的校准实验 |
| `tests/` | 合成数据与界面回归测试 |
| `docs/methodology.md` | 研究问题、评估设计和公开范围 |

## 限制与后续工作

当前输入范围继承原研究原型的约束，并非健康标准。横截面研究无法据此推断个体未来变化；不同人群、设备和测量方法下的泛化需要独立验证。区间的标称覆盖率不保证对新分布成立。暂无随公开仓库发布的真实测试集、训练模型和可复核的新评分，也不提供医疗或营养建议。

下一步优先改进数据来源契约、独立外部验证、输入测量噪声评估和分组误差分析。代码中的历史实验保留用于学习与复现，公开版已去掉“缺少评估信息时自动套用旧成绩”的行为。

本仓库暂未附加开源许可证；公开源码供展示和交流，第三方依赖及数据仍适用各自条款。
