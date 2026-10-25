# 复现与运行

## 三种不同的复核范围
1. **结果复核**：下载仓库，使用标准库检查公开表、评价和排序，不需要GPU。
2. **模型加载**：下载Release中的heads.pt，在已有PyTorch环境读取；必须配合正确特征。
3. **完整云端重跑**：重新获取许可允许的数据、生成裁剪、在AMD ROCm提取特征并拟合小头。发布整理阶段没有重跑此完整路径；随机性/库版本及准备实现差异可能影响结果。

## 结果复核（最短路径）
```bash
git clone https://github.com/cool6boy/amd-optical-sar-damage-review.git
cd amd-optical-sar-damage-review
python scripts/verify_results.py
python -m unittest discover -s tests -v
```

完整原manifest、预测概率、训练日志及权重见Release，下载并解压结果包后可让verify_results.py额外核对这些文件的SHA。结果复核与轻量测试均不训练模型。

## 环境
云端已有Python 3.12、ROCm PyTorch 2.9.1及对应TorchVision。保持平台原torch/torchvision；requirements-cloud.txt只列数据依赖，不含CUDA或大型模型安装。GPU doctor实际记录见results/environment.json。

```bash
python -m pip install -r requirements-cloud.txt
export ADR_ROOT=/home/jovyan/adr-reproduce-v2
export TORCH_HOME="$ADR_ROOT/cache/torch"
export XDG_CACHE_HOME="$ADR_ROOT/cache"
```

所有后续命令显式带configs/formal-v2.lock.json。scripts/terminal.sh为原课程目录编写，默认指向旧adr-work；外部复现者不要不加核对就source它。

## 数据准备（CPU）
```bash
python -m adr.prepare --root "$ADR_ROOT" --config configs/formal-v2.lock.json --cloud-confirm
python -m adr.pipeline weights --root "$ADR_ROOT" --config configs/formal-v2.lock.json --cloud-confirm
python -m adr.continue_run validate --root "$ADR_ROOT" --config configs/formal-v2.lock.json
```

使用原锁定revision，不用preflight刷新版本。HTTP Range不被遵守时程序会拒绝下载完整大包。配置允许val缺类只是探索性警告，不得伪造类别。旧目标字段未完全执行的偏差见DATA_CARD.md；需先核对新建manifest是否与历史SHA一致，不能仅凭同名配置宣布完全复现。

## GPU特征
先在AUP选择GPU并确认自动停止、持久目录和真实时长。下列变量必须来自当前平台，不能照填历史时长。

```bash
read -rp "Actual full session minutes: " SESSION_MINUTES
read -rp "Actual remaining minutes: " REMAINING_MINUTES
python -m adr.continue_run start --kind gpu \
  --session-minutes "$SESSION_MINUTES" --remaining-minutes "$REMAINING_MINUTES" \
  --persistent-confirmed --auto-stop-confirmed \
  --root "$ADR_ROOT" --config configs/formal-v2.lock.json
python -m adr.continue_run pilot --root "$ADR_ROOT" --config configs/formal-v2.lock.json
python -m adr.continue_run features --root "$ADR_ROOT" --config configs/formal-v2.lock.json
```

features单次有时间上限，可在同一有效会话内恢复，直到实际pending=0并生成一致的完成标记。不能靠复制或手写完成标记跳过计算。若中断，保留缓存，真实新会话按实际信息重新登记。停止计算不等于Stop My Server。

## 小头、预测和评价（CPU运算）
交接源码continue_run仍把fit/predict/evaluate路由到CPU receipt；该限制与所处主机是否有GPU不同。可以在有CPU资源的云端直接使用以下pipeline入口，明确记录实际设备；完整GPU会话控制仅用于特征阶段。不要伪造CPU会话来绕过原路由检查。

```bash
python -m adr.pipeline fit --root "$ADR_ROOT" --config configs/formal-v2.lock.json --cloud-confirm
python -m adr.pipeline predict --root "$ADR_ROOT" --config configs/formal-v2.lock.json --cloud-confirm
python -m adr.pipeline evaluate --root "$ADR_ROOT" --config configs/formal-v2.lock.json --cloud-confirm
python -m adr.deliver --root "$ADR_ROOT" --config configs/formal-v2.lock.json
```

predict不读取测试标签，evaluate读取封存标签。测试后不据此调参。已有delivery目录时程序拒绝覆盖，先保留版本并选择明确的新输出路径，不删除历史结果来冒充重跑。

## 复核查看器
```bash
python -m adr.viewer --root "$ADR_ROOT" --port 8765
```

在AUP代理/Ports中打开，非公网服务。需存在原裁剪和当前预测；源码仓库或Release不包含全部裁剪，因此无法仅靠下载结果包显示所有图片。判断支持完整/受损/毁坏/无法判断，保存真人记录与理想复核分开。

## 安全与存储
本地不安装训练环境或下载完整原始数据；图像与缓存保留云端。每次结束实际Stop My Server并确认Stopped。运行receipt不是账单，未拿到平台Usage时不能声称真实费用或总GPU使用时长。