# 模型说明

## 结构
冻结torchvision ResNet18 ImageNet1K V1，去除fc获得512维。光学与SAR分别提取，融合为1024维拼接。每模态使用3个Linear(input_dim,3)小头，seed=11/23/47，softmax概率平均。

仅训练小头，主干权重不包含在heads.pt。源码小头用CPU张量；GPU环境信息证明该环境可用，不能据此声称训练的每一步都在GPU执行。

## 模型文件
- Release资产：heads.pt。
- SHA256：1cee7009e468c2ee2fbb1eb73d341ef02477ee1f5f6404e76c5b328f457523d1。
- 文件保存每个模态的mean、std和members；每个member含seed与线性层state。
- 主干需从torchvision官方来源另行获取，同一输入与实际预处理规范才能得到兼容特征。

## 安全加载与预测示例
以下在已有PyTorch环境、获得正确512/1024维缓存特征后使用；不构成独立完整遥感预处理工具。

```python
import torch
checkpoint = torch.load("heads.pt", map_location="cpu", weights_only=True)

def classify_features(features, mode):
    # features: [N, 512] for optical/SAR; [N, 1024] for fusion
    model = checkpoint[mode]
    x = (features.float() - model["mean"]) / model["std"]
    probabilities = []
    for member in model["members"]:
        head = torch.nn.Linear(x.shape[1], 3)
        head.load_state_dict(member["state"])
        with torch.no_grad():
            probabilities.append(head(x).softmax(dim=1))
    return torch.stack(probabilities).mean(dim=0)
```

不能把未缩放、不同主干或不同特征排列的数组直接输入上述模型。融合排列与pipeline.matrix一致，先光学后SAR。权重缺少原始影像，不提供开箱即用的公众在线推理服务。

## 训练
train=2236，类计数2184/44/8。仅train统计标准化参数及逆类频权重，AdamW，lr=0.01，weight_decay=0.01，每seed上限60epoch，用val交叉熵选检查点。val缺毁坏、仅3受损，选择偏差显著。

## 适用与不适用
用途为课程探索、负结果分析、可复核的排序实验。不得用于结构安全鉴定、救援优先级自动决策或无需人工确认的现实灾情判定。

## 权利
原创模型小模块可作为非商业研究成果随Release分发；不包含第三方主干权重。训练数据、图像和对应标签的原许可及归属保留。代码MIT不等于所有模型/数据素材可商业使用。