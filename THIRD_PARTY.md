# 第三方来源与许可

## BRIGHT
官方数据：https://huggingface.co/datasets/Kullervo/BRIGHT
官方代码：https://github.com/ChenHongruixuan/BRIGHT
固定revision：46f202d9520cb0844a6ca4679d016d0d27a6d58c。

所选事件的影像、实例标注和派生成果保留原来源及许可。官方数据卡说明多数Maxar光学与相应标签采用CC BY-NC 4.0，SAR来自Capella/Umbra，采用CC BY 4.0；不能只使用数据卡顶层license字段概括所有材料。Hawaii和La Palma有额外来源，本实验未选为train/val/test。

本仓库不重新分发原始影像、完整原始标注档案、全部裁剪或NPZ缓存。公开的逐建筑评价/划分是本项目的派生研究记录，仍应保留数据归属及非商业使用边界。

## 主干
torchvision ResNet18 ImageNet1K V1：
https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

交接证据记录该权重SHA256为f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec。第三方预训练权重不纳入仓库或Release；由使用者按官方来源获取。

## 可视化与代码
图表由Matplotlib从交接结果绘制，未加入未观测数据点。封面为原创抽象示意，不是遥感影像。源码MIT仅覆盖本项目原创代码及文档，不对第三方影像、标签、预训练模型重新授权。

## AMD
使用AMD云端ROCm环境开展课程研究。项目独立实现，不代表AMD官方产品或认证。平台运行环境见environment.json，实际账单和停机确认未由交接文件独立核验。