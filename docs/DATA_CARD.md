# 数据说明

## 来源与任务
- 数据：BRIGHT官方公开影像及实例损毁标注。
- 入口：https://huggingface.co/datasets/Kullervo/BRIGHT
- 固定revision：46f202d9520cb0844a6ca4679d016d0d27a6d58c。
- 输入：灾前RGB光学、灾后SAR；已有实例bbox条件下分类。
- 官方ID 1/2/3映射为代码ID 0/1/2，即intact/damaged/destroyed。
- 自定义事件隔离课程划分，不是官方holdout或排行榜评测。

## 最终统计
| 集合 | 完整 | 受损 | 毁坏 | 总计 |
|---|---:|---:|---:|---:|
| train | 2184 | 44 | 8 | 2236 |
| val | 827 | 3 | 0 | 830 |
| test（最终评价后统计） | 2188 | 55 | 167 | 2410 |

训练事件：Bata 643、Beirut 832、Turkey 761。
验证事件：Haiti 830。
测试事件：Noto 780、Marshall 1630。

manifest中test标签为null，评价真值从交接的per_building.csv及最终评价读取。精简划分索引见results/split_manifest.csv；完整原manifest在Release结果包中，保留原SHA，避免用精简索引冒充原manifest。

## 配对与预处理
准备器核查图像与标注尺寸、光学/SAR CRS及仿射变换，拒绝明显不一致。bbox上下文1.4倍，最小bbox边4像素，裁剪分位数1%—99%显示拉伸后缩放224×224。SAR单通道复制为3通道；主干前继续应用torchvision该权重的标准transform。该显示处理不是SAR物理标定或专门去斑算法。

QC会记录小目标、非有限值或无信息裁剪等unable_to_judge原因。归档中的QC清单共83条；这是清单条数，不等同于完整数据质量审计，也不自动代表独立建筑总数。

## 计划执行偏差
配置保留train_target_min_per_class=50、train_intact_cap=400、val_target_min_per_class=20、test_max_buildings_per_event=400等目标/参数。当前准备实现的训练分层函数在每片调用，intact cap不是全事件/全训练集上限；测试每事件400栋目标没有得到全局执行。实际测试2410栋、训练稀有类未达目标，必须以实际数据为准。

不对既有测试结果进行事后删样本以符合目标；此版本公开偏差作为局限。严格复现/新版本应先修正并重新锁定抽样实现，再使用新的未评价测试设计，不能把修正后的新实验混入本次结果。

## 不随仓库分发
完整原始影像、全部裁剪、NPZ缓存、第三方主干权重、封存标签目录和云端环境。结果包有最终评价真值的逐建筑表，不包含完整源实例标注档案。数据需按原来源与许可重新获取。

## 许可
影像和标签按事件保留原许可；多数Maxar光学及对应标签CC BY-NC 4.0，SAR CC BY 4.0。源码MIT不覆盖数据。各事件具体来源以官方数据卡为准。