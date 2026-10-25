# 交付文件导航

## GitHub项目主页
README包含研究问题、原始规模、事件划分、方法流程、实测图表、复核与漏判风险、运行环境、复现与局限。

## 仓库中的结果
- results/run/final_evaluation.json：原始最终三模态与复核评价。
- results/run/per_building.csv：2410栋测试逐建筑真值/预测/不确定性/分歧。
- results/run/ranking_*.json：当前测试集的三种完整排名。
- results/run/policy.lock.json、prediction_receipt.json、evaluation_receipt.json：策略与文件关联。
- results/experiment.lock.json、features_complete.json、preparation_complete.json、selected_tiles.json、data_qc.json、environment.json。
- results/split_manifest.csv：5476栋精简划分索引；原manifest在结果包内。
- results/classification_comparison.csv、review_comparison.csv：从原报告导出的对照表。
- results/publication_check.json：发布整理验证范围。
- assets/*.svg：从真实结果生成的五组图，另有原创项目封面。
- docs/FAILURE_CASES.md：已有数字失败案例，未附原始图片。

## 版本发布
[打开v0.2.0-course Release](https://github.com/cool6boy/amd-optical-sar-damage-review/releases/tag/v0.2.0-course)

资产：
- heads.pt：自训轻量分类头，第三方ResNet权重不包含其中。
- experiment-results-v2.zip：完整原结果JSON/CSV、manifest、概率、训练日志、小头、配置和发布说明；不含全部裁剪/NPZ/原始影像。
- SHA256SUMS.txt：上述二进制资产的校验值。

结果包内results保持原始文件字节和哈希，新增说明与图表另列docs/assets。排除session、runtime私有记录、凭据、旧补丁、旧交接摘要及原始数据缓存。包内MANIFEST_SHA256.json记录各文件哈希，不对自身作递归哈希。

## 使用边界
模型、数据和衍生成果的归属见THIRD_PARTY及MODEL_CARD。用于课程与非商业研究的展示，不代表灾后安全鉴定或已验证的救援系统。真人复核尚未开展，不把模拟收益称为实测人工效果。