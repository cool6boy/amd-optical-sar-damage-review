# 发布整理的源码变化

保留交接的实验实现和锁定结果，仅修正test_core.py中两个固定/tmp测试夹具为TemporaryDirectory，避免Windows路径错误和固定目录清理。新增scripts/verify_results.py用于从已交接结果进行标准库离线核对。未重训、未调参、未修改预测/评价原文件。文档和图表由实际结果整理，源码旧README中的“尚未实验”过时文字已替换。
