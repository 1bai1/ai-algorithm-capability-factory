# 任务池

每次用户任务建立一个独立目录，目录名建议为 `日期_对象_任务名称`。不得覆盖历史任务。

标准任务目录包含：

```text
任务说明.md
分析方案.md
generated/      生成的算法（algorithm.py + manifest.json 等）
predictions/    任务产出的预测/结果文件
评估/            评估脚本与指标输出
validation/     harness 验证报告（report.json / report.md）
report.md       给人看的任务总结
```

运行完成后，将验证结果和失败经验通过审核回写到 `提炼池/` 对应来源目录，
再将可执行结论更新到 `复用池/` 对应能力目录（均由知识管理员执行）。
