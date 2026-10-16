# 2026-10-16 源码维护记录

本次解决代码审查确认的六项运行问题，没有更换项目业务或增加新的模型框架。

| 问题 | 修复后的行为 | 回归证据 |
| --- | --- | --- |
| Replay 返回 HTTP 500，报告仍无故障 | 最新 Replay 的状态码、响应体、超时信号进入最终确定性诊断；无有效响应时保持不确定 | 500、429、timeout、200 错误响应 Schema、正常响应、blocked/network/oversize 用例 |
| 重用 thread_id 带出旧报告和旧假设 | 新诊断重置报告、假设、查询、工具和审批状态；进入 Checkpoint 前脱敏 | 同线程旧版本故障→新请求待审批→拒绝→再次诊断 |
| 模型标签正确便能绕过 Guardrail | 默认保护所有事实字段，包含根因、证据、修复建议和复验步骤 | 同标签但伪造正文的报告被纠正 |
| 全局 Replay 计数耗尽后所有任务失败 | 每次任务使用独立受限 transport；图内工具历史限制重复 Replay | 连续三个独立任务成功，各自计数 1；重复审批令牌拒绝 |
| Provider 静默忽略 replay_requested | 仅 mode=agent 支持审批 Replay，其他模式携带该标记返回 422 | HTTP 边界测试，确认不会调用模型 |
| Provider 综合超时变成 HTTP 400 | 使用确定性报告降级，返回 200 并注明来源；不泄漏异常中的凭据 | 模拟超时与无效 JSON；正常 Provider 路径也有回归 |

`scripts/run_real_eval.py` 新增 `deterministic_fallback_count/rate` 与逐例降级标记。Agent 准确率标为包含兜底的系统准确率；不能当作独立模型能力。历史真实模型结果未重跑、未覆盖。

检查结果：47 项测试通过，Ruff、mypy（17 个文件）、依赖一致性通过。综合覆盖率 82.25%，独立分支覆盖率 70.90%；修正 README 中把综合覆盖率误写成分支覆盖率的描述。并在 CI 的并行 coverage 数据汇总后再生成报告。

原始输出：[checks.json](../eval-results/maintenance-2026-10-16/checks.json)、[pytest.txt](../eval-results/maintenance-2026-10-16/pytest.txt)、[coverage.json](../eval-results/maintenance-2026-10-16/coverage.json)。新回归集中在 [test_regressions.py](../tests/test_regressions.py)。

仍有的实现边界：仅 Weather Mock 合同；关键词/元数据检索；内存 Checkpoint；模型规划与 Replay/HITL 是独立模式。SDK 原生超时与重试不等于整个请求的硬截止时间。本次没有调用付费模型，没有重新构建 Docker，也没有运行云端 CI。
