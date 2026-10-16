# GitHub 复用与边界决策

> 调研日期：2026-08-16  
> 适用范围：`api-diagnosis-agent` 的 V0 基础和后续 V1/V2 演进  
> 说明：以下是工程复用决策记录，不是对任何许可证的法律意见。真正发布或分发前，仍需核对所安装版本的许可证、NOTICE 和传递依赖。

## 1. 结论

采用“依赖成熟库 + 参考测试思路 + 自己实现业务闭环”的方案：

1. V0 直接依赖 FastAPI、Uvicorn、HTTPX、`openapi-core`、`openapi-spec-validator` 和 pytest；V1/V2 在同一条主线上加入 `langchain-core` 与 LangGraph。
2. OpenAPI 规范校验与请求/响应校验使用上游库的公开 API；不复制其源码，不复制完整仓库。
3. Mock API、故障注入、白名单重放、Ground Truth、版本规则和诊断报告属于本项目自己的业务代码。
4. V1 使用 LangChain Core 的 `Document`/`BaseRetriever`/structured tools 与 LangGraph 的 StateGraph/checkpoint；默认仍然离线，不调用第三方 API。V2 的 trace 使用本地结构化事件，未来才评估 OpenTelemetry 后端。
5. 每次升级依赖时重新运行固定案例，并把实际解析出的版本和许可证补入 `THIRD_PARTY.md`。

推荐的复用关系：

```text
openapi-spec-validator  -> 校验本项目 OpenAPI 文档本身
openapi-core            -> 对受控请求/响应执行确定性验证
FastAPI + Uvicorn       -> 自建 Mock API 服务与本地运行
urllib.request          -> 仅访问白名单 Mock URL 的无代理重放客户端
HTTPX                   -> FastAPI TestClient 和 HTTP 测试工具
pytest                  -> 固定案例、工具和端到端回归
langchain-core         -> Document、BaseRetriever、结构化工具和可选模型接口
langgraph               -> StateGraph、条件边、checkpoint 和 HITL 扩展点
```

## 2. 候选仓库与复用范围

| 仓库 | 许可证/维护信号 | 可复用内容 | 本项目边界与决定 |
| --- | --- | --- | --- |
| [python-openapi/openapi-core](https://github.com/python-openapi/openapi-core) / [LICENSE](https://github.com/python-openapi/openapi-core/blob/master/LICENSE) | BSD-3-Clause；提供 OpenAPI 请求/响应验证与反序列化 | 验证器公开 API、异常分类思路、请求/响应适配方式 | 作为 V0 确定性验证依赖；不复制源码，不把其适配器当作本项目业务逻辑；升级时重新核对 API |
| [python-openapi/openapi-spec-validator](https://github.com/python-openapi/openapi-spec-validator) / [LICENSE](https://github.com/python-openapi/openapi-spec-validator/blob/master/LICENSE) | Apache-2.0；面向 OpenAPI 文档合规性验证 | CI/启动前验证 OpenAPI 文件的方式 | 只负责“规范文件是否合法”，不替代请求/响应故障诊断；不复制 CLI 或测试代码 |
| [fastapi/fastapi](https://github.com/fastapi/fastapi) / [LICENSE](https://github.com/fastapi/fastapi/blob/master/LICENSE) | MIT；成熟 Python ASGI API 框架 | 路由、Pydantic 输入模型、错误响应和自动生成 OpenAPI | 作为 V0 Mock 服务框架；自己定义 Endpoint 与故障注入，不复制示例应用 |
| [encode/uvicorn](https://github.com/encode/uvicorn) / [LICENSE](https://github.com/encode/uvicorn/blob/master/LICENSE.txt) | BSD-3-Clause；ASGI 运行器 | 本地启动和受控测试服务 | 只作为运行依赖；不引入多进程或生产部署设计 |
| [encode/httpx](https://github.com/encode/httpx) / [LICENSE](https://github.com/encode/httpx/blob/master/LICENSE.md) | BSD-3-Clause；支持同步/异步 HTTP 与严格超时 | FastAPI TestClient 和 HTTP 边界测试 | 运行时 Replay 使用标准库 `urllib.request` 并禁用系统代理，避免额外客户端抽象 |
| [pytest-dev/pytest](https://github.com/pytest-dev/pytest) / [LICENSE](https://github.com/pytest-dev/pytest/blob/main/LICENSE) | MIT；Python 测试框架 | 参数化固定案例、夹具和回归测试组织方式 | 作为开发依赖；测试数据和判定标准由本项目维护 |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) / [LICENSE](https://github.com/langchain-ai/langchain/blob/master/LICENSE) | MIT；LangChain Core 提供稳定的 Document、Retriever、Tool 和 model protocol | `Document`、`BaseRetriever`、`@tool`、structured output 边界 | 只使用 Core 接口；不复制 Agent runtime，不引入第二套 Agent 框架 |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) / [LICENSE](https://github.com/langchain-ai/langgraph/blob/main/LICENSE) | MIT；StateGraph、条件边、持久化 checkpoint 和 HITL 扩展 | V1 Agent 状态图、最大两次重试和 MemorySaver | 图中节点只调用本项目工具；模型调用需显式配置 provider |

## 3. 明确不复用的候选

### 3.1 Dredd

[apiaryio/dredd](https://github.com/apiaryio/dredd) 采用 MIT 许可证，但仓库已被归档为只读状态；它是 Node.js 生态的语言无关 CLI，主要通过 API 描述文件驱动契约测试和 HTTP 事务回放。

本项目不把 Dredd 加入 Python 运行时、CI 或 Docker 依赖，也不复制其 CLI、hooks、测试运行器或仓库代码。原因是：

- V0 的目标是学习并验证 Python 确定性工具链，而不是再引入一个 Node.js 测试运行时；
- Dredd 的 API Blueprint/OpenAPI 事务模型与本项目的“故障类别—根因—证据—修复后重放”Ground Truth 仍有差距；
- 已归档状态会增加升级、维护和面试解释成本。

允许的参考范围仅限于：阅读其契约测试/事务矩阵的产品思路，转化为自己的 pytest 案例；不复制代码或文档段落。

### 3.2 Pydantic AI

[pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) 采用 MIT 许可证，功能覆盖类型安全 Agent、工具和模型集成。本项目不把它作为依赖，也不复制其 Agent、工具集或运行时实现：

- V0 明确先验证不用 Agent 的确定性工具；
- 后续 Agent 编排已经规划由 LangChain/LangGraph 承担，引入第二套 Agent 框架会扩大依赖和面试解释面；
- Pydantic 模型或工具 Schema 的通用设计可以作为阅读材料，但不能把 Pydantic AI 的代码包装成自己的核心实现。

### 3.3 Langfuse Enterprise Edition

[langfuse/langfuse](https://github.com/langfuse/langfuse) 的核心部分采用 MIT 许可证，但仓库明确说明 `ee/` 目录包含企业特性，相关功能可能需要商业许可证。V0 不引入 Langfuse、Langfuse Cloud 或自托管 EE，不复制 `ee/` 代码、镜像、生成文件、受限 API 或许可证门控逻辑。

V2 已使用本地结构化 trace 记录调用、耗时、错误与证据；若后续需要远程 Trace 后端，只能在明确选择 OSS 核心/官方 SDK、核对版本许可证并完成脱敏设计后再评估。不能为了展示“可观测性”而依赖 EE 功能。

## 4. 许可证与代码来源规则

- 依赖库通过包管理器安装，保留其版权和许可证信息；不把依赖源码复制到 `src/`。
- 若未来确需复制一小段代码，必须先记录原始仓库、提交/版本、文件路径、许可证、改动内容和对应 NOTICE 要求，并经过一次独立许可证复核。
- 不允许把不同许可证的整仓库代码混合后只改项目名、Prompt 或界面。
- 项目自己的 Mock API、故障注入器、验证编排、Ground Truth、评测指标和报告格式必须有独立实现和测试证据。
- `THIRD_PARTY.md` 记录直接依赖、主要传递依赖、准确版本、许可证和来源；发布前仍需由最终镜像生成 SBOM。

## 5. 待确认与升级门槛

- `pyproject.toml` 给出保守版本范围，`requirements*.lock` 固定当前解析版本；锁文件未包含 wheel hash，发布前仍需生成 SBOM。
- 版本范围以 2026-08-16 可查到的稳定版本为基线：FastAPI 0.139.2（[PyPI](https://pypi.org/project/fastapi/)）使用 `>=0.136,<0.140`，Uvicorn 0.51.0（[PyPI](https://pypi.org/project/uvicorn/)）使用 `>=0.35,<0.52`，HTTPX 0.28.1（[PyPI](https://pypi.org/project/httpx/)）使用 `>=0.28,<0.29`，openapi-core 0.23.1（[PyPI](https://pypi.org/project/openapi-core/)）使用 `>=0.23,<0.24`，openapi-spec-validator 暂用兼容的 0.8.x 线（`>=0.8.5,<0.9`），pytest 9.1.1（[PyPI](https://pypi.org/project/pytest/)）使用 `>=9,<10`。
- 选择 openapi-spec-validator 0.8.x 是有证据的依赖兼容性决策：0.9.x 要求 `jsonschema-path>=0.5`，而 openapi-core 0.23.x 要求 `<0.5`；V0 先保持同一环境可解析，后续再单独评估升级方案。
- 这些上限有意限制在下一小版本（pytest 限制在下一大版本）：0.x 库的小版本可能包含不兼容 API 变化，V0 需要可重复验证；上限放宽前必须先运行固定案例并更新许可证清单。FastAPI 的范围保留 0.136—0.139 是为了接收同一 API 代际的修复版本，而不是追逐任意最新版本。
- `openapi-core` 与 `openapi-spec-validator` 的职责不同：前者验证请求/响应，后者验证 OpenAPI 文档，不能互相替代。
- V0 不因这些候选仓库而加入 Dredd、Pydantic AI 或 Langfuse；LangChain Core/LangGraph 仅在 V1/V2 的已定义边界内使用。
- 任一后续依赖升级都必须通过固定正常案例、每类故障案例、拒绝越权 URL 案例和许可证清单检查。
