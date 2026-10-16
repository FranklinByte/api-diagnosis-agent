# Third-party notices

本文件记录当前锁文件和 Docker 构建解析到的直接依赖版本。项目没有复制第三方源码；发布前仍应从最终镜像重新核对 SPDX、NOTICE 和传递依赖。当前锁文件固定版本但不含 wheel hash，不能替代 SBOM。

## 直接依赖

| Package | Resolved version | License | Source / notice | Usage |
| --- | --- | --- | --- | --- |
| fastapi | 0.139.2 | MIT | [repository](https://github.com/fastapi/fastapi) / [LICENSE](https://github.com/fastapi/fastapi/blob/master/LICENSE) | V0 Mock API and V2 service boundary |
| uvicorn | 0.51.0 | BSD-3-Clause | [repository](https://github.com/encode/uvicorn) / [LICENSE](https://github.com/encode/uvicorn/blob/master/LICENSE.txt) | Local ASGI server |
| httpx | 0.28.1 | BSD-3-Clause | [repository](https://github.com/encode/httpx) / [LICENSE](https://github.com/encode/httpx/blob/master/LICENSE.md) | TestClient and HTTP utilities |
| openapi-core | 0.23.1 | BSD-3-Clause | [repository](https://github.com/python-openapi/openapi-core) / [LICENSE](https://github.com/python-openapi/openapi-core/blob/master/LICENSE) | Deterministic request/response validation |
| openapi-spec-validator | 0.8.5 | Apache-2.0 | [repository](https://github.com/python-openapi/openapi-spec-validator) / [LICENSE](https://github.com/python-openapi/openapi-spec-validator/blob/master/LICENSE) | OpenAPI document validation |
| langchain-core | 1.5.5 | MIT | [repository](https://github.com/langchain-ai/langchain) / [LICENSE](https://github.com/langchain-ai/langchain/blob/master/LICENSE) | Documents, retriever, structured tools and model protocol |
| langgraph | 1.2.11 | MIT | [repository](https://github.com/langchain-ai/langgraph) / [LICENSE](https://github.com/langchain-ai/langgraph/blob/main/LICENSE) | StateGraph, conditional loop and checkpoint |
| langchain-openai (optional) | 1.5.1 | MIT | [repository](https://github.com/langchain-ai/langchain-openai) / [LICENSE](https://github.com/langchain-ai/langchain-openai/blob/main/LICENSE) | Explicit provider-backed structured output |
| pytest | 9.1.1 | MIT | [repository](https://github.com/pytest-dev/pytest) / [LICENSE](https://github.com/pytest-dev/pytest/blob/main/LICENSE) | Development and regression tests |

## 构建时依赖

| Package | Resolved version | License | Source / notice | Usage |
| --- | --- | --- | --- | --- |
| hatchling | 1.29.0 | MIT | [repository](https://github.com/pypa/hatchling) | Build backend; not a runtime dependency |

## 传递依赖

当前运行环境还包含以下主要传递依赖（完整解析以锁文件为准）：

- `pydantic 2.13.4` (MIT)、`starlette 1.6.0` (BSD-3-Clause)、`langsmith 0.11.0` (MIT)、`langgraph-checkpoint 4.2.0` (MIT)、`langgraph-sdk 0.4.2` (MIT)、`openai 3.1.0` (Apache-2.0, optional path)、`tiktoken 0.13.0` (MIT, optional path)。
- 其它传递包由 pip 安装元数据提供；发布镜像前应导出 lock/SBOM 并补充其许可证。

## 明确未引入

以下候选不会出现在 V0 的直接或传递依赖中：

- Dredd；
- Pydantic AI；
- Langfuse Enterprise Edition 代码、镜像或 SDK；
- 任意第二套 Agent 框架。

详见 [docs/reuse-decision.md](docs/reuse-decision.md)。许可证记录用于工程合规核对，不构成法律意见。
