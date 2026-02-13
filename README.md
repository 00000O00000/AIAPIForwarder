# AI API Gateway

轻量级 AI 接口网关，支持：
- 多提供商调度（优先级 + 权重）
- 自动重试与故障切换
- OpenAI / Claude / Gemini 请求格式兼容
- 上游协议格式自动转换（`openai` / `openai-response` / `claude` / `gemini`）
- Provider 限额与周期重置
- Provider 级并发上限（`rate_limit.max_worker`）

## 快速开始

1. 配置 `config/provider.json`
2. 启动：

```bash
docker-compose up -d
```

3. 验证：

```bash
curl http://localhost:6010/health
curl http://localhost:6010/v1/models
```

## 配置示例

见：`config/provider.example.json`

## 调度流程

```mermaid
flowchart TD
    Start(["客户端请求到达"]) --> Parse["解析请求体<br/>识别 client_format / model / stream"]
    Parse --> GetPriority["获取 model 的全部 provider<br/>按 priority 分组排序"]
    GetPriority --> HasPriority{{"还有未尝试的<br/>优先级组?"}}
    HasPriority -- 否 --> FinalError

    HasPriority -- 是 --> EnterPriority["进入当前优先级组<br/>初始化每个 provider 的剩余重试次数"]
    EnterPriority --> RoundStart["开始一轮遍历<br/>按 weight 加权随机打乱顺序"]

    RoundStart --> NextProvider{{"组内还有<br/>可尝试的 provider?"}}
    NextProvider -- 否 --> RoundCheck

    NextProvider -- 是 --> CheckAndAcquire["原子检查：rate_limit 限额<br/>+ 尝试获取并发槽位"]
    CheckAndAcquire -- 超限 --> SkipProvider["标记该 provider 已跳过"] --> NextProvider
    CheckAndAcquire -- 并发已满 --> MarkBlocked["标记 blocked_by_max_worker"] --> NextProvider
    CheckAndAcquire -- 获取成功 --> SendRequest["转发请求到上游 provider"]

    SendRequest --> CheckResult{{"请求结果"}}
    CheckResult -- "2xx 成功" --> RecordUsage["记录用量<br/>释放 worker 槽位"] --> ReturnOK(["返回成功响应"])
    CheckResult -- "400 客户端错误" --> ReleaseAndReturn["释放 worker 槽位<br/>直接返回错误"] --> ReturnClientErr(["返回 4xx"])
    CheckResult -- "401/403/429<br/>认证/限流错误" --> SkipAuth["释放 worker 槽位<br/>标记该 provider 已跳过"] --> DecrRetry
    CheckResult -- "5xx / 超时<br/>服务端错误" --> ReleaseRetry["释放 worker 槽位"] --> DecrRetry

    DecrRetry["扣减重试次数"] --> NextProvider

    RoundCheck{{"本轮是否有<br/>provider 被尝试过?"}}
    RoundCheck -- "是（有 provider 被调用）" --> RoundStart
    RoundCheck -- "否 + 被 max_worker 阻塞" --> EnterQueue["进入优先级排队<br/>wait_for_priority_capacity"]

    EnterQueue --> QueueResult{{"排队结果"}}
    QueueResult -- "队列溢出 / 超时 / 禁用" --> HasPriority
    QueueResult -- "被唤醒：有容量" --> RoundStart

    RoundCheck -- "否 + 无阻塞<br/>（全部跳过/耗尽重试）" --> HasPriority

    FinalError{{"存在 last_error?"}}
    FinalError -- 是 --> ReturnLastErr(["返回最后一个上游错误"])
    FinalError -- "否 + 曾被并发限制" --> Return429(["返回 429 All providers busy"])
    FinalError -- "否 + 无可用 provider" --> Return502(["返回 502 No provider"])

    style Start fill:#2d2d2d,stroke:#fff,color:#fff
    style ReturnOK fill:#2d2d2d,stroke:#fff,color:#0f0
    style ReturnClientErr fill:#2d2d2d,stroke:#fff,color:#f80
    style ReturnLastErr fill:#2d2d2d,stroke:#fff,color:#f00
    style Return429 fill:#2d2d2d,stroke:#fff,color:#f80
    style Return502 fill:#2d2d2d,stroke:#fff,color:#f00
```

## 核心配置说明

### `_global`
- `default_timeout`: 默认超时秒数
- `default_retry`: 默认重试次数
- `queue_overflow_factor`: 优先级队列溢出倍率（默认 `2.0`）
- `log_requests`: 是否记录请求日志
- `api_key`: 网关鉴权密钥（为空则不启用）

`queue_overflow_factor` 规则：
- `null`：按默认值 `2.0`
- `< 1`：按 `1.0` 处理
- `= 1`：不排队，当前优先级满并发后直接尝试下一优先级
- `NaN/Infinity/非法值`：按默认值 `2.0`

鉴权支持：
- `Authorization: Bearer <gateway_api_key>`
- `x-api-key: <gateway_api_key>`
- `x-goog-api-key: <gateway_api_key>`
- 查询参数 `?key=<gateway_api_key>`

### 模型层
- `providers`: provider 列表
- `max_worker`: 已废弃（若配置会被忽略，请改为在每个 provider 的 `rate_limit.max_worker` 配置）

### Provider 层
- `name`: provider 标识
- `endpoint`: 上游地址
- `api_key`: 上游 key
- `model`: 上游模型名
- `format`: `openai` / `openai-response` / `claude` / `gemini`
- `priority`: 优先级（越小越优先）
- `weight`: 同优先级权重（加权随机：权重越高被优先尝试的概率越大）
- `rate_limit`: 限额和并发配置
- `retry`: provider 重试次数
- `timeout`: provider 超时秒数
- `stream_support`: 是否支持流式
- `non_stream_support`: 是否支持非流式
- `enabled`: 是否启用
- `custom_headers`: 自定义请求头
- `max_context_length`: 最大上下文长度（预留）

### `rate_limit`
- `requests_per_period`: 周期内请求上限
- `tokens_per_period`: 周期内 token 上限
- `max_worker`: provider 当前最大并发上限（关键）
- `period_cron`: 重置周期（5 位 Cron）

最小示例：
```json
"rate_limit": {
  "requests_per_period": null,
  "tokens_per_period": null,
  "max_worker": 1,
  "period_cron": "0 0 * * *"
}
```

并发行为：
- `max_worker` 达到上限时，该 provider 会被调度器自动跳过；
- 网关会尝试同模型下的其他 provider，避免该 key 触发上游 429。
- 当同一优先级的 provider 都因 `max_worker` 满并发而不可用时，网关会进入该优先级排队。
- 若排队请求数超过 `（该优先级所有 provider 的 max_worker 总和）* queue_overflow_factor`，后续请求会自动转移到下一优先级。
- 排队上限按 `floor(总max_worker * queue_overflow_factor)` 计算，确保语义是“超过阈值才转移”。

## API 路由

### OpenAI 兼容
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`

### Claude 兼容
- `POST /v1/messages`

### Gemini 兼容
- `POST /v1beta/models/<model>:generateContent`
- `POST /v1beta/models/<model>:streamGenerateContent`
- `POST /v1/models/<model>:generateContent`
- `POST /v1/models/<model>:streamGenerateContent`

### 管理接口
- `GET /health`
- `GET /admin/stats`
- `POST /admin/reload`
- `GET /admin/providers/<model_name>`

## 环境变量

- `CONFIG_PATH`（默认 `/app/config/provider.json`）
- `USAGE_DATA_DIR`（默认 `/app/data/usage`）
- `ENABLE_SCHEDULER`（默认 `true`）
- `LOG_LEVEL`（默认 `INFO`）
- `TZ`

## 说明

- 多实例部署时，`max_worker` 是进程内并发；若要全局并发，建议加外部协调（Redis/网关限流层）。
- 使用反向代理时，请关闭 SSE 缓冲。
