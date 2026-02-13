# AI API Gateway

轻量级 AI 接口网关，提供以下能力：
- 多上游提供商调度（优先级 + 权重）
- 失败自动切换与重试
- OpenAI / Claude / Gemini 调用格式兼容
- 提供商协议格式自动转换（`openai` / `openai-response` / `claude` / `gemini`）
- Provider 限额与定时重置
- 模型级并发上限（`max_worker`）

## 快速开始

1. 配置 `config/provider.json`
2. 启动服务：

```bash
docker-compose up -d
```

3. 验证：

```bash
curl http://localhost:6010/health
curl http://localhost:6010/v1/models
```

## provider.json 结构

```json
{
  "_global": {
    "default_timeout": 120,
    "default_retry": 3,
    "log_requests": true,
    "api_key": ""
  },
  "gpt-4o": {
    "max_worker": 16,
    "providers": [
      {
        "name": "provider-1",
        "endpoint": "https://api.example.com/v1",
        "api_key": "sk-xxx",
        "model": "gpt-4o",
        "format": "openai",
        "priority": 1,
        "weight": 10,
        "rate_limit": {
          "requests_per_period": 100,
          "tokens_per_period": 500000,
          "period_cron": "0 0 * * *"
        },
        "retry": 3,
        "timeout": 60,
        "stream_support": true,
        "non_stream_support": true,
        "enabled": true,
        "custom_headers": {
          "X-Env": "prod"
        },
        "max_context_length": 128000
      }
    ]
  }
}
```

## 配置说明

### 全局配置 `_global`
- `default_timeout`: 默认超时秒数
- `default_retry`: 默认重试次数
- `log_requests`: 是否记录请求日志
- `api_key`: 网关鉴权密钥，留空表示不启用

网关鉴权支持以下任一方式：
- `Authorization: Bearer <gateway_api_key>`
- `x-api-key: <gateway_api_key>`
- `x-goog-api-key: <gateway_api_key>`
- 查询参数 `?key=<gateway_api_key>`

### 模型级配置
- `max_worker`: 模型当前最大并发请求数
  - `null` 或不填：不限制
  - 达到上限后，新请求会跳过该模型并返回 `429`
- `providers`: 上游提供商列表

### Provider 配置
- `name`: 提供商唯一标识
- `endpoint`: 上游基础地址
- `api_key`: 上游密钥
- `model`: 上游真实模型名
- `format`: 上游协议格式
  - 可选：`openai`、`openai-response`、`claude`、`gemini`
- `priority`: 优先级（越小越优先）
- `weight`: 同优先级权重
- `rate_limit`: 限额配置
- `retry`: 每个 provider 的重试次数
- `timeout`: 上游请求超时秒数
- `stream_support`: 是否支持流式
- `non_stream_support`: 是否支持非流式
- `enabled`: 是否启用
- `custom_headers`: 自定义请求头
- `max_context_length`: 最大上下文长度（预留）

### 限额配置 `rate_limit`
- `requests_per_period`: 周期内最大请求数
- `tokens_per_period`: 周期内最大 token 数
- `period_cron`: 重置周期（5 位 Cron）

## 调用格式兼容

### 客户端格式自动适配
- 若客户端按 Claude Messages 或 Gemini generateContent 格式请求，网关会自动识别并转换到内部统一格式。
- 响应会按客户端原始格式返回（OpenAI / Claude / Gemini）。

### 提供商格式主动适配
- 每个 provider 可通过 `format` 指定上游协议。
- 网关在转发时自动做请求与响应转换。
- 转换时尽量保留可映射参数，未映射字段尽量透传。

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

## 请求示例

### OpenAI Chat Completions

```bash
curl http://localhost:6010/v1/chat/completions \
  -H "Authorization: Bearer <gateway_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false
  }'
```

### Claude Messages

```bash
curl http://localhost:6010/v1/messages \
  -H "Authorization: Bearer <gateway_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
  }'
```

### Gemini generateContent

```bash
curl http://localhost:6010/v1beta/models/gpt-4o:generateContent \
  -H "Authorization: Bearer <gateway_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {"role": "user", "parts": [{"text": "hello"}]}
    ],
    "generationConfig": {"temperature": 0.7}
  }'
```

## 环境变量

- `CONFIG_PATH`: 配置文件路径（默认 `/app/config/provider.json`）
- `USAGE_DATA_DIR`: 使用量数据目录（默认 `/app/data/usage`）
- `ENABLE_SCHEDULER`: 是否启用限额重置调度（默认 `true`）
- `LOG_LEVEL`: 日志级别（默认 `INFO`）
- `TZ`: 时区

## 说明

- 若使用反向代理，请对 SSE 流式响应关闭缓冲。
- 并发控制为进程内计数；多实例部署时建议结合外部限流组件做全局并发控制。
