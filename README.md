# 🚀 AI API Gateway

一个轻量级、高可用的 AI API 中转网关系统，支持多提供商负载均衡、智能故障转移、限额管理等功能。

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## ✨ 功能特性

### 核心功能

| 功能 | 描述 |
|------|------|
| 🔄 **多提供商支持** | 同一模型可配置多个上游提供商，实现冗余备份 |
| ⚖️ **智能负载均衡** | 基于优先级 + 权重的两级调度策略 |
| 🔁 **自动故障转移** | 5xx 错误自动重试，4xx 错误自动切换提供商 |
| 📊 **限额管理** | 支持请求次数和 Token 数量双重限制 |
| ⏰ **定时重置** | 使用 Cron 表达式灵活配置限额刷新周期 |
| 🌊 **流式兼容** | 自动转换流式/非流式响应格式 |

### 扩展功能

- 🔐 **网关认证** - 可选的 API Key 保护
- 📈 **使用统计** - 实时查看各提供商使用情况
- 🔄 **热重载** - 不重启服务更新配置
- 🏥 **健康检查** - 支持 Docker 健康检查
- 📝 **请求日志** - 完整的请求链路追踪

## 🏗️ 系统架构

```
                                    ┌─────────────────┐
                                    │   Provider A    │
                                    │  (Priority: 1)  │
┌──────────┐     ┌──────────────┐   ├─────────────────┤
│  Client  │────▶│  API Gateway │──▶│   Provider B    │
└──────────┘     └──────────────┘   │  (Priority: 1)  │
                        │           ├─────────────────┤
                        │           │   Provider C    │
                        ▼           │  (Priority: 2)  │
                 ┌──────────────┐   └─────────────────┘
                 │ provider.json│
                 │  usage_data/ │
                 └──────────────┘
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/ai-api-gateway.git
cd ai-api-gateway
```

### 2. 配置提供商

编辑 `config/provider.json`：

```json
{
  "_global": {
    "default_timeout": 120,
    "api_key": ""
  },
  "gpt-4o": {
    "providers": [
      {
        "name": "provider-1",
        "endpoint": "https://api.example.com/v1",
        "api_key": "sk-your-api-key",
        "model": "gpt-4o",
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
        "enabled": true
      }
    ]
  }
}
```

### 3. 启动服务

```bash
docker-compose up -d
```

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:6010/health

# 列出模型
curl http://localhost:6010/v1/models

# 发送测试请求
curl http://localhost:6010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 📖 配置说明

### 目录结构

```
ai-api-gateway/
├── docker-compose.yml      # Docker 编排配置
├── Dockerfile              # 镜像构建文件
├── requirements.txt        # Python 依赖
├── config/
│   └── provider.json       # 提供商配置文件
├── data/
│   └── usage/              # 使用量数据（自动生成）
├── logs/
│   └── gateway.log         # 运行日志
└── app/
    ├── main.py             # 应用入口
    ├── config.py           # 配置管理
    ├── models.py           # 数据模型
    ├── provider_manager.py # 提供商管理
    ├── proxy.py            # 请求代理
    ├── rate_limiter.py     # 限额管理
    ├── scheduler.py        # 定时任务
    └── utils.py            # 工具函数
```

### 全局配置 (`_global`)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `default_timeout` | int | 120 | 默认请求超时时间（秒） |
| `default_retry` | int | 3 | 默认重试次数 |
| `log_requests` | bool | true | 是否记录请求日志 |
| `api_key` | string | "" | 网关认证密钥（留空则不验证） |

### 模型配置

每个模型（如 `gpt-4o`、`deepseek-chat`）包含一个 `providers` 数组：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `name` | string | ✅ | - | 提供商唯一标识 |
| `endpoint` | string | ✅ | - | API 端点 URL |
| `api_key` | string | ✅ | - | 提供商 API 密钥 |
| `model` | string | ✅ | - | 上游实际模型名 |
| `priority` | int | ❌ | 1 | 优先级（越小越优先） |
| `weight` | int | ❌ | 10 | 同优先级轮询权重 |
| `rate_limit` | object | ❌ | null | 限额配置 |
| `retry` | int | ❌ | 3 | 重试次数 |
| `timeout` | int | ❌ | 60 | 超时时间（秒） |
| `stream_support` | bool | ❌ | true | 是否支持流式响应 |
| `non_stream_support` | bool | ❌ | true | 是否支持非流式响应 |
| `enabled` | bool | ❌ | true | 是否启用 |
| `custom_headers` | object | ❌ | null | 自定义请求头 |
| `max_context_length` | int | ❌ | null | 最大上下文长度 |

### 限额配置 (`rate_limit`)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `requests_per_period` | int | null | 周期内最大请求数（null 不限制） |
| `tokens_per_period` | int | null | 周期内最大 Token 数（null 不限制） |
| `period_cron` | string | "0 0 * * *" | 重置周期 Cron 表达式 |

### Cron 表达式示例

| 表达式 | 说明 |
|--------|------|
| `0 0 * * *` | 每天 00:00 重置 |
| `0 */6 * * *` | 每 6 小时重置 |
| `0 0 * * 0` | 每周日 00:00 重置 |
| `0 0 * * 1` | 每周一 00:00 重置 |
| `0 0 1 * *` | 每月 1 号 00:00 重置 |
| `*/30 * * * *` | 每 30 分钟重置 |

> 格式：`分 时 日 月 周` （标准 5 位 Cron 表达式）

## 📡 API 文档

### OpenAI 兼容接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 列出所有可用模型 |
| `/v1/chat/completions` | POST | Chat Completion |
| `/v1/completions` | POST | Text Completion |
| `/v1/embeddings` | POST | 文本向量化 |

### 管理接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/admin/stats` | GET | 获取使用统计 |
| `/admin/reload` | POST | 热重载配置 |
| `/admin/providers/<model>` | GET | 获取模型提供商状态 |

### 请求示例

#### 列出模型

```bash
curl http://localhost:6010/v1/models \
  -H "Authorization: Bearer your-gateway-key"
```

响应：
```json
{
  "object": "list",
  "data": [
    {"id": "gpt-4o", "object": "model", "owned_by": "api-gateway"},
    {"id": "deepseek-chat", "object": "model", "owned_by": "api-gateway"}
  ]
}
```

#### Chat Completion（非流式）

```bash
curl http://localhost:6010/v1/chat/completions \
  -H "Authorization: Bearer your-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7
  }'
```

#### Chat Completion（流式）

```bash
curl http://localhost:6010/v1/chat/completions \
  -H "Authorization: Bearer your-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Write a poem"}],
    "stream": true
  }'
```

#### 查看使用统计

```bash
curl http://localhost:6010/admin/stats
```

响应：
```json
{
  "gpt-4o": {
    "provider-1": {
      "requests": 42,
      "tokens": 15680,
      "last_reset": "2024-01-15T00:00:00",
      "limit_requests": 100,
      "limit_tokens": 601000
    }
  }
}
```

#### 热重载配置

```bash
curl -X POST http://localhost:6010/admin/reload
```

## 🔧 高级配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TZ` | UTC | 时区设置 |
| `LOG_LEVEL` | INFO | 日志级别 (DEBUG/INFO/WARNING/ERROR) |
| `FLASK_ENV` | production | Flask 运行环境 |

### Docker Compose 配置

```yaml
services:
  api-gateway:
    build: .
    container_name: ai-api-gateway
    ports:
      - "6010:6010"
    volumes:
      - ./config:/app/config:ro    # 配置文件（只读）
      - ./data:/app/data            # 使用量数据
      - ./logs:/app/logs            # 日志文件
    environment:
      - TZ=Asia/Shanghai
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6010/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 生产环境建议

1. **设置网关 API Key**
   ```json
   {
     "_global": {
       "api_key": "your-secure-gateway-key"
     }
   }
   ```

2. **配置反向代理（Nginx）**
   ```nginx
   server {
       listen 443 ssl;
       server_name api.yourdomain.com;

       location / {
           proxy_pass http://localhost:6010;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_read_timeout 300s;
           proxy_buffering off;  # 重要：流式响应需要
       }
   }
   ```

3. **日志轮转**
   ```bash
   # /etc/logrotate.d/ai-gateway
   /path/to/logs/*.log {
       daily
       rotate 7
       compress
       missingok
       notifempty
   }
   ```

## 🔄 故障转移逻辑

```
请求进入
    │
    ▼
选择最高优先级的可用提供商
    │
    ▼
发送请求 ──────────────────┐
    │                      │
    ▼                      │
成功 (2xx)?                │
    │                      │
    ├── 是 ──▶ 返回响应     │
    │                      │
    └── 否                  │
         │                 │
         ▼                 │
    5xx 错误?              │
         │                 │
         ├── 是 ──▶ 重试（最多N次）
         │         超过次数则切换提供商
         │                 │
         └── 否            │
              │            │
              ▼            │
         4xx 错误?         │
              │            │
              ├── 401/403/429 ──▶ 切换提供商
              │            │
              └── 其他 ──▶ 返回错误
                           │
                           ▼
                   还有可用提供商?
                           │
                           ├── 是 ──▶ 选择下一个提供商
                           │
                           └── 否 ──▶ 返回 502 错误
```

## 📊 监控告警

### Prometheus 指标（可选扩展）

如需集成 Prometheus，可在 `main.py` 中添加：

```python
from prometheus_flask_exporter import PrometheusMetrics
metrics = PrometheusMetrics(app)
```

### 日志监控

关键日志关键词：
- `ERROR` - 错误事件
- `Selected provider` - 提供商选择
- `Switching provider` - 故障转移
- `Reset usage` - 限额重置

## ❓ 常见问题

### Q: 如何添加新模型？

编辑 `config/provider.json`，添加新的模型配置，然后调用：
```bash
curl -X POST http://localhost:6010/admin/reload
```

### Q: 限额用完了怎么办？

- 等待自动重置（根据 `period_cron` 配置）
- 手动删除 `data/usage/` 下对应的 JSON 文件
- 添加更多提供商作为备用

### Q: 如何查看哪个提供商被使用了？

查看日志文件 `logs/gateway.log`，或设置 `LOG_LEVEL=DEBUG` 获取详细信息。

### Q: 流式响应不工作？

1. 确保 Nginx 配置 `proxy_buffering off;`
2. 检查提供商是否支持流式 (`stream_support: true`)
3. 检查网络是否有缓冲代理

### Q: 如何备份数据？

```bash
# 备份使用量数据和日志
tar -czvf backup-$(date +%Y%m%d).tar.gz data/ logs/
```

## 📝 更新日志

### v1.0.0 (2024-01-15)
- 🎉 首次发布
- ✅ 多提供商支持
- ✅ 优先级 + 权重负载均衡
- ✅ 限额管理与自动刷新
- ✅ 故障自动转移
- ✅ 流式/非流式自动转换
- ✅ 热重载配置
- ✅ 使用统计接口

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
