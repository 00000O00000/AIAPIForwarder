FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/

# 创建必要目录
RUN mkdir -p /app/data/usage /app/logs

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 6010

CMD ["gunicorn", "--bind", "0.0.0.0:6010", "--workers", "1", "--threads", "8", "--timeout", "120", "app.main:app"]