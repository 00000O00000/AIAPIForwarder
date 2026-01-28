"""
工具函数
"""

import time
import hashlib
from typing import Optional


def generate_request_id() -> str:
    """生成请求ID"""
    return hashlib.md5(f"{time.time()}".encode()).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量
    简单估算：英文约 4 字符一个 token，中文约 1.5 字符一个 token
    """
    # 简单实现：每4个字符算一个token
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list) -> int:
    """估算消息列表的 token 数量"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total += estimate_tokens(item["text"])
    return total


def safe_int(value, default: int = 0) -> int:
    """安全转换为整数"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default