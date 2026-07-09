"""
实时消息总线 - 模型间通信的核心基础设施

支持频道订阅、广播/定向消息、历史回溯。
贯穿研讨和执行两个阶段。
"""

import uuid
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────────
# 消息
# ──────────────────────────────────────────────


@dataclass
class Message:
    """总线上的单条消息"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    channel: str = ""
    sender: str = ""
    content: str = ""
    recipient: Optional[str] = None   # None = 广播
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")

    def summary(self) -> str:
        label = f"[{self.channel}] {self.sender}"
        if self.recipient:
            label += f" -> {self.recipient}"
        preview = self.content[:80].replace("\n", " ")
        if len(self.content) > 80:
            preview += "..."
        return f"{self.time_str} {label}: {preview}"


# ──────────────────────────────────────────────
# 消息总线
# ──────────────────────────────────────────────


# 预定义频道
CHANNEL_DEBATE = "debate"         # 圆桌研讨
CHANNEL_PROGRESS = "progress"     # 执行进度
CHANNEL_PROBLEM = "problem"       # 问题求助
CHANNEL_SOLUTION = "solution"     # 问题解答
CHANNEL_CONTROL = "control"       # 主模型指令
CHANNEL_RESULT = "result"         # 任务结果
CHANNEL_CROSS_REVIEW = "review"   # 交叉验证


class MessageBus:
    """消息总线 - 频道路由 + 历史存档"""

    def __init__(self):
        self._channels: dict[str, list[Message]] = defaultdict(list)
        self._all_messages: list[Message] = []

    # ── 发布 ──

    def publish(
        self,
        channel: str,
        sender: str,
        content: str,
        recipient: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Message:
        msg = Message(
            channel=channel,
            sender=sender,
            content=content,
            recipient=recipient,
            metadata=metadata or {},
        )
        self._channels[channel].append(msg)
        self._all_messages.append(msg)
        return msg

    # ── 读取 ──

    def get_channel_history(self, channel: str) -> list[Message]:
        """获取指定频道的全部历史"""
        return list(self._channels.get(channel, []))

    def get_all_messages(self) -> list[Message]:
        """按时间顺序返回全部消息"""
        return list(self._all_messages)

    def get_messages_since(self, message_id: str) -> list[Message]:
        """获取某条消息之后的所有消息"""
        found = False
        result = []
        for msg in self._all_messages:
            if found:
                result.append(msg)
            elif msg.id == message_id:
                found = True
        return result

    def get_latest(self, channel: str, n: int = 5) -> list[Message]:
        """获取频道最近 N 条消息"""
        hist = self._channels.get(channel, [])
        return hist[-n:]

    def search(self, keyword: str, channel: Optional[str] = None) -> list[Message]:
        """全文搜索消息内容"""
        source = self._all_messages if channel is None else self._channels.get(channel, [])
        return [m for m in source if keyword.lower() in m.content.lower()]

    # ── 构建上下文 ──

    def format_channel_context(self, channel: str, max_messages: int = 50) -> str:
        """将频道历史格式化为对话上下文文本"""
        history = self._channels.get(channel, [])[-max_messages:]
        lines = []
        for m in history:
            tag = f"{m.sender} → {m.recipient}" if m.recipient else m.sender
            lines.append(f"[{m.time_str}] {tag}: {m.content}")
        return "\n\n".join(lines)

    def format_full_debate(self) -> str:
        """格式化整个研讨记录（便于喂给主模型）"""
        return self.format_channel_context(CHANNEL_DEBATE)

    # ── 打印 ──

    def print_transcript(self, channel: Optional[str] = None):
        """打印消息记录到控制台"""
        messages = self._all_messages if channel is None else self._channels.get(channel, [])
        for m in messages:
            print(m.summary())

    def print_channel(self, channel: str):
        """打印指定频道的全部消息（含完整内容）"""
        import sys
        print(f"\n{'='*60}")
        print(f"  频道: {channel}")
        print(f"{'='*60}")
        for m in self._channels.get(channel, []):
            label = f"\n[{m.time_str}] {m.sender}"
            if m.recipient:
                label += f" -> {m.recipient}"
            print(label)
            print("-" * 40)
            text = m.content.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
            print(text)
        print(f"{'='*60}\n")

    # ── 统计 ──

    def stats(self) -> dict:
        return {
            "total_messages": len(self._all_messages),
            "channels": {ch: len(msgs) for ch, msgs in self._channels.items()},
            "participants": list(set(m.sender for m in self._all_messages)),
        }
