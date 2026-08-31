from __future__ import annotations

import requests

from config import Settings


def send_pushplus(settings: Settings, title: str, content: str) -> None:
    if not settings.pushplus_token:
        print("未设置 PUSHPLUS_TOKEN，跳过微信推送。")
        return

    try:
        requests.post(
            "http://www.pushplus.plus/send",
            json={
                "token": settings.pushplus_token,
                "title": title,
                "content": content,
                "template": "html",
            },
            timeout=settings.request_timeout,
        )
        print("微信通知已发送。")
    except Exception as exc:
        print(f"微信通知失败: {exc}")
