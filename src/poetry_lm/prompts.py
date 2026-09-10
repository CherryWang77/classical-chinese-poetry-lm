from __future__ import annotations

from .research_types import TemplateSpec

SYSTEM_PROMPT = "你是宋词创作助手，只输出词作正文，不解释。"


def user_prompt(template: TemplateSpec, theme: str | None = None) -> str:
    topic = f"请以“{theme}”为主题，" if theme else "请"
    return (
        f"{topic}依照词牌《{template.cipai}》和句式 "
        f"{template.format_string()} 创作一首宋词。"
    )


def character_control_prefix(template: TemplateSpec, theme: str | None = None) -> str:
    theme_text = theme or "自选"
    return f"词牌：{template.cipai}\n主题：{theme_text}\n句式：{template.format_string()}\n正文："
