"""Fakes and builders shared by the offline tests."""

import base64
import io

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from PIL import Image

from models.llm_clients import LLMClients


class FakeLLM(RunnableLambda):
    """
    A stand-in chat model. `reply` is a string or a callable(prompt_text)
    returning a string; raising inside the callable simulates a provider
    error. Every prompt it receives is recorded in .prompts.
    """

    def __init__(self, reply="A fake summary. It has two sentences."):
        self.prompts: list[str] = []
        self._reply = reply
        super().__init__(self._respond)

    def _respond(self, value):
        text = _prompt_text(value)
        self.prompts.append(text)
        reply = self._reply(text) if callable(self._reply) else self._reply
        return AIMessage(content=reply)


def _prompt_text(value) -> str:
    """Flatten whatever a chain or a direct .invoke() passes to the model."""
    messages = value.to_messages() if hasattr(value, "to_messages") else value
    if isinstance(messages, str):
        return messages
    parts = []
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            parts.append(content)
        else:
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, dict) and block.get("type") == "image_url":
                    parts.append("<image>")
    return "\n".join(parts)


def make_clients(**overrides) -> LLMClients:
    """An LLMClients bundle of FakeLLMs; pass fields to override individual ones."""
    fields = {
        "table_summary_llm": FakeLLM("Summary of a table."),
        "image_summary_llm": FakeLLM("Summary of a figure."),
        "text_summary_llm": FakeLLM("Summary of a passage."),
        "agent_llm": FakeLLM("text_query"),
        "agent_vision_llm": FakeLLM("The figure shows an encoder and a decoder. See (page 3)."),
    }
    fields.update(overrides)
    return LLMClients(**fields)


def jpeg_base64(width=40, height=30, color=(200, 30, 30)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_pdf_bytes(pages=2, marker="paper") -> bytes:
    """A real, minimal PDF. `marker` makes the bytes (and so the paper id) unique."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": marker})
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
