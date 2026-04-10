from __future__ import annotations

import json
import os
import re
from typing import List

from groq import AsyncGroq

from extractor import ExtractedContent


GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
SUMMARY_CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", "12000"))
MAX_CHUNKS = int(os.getenv("SUMMARY_MAX_CHUNKS", "8"))
GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "90"))


class SummarizationError(Exception):
    """Raised when LLM summarization fails in a controlled way."""


class GroqSummarizer:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncGroq(api_key=api_key, timeout=GROQ_TIMEOUT)

    async def summarize(self, extracted: ExtractedContent) -> list[str]:
        text = extracted.text.strip()
        if not text:
            raise SummarizationError("No text was available for summarization.")

        if len(text) <= SUMMARY_CHUNK_SIZE:
            return await self._final_summary(text, extracted)

        partial_summaries: list[str] = []
        for index, chunk in enumerate(chunk_text(text, SUMMARY_CHUNK_SIZE)[:MAX_CHUNKS], start=1):
            partial_summaries.append(
                await self._chunk_summary(chunk=chunk, index=index)
            )

        combined = "\n\n".join(partial_summaries)
        return await self._final_summary(combined, extracted, is_from_chunk_summaries=True)

    async def _chunk_summary(self, chunk: str, index: int) -> str:
        prompt = (
            "Summarize this content chunk into 4 very concise bullet points in the same language as the chunk. "
            "Preserve names, claims, numbers, and decisions. Avoid intro/conclusion fluff.\n\n"
            f"Chunk #{index}:\n{chunk}"
        )

        try:
            response = await self.client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You compress content faithfully and keep only high-signal information.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            raise SummarizationError(f"Groq chunk summarization request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise SummarizationError("Groq returned an empty chunk summary.")
        return content.strip()

    async def _final_summary(
        self,
        text: str,
        extracted: ExtractedContent,
        is_from_chunk_summaries: bool = False,
    ) -> list[str]:
        stage_hint = (
            "The input already contains condensed chunk summaries from a longer transcript/article. "
            if is_from_chunk_summaries
            else ""
        )

        prompt = (
            f"{stage_hint}Return ONLY a valid JSON object with this schema: "
            '{"language":"<detected-language>","bullets":["point 1","point 2","point 3"]}. '
            "Rules: "
            "1) bullets must be 3 to 5 items; "
            "2) write bullets in the detected language of the content; "
            "3) each bullet must be concise, factual, and high-signal; "
            "4) preserve key names, claims, events, and numbers; "
            "5) no markdown, no extra keys, no commentary.\n\n"
            f"Title: {extracted.title}\n"
            f"Source: {extracted.source}\n"
            f"Length: {extracted.length}\n\n"
            f"Content:\n{text}"
        )

        try:
            response = await self.client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise summarizer. Output must be strict JSON only. "
                            "Do not invent facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            raise SummarizationError(f"Groq final summarization request failed: {exc}") from exc

        raw_content = response.choices[0].message.content
        if not raw_content:
            raise SummarizationError("Groq returned an empty final summary.")

        return parse_bullets_from_json(raw_content)


def chunk_text(text: str, chunk_size: int) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            sentence_break = normalized.rfind(". ", start, end)
            if sentence_break > start + int(chunk_size * 0.6):
                end = sentence_break + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


def parse_bullets_from_json(raw_content: str) -> list[str]:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        # Fallback: try to recover JSON if the model wraps it in text.
        match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
        if not match:
            raise SummarizationError("Failed to parse JSON returned by Groq.")
        payload = json.loads(match.group(0))

    bullets = payload.get("bullets")
    if not isinstance(bullets, list) or not bullets:
        raise SummarizationError("Groq response JSON did not contain a valid bullets list.")

    cleaned = [str(item).strip(" -•\n\t") for item in bullets if str(item).strip()]
    if not cleaned:
        raise SummarizationError("Groq bullets list was empty after cleaning.")

    return cleaned[:5]
