from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from extractor import ExtractedContent, ExtractionError, extract_content
from summarizer import GroqSummarizer, SummarizationError


load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TELEGRAM_MESSAGE_LIMIT = 4096

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    message = (
        "Send one URL from YouTube, TikTok, Instagram, Twitter/X, VK, or a normal article page.\n\n"
        "I will extract the content, transcribe audio/video when needed, and return a concise summary."
    )
    await update.message.reply_text(message)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    message = (
        "Supported inputs:\n"
        "• YouTube\n"
        "• TikTok\n"
        "• Instagram posts/reels\n"
        "• Twitter/X\n"
        "• VK\n"
        "• Generic article URLs\n\n"
        "Send one public URL per message."
    )
    await update.message.reply_text(message)


async def url_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    urls = extract_urls(update.message.text)
    if not urls:
        await update.message.reply_text(
            "I could not find a valid URL in your message. Send one public link to summarize."
        )
        return

    if len(urls) > 1:
        await update.message.reply_text("Send one URL at a time. Multiple links in one message are not supported in this MVP.")
        return

    url = urls[0]
    processing_message = await update.message.reply_text("⏳ Processing...")

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        await processing_message.edit_text("⏳ Processing... extracting content")
        extracted = await asyncio.to_thread(extract_content, url)

        await processing_message.edit_text("⏳ Processing... summarizing content")
        summarizer = get_summarizer(context)
        bullets = await summarizer.summarize(extracted)

        response_text = build_telegram_response(extracted, bullets)
        await processing_message.edit_text(
            response_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except ExtractionError as exc:
        logger.warning("Extraction error for %s: %s", url, exc)
        await processing_message.edit_text(f"❌ Could not process this URL.\n\nReason: {exc}")
    except SummarizationError as exc:
        logger.warning("Summarization error for %s: %s", url, exc)
        await processing_message.edit_text(f"❌ Content was extracted, but summarization failed.\n\nReason: {exc}")
    except Exception as exc:  # pragma: no cover - last-resort guardrail
        logger.exception("Unhandled error while processing %s", url)
        await processing_message.edit_text(
            "❌ Unexpected internal error while processing the link. "
            "Try another public URL or check your environment logs."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Telegram update caused an error", exc_info=context.error)


def extract_urls(text: str) -> list[str]:
    return [match.group(0).rstrip('.,);]') for match in URL_RE.finditer(text)]


def get_summarizer(context: ContextTypes.DEFAULT_TYPE) -> GroqSummarizer:
    summarizer = context.application.bot_data.get("summarizer")
    if summarizer is None:
        summarizer = GroqSummarizer(api_key=GROQ_API_KEY)
        context.application.bot_data["summarizer"] = summarizer
    return summarizer


def build_telegram_response(extracted: ExtractedContent, bullets: list[str]) -> str:
    safe_title = html.escape(truncate_field(extracted.title, 300))
    safe_source = html.escape(truncate_field(extracted.source, 100))
    safe_length = html.escape(truncate_field(extracted.length, 50))

    bullet_lines = [f"• {html.escape(truncate_field(bullet, 600))}" for bullet in bullets]
    summary_block = "\n".join(bullet_lines)

    prefix = (
        f"📌 <b>Title</b>: {safe_title}\n"
        f"🗂 <b>Source</b>: {safe_source}\n"
        f"⏱ <b>Length</b>: {safe_length}\n\n"
        f"<b>Summary:</b>\n"
    )

    message = prefix + summary_block
    if len(message) <= TELEGRAM_MESSAGE_LIMIT:
        return message

    suffix = "…[truncated]"
    allowed = TELEGRAM_MESSAGE_LIMIT - len(prefix) - len(suffix)
    truncated_summary = summary_block[: max(0, allowed)] + suffix
    return prefix + truncated_summary


def truncate_field(value: Optional[str], max_length: int) -> str:
    if not value:
        return "Unknown"
    value = value.strip()
    if len(value) <= max_length:
        return value
    return value[: max_length - 1].rstrip() + "…"


def validate_environment() -> None:
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def build_application() -> Application:
    validate_environment()

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    application.add_handler(CommandHandler("start", start_handler, block=False))
    application.add_handler(CommandHandler("help", help_handler, block=False))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, url_handler, block=False))
    application.add_error_handler(error_handler)
    return application


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = build_application()
    logger.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
