"""
Async scraper primitives for DGSI.

Two layers:
  * listing fetch: given a view URL with Start=N, returns (doc_urls, next_url)
  * document fetch/parse: returns {doc_id, url, source_db, court, court_short, full_text}

Only the full text is captured; structured fields are produced downstream by the LLM.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiohttp
import chardet
from bs4 import BeautifulSoup

from scraper.config import (
    BASE_URL, MAX_RETRIES, REQUEST_TIMEOUT, RETRY_BACKOFF_BASE,
)

DATA_DIR = Path("data/raw")
STATE_FILE = Path("data/scraper_state.json")


# DGSI serves Lotus Notes pages as Windows-1252; decode defensively.
def decode_content(content: bytes) -> str:
    try:
        return content.decode("cp1252")
    except UnicodeDecodeError:
        enc = (chardet.detect(content).get("encoding") or "utf-8")
        return content.decode(enc, errors="replace")


_WS_RE = re.compile(r"[ \t]+")


def extract_full_text(soup: BeautifulSoup) -> str:
    """
    Return the full readable text of a DGSI document page — metadata labels
    (Processo, Relator, Descritores, Decisão, Sumário, …) + the decision body.
    The LLM downstream parses everything out of this blob.
    """
    for tag in soup(["script", "style", "noscript", "form", "button", "input"]):
        tag.decompose()

    body = soup.body or soup
    raw = body.get_text("\n", strip=False)

    lines: list[str] = []
    for line in raw.split("\n"):
        line = _WS_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_document(html: str, url: str, db_config: dict) -> Optional[dict]:
    """Return a minimal record with the full page text."""
    soup = BeautifulSoup(html, "lxml")
    full_text = extract_full_text(soup)
    if len(full_text) < 200:
        return None

    doc_id = url.rstrip("/").split("/")[-1].split("?")[0]
    return {
        "doc_id": doc_id,
        "url": url,
        "source_db": db_config["db"],
        "court": db_config["label"],
        "court_short": db_config["short"],
        "full_text": full_text,
    }


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = MAX_RETRIES,
) -> Optional[bytes]:
    """GET with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                if resp.status in (429, 500, 502, 503, 504):
                    await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)
                    continue
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)
    return None


async def get_doc_links_from_page(
    session: aiohttp.ClientSession,
    url: str,
    db_config: dict,
) -> tuple[list[str], Optional[str]]:
    """Return (doc urls on listing page, next-listing-page url)."""
    content = await fetch_with_retry(session, url)
    if not content:
        return [], None
    soup = BeautifulSoup(decode_content(content), "lxml")

    doc_urls: list[str] = []
    next_url: Optional[str] = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "OpenDocument" in href:
            doc_urls.append(urljoin(BASE_URL + "/", href))
        elif "Start=" in href and a.get_text(strip=True).lower() in ("seguinte", "próximo", "proximo"):
            next_url = urljoin(BASE_URL + "/", href)
    return doc_urls, next_url
