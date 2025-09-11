###-------(crawl4ai-only; robust MD + redirects)---###

import asyncio
from typing import Tuple, Dict, Any, List
from urllib.parse import urljoin, urldefrag, urlparse
from bs4 import BeautifulSoup
import httpx

from utils.logger import get_logger
from utils.config import PAGE_TIMEOUT_MS, CACHE_MODE, USER_AGENT, PRE_RESOLVE_REDIRECTS, STRICT_CRAWL4AI_ONLY
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

log = get_logger("extractor")

def _cache_mode():
    return getattr(CacheMode, CACHE_MODE) if isinstance(CACHE_MODE, str) else CacheMode.BYPASS

def resolve_redirect(url: str) -> str:
    if not PRE_RESOLVE_REDIRECTS:
        return url
    try:
        with httpx.Client(follow_redirects=True, timeout=20.0, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(url)
            return str(r.url)
    except Exception as e:
        log.warning(f"Redirect resolve failed for {url}: {e}")
        return url

def _basic_html_to_md(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    parts = []
    for tag in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li"]):
        t = tag.get_text(" ", strip=True)
        if not t: 
            continue
        if tag.name.startswith("h"):
            level = int(tag.name[1:])
            parts.append("#"*max(1,min(6,level)) + " " + t)
        elif tag.name == "li":
            parts.append(f"- {t}")
        else:
            parts.append(t)
    return "\n\n".join(parts).strip()

def _extract_links_from_html(html: str, base_url: str) -> List[str]:
    """Parse <a href> from page HTML; absolute-ize; drop fragments."""
    out: List[str] = []
    if not html:
        return out
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        u = urljoin(base_url, href)
        u, _ = urldefrag(u)
        out.append(u)
    return out

def _merge_links(base_url: str, crawl4ai_links: Dict[str, Any], html_links: List[str]) -> Dict[str, List[str]]:
    """Union of crawl4ai links and HTML-parsed links; bucket them into internal/external."""
    seen = set()
    merged_all = []
    # flatten crawl4ai links
    for k in ("internal", "external", "other"):
        v = crawl4ai_links.get(k) if isinstance(crawl4ai_links, dict) else None
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    merged_all.append(x)
    merged_all.extend(html_links)

    # de-dup, bucket
    internal, external = [], []
    base_host = urlparse(base_url).netloc.lower()
    for u in merged_all:
        if u in seen:
            continue
        seen.add(u)
        host = urlparse(u).netloc.lower()
        if host == base_host:
            internal.append(u)
        else:
            external.append(u)
    return {"internal": internal, "external": external, "other": []}

async def _crawl4ai(final_url: str) -> Tuple[str, Dict[str, Any]]:
    browser_conf = BrowserConfig(headless=True, user_agent=USER_AGENT, ignore_https_errors=True)
    run_conf = CrawlerRunConfig(
        cache_mode=_cache_mode(),
        page_timeout=PAGE_TIMEOUT_MS,
        markdown_generator=DefaultMarkdownGenerator(),  # no aggressive pruning
    )
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(url=final_url, config=run_conf)

    if not getattr(result, "success", True):
        raise RuntimeError(f"Crawl failed for {final_url}: {getattr(result, 'error_message', 'unknown error')}")

    # Markdown (robust across versions)
    markdown = ""
    md_block = getattr(result, "markdown", None)
    if isinstance(md_block, str):
        markdown = md_block
    elif md_block is not None:
        markdown = getattr(md_block, "fit_markdown", None) or getattr(md_block, "raw_markdown", None) or ""
    if not markdown.strip():
        cleaned_html = getattr(result, "cleaned_html", None)
        if cleaned_html:
            markdown = _basic_html_to_md(cleaned_html)

    # Links: merge crawl4ai's links with those parsed from HTML
    cleaned_html = getattr(result, "cleaned_html", None)
    html_links = _extract_links_from_html(cleaned_html, final_url)
    c4_links = getattr(result, "links", {}) or {}
    links = _merge_links(final_url, c4_links, html_links)

    meta = {
        "url": final_url,
        "links": links,
        "title": (getattr(result, "metadata", {}) or {}).get("title"),
    }
    return (markdown or ""), meta

async def _fetch_once(url: str) -> Tuple[str, Dict[str, Any], str]:
    final_url = resolve_redirect(url)
    md, meta = await _crawl4ai(final_url)
    if STRICT_CRAWL4AI_ONLY and not md.strip():
        raise RuntimeError(f"Crawl4AI returned empty content for {final_url}")
    return md, meta, final_url

def fetch_markdown(url: str) -> Tuple[str, Dict[str, Any], str]:
    return asyncio.run(_fetch_once(url))
