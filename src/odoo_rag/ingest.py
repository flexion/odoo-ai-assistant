"""Scrape Odoo 18 documentation from sources.json with recursive link following.

This module downloads web pages from odoo.com and extracts plain text.
Output: Raw text files saved to data/raw/ directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup  # HTML parser library


def fetch_page(url: str) -> str:
    """Download a single web page and extract text content.

    Process:
    1. Send HTTP request to URL
    2. Parse HTML using BeautifulSoup
    3. Remove navigation, scripts, and styling
    4. Extract only readable text

    Args:
        url: Web address to download (e.g., https://odoo.com/documentation/...)

    Returns:
        Plain text content from the page
    """
    # Set user agent to identify the scraper
    headers = {"User-Agent": "Odoo-RAG-Bot/1.0"}

    # Download the page (30 second timeout)
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()  # Raise error if download fails

    # Parse HTML into structured format
    soup = BeautifulSoup(response.text, "lxml")

    # Remove unwanted elements (navigation menus, JavaScript, CSS)
    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.decompose()  # Delete element from HTML tree

    # Extract all text, separated by newlines
    return soup.get_text(separator="\n", strip=True)


def fetch_pages_recursive(start_url: str, max_pages: int = 50) -> str:
    """Download multiple related pages by following links (recursive crawling).

    Starts at one page, finds all links, downloads linked pages, and repeats.
    Stays within the same documentation section (e.g., /developer/ or /applications/).

    Args:
        start_url: Starting web address (e.g., https://odoo.com/documentation/18.0/developer/...)
        max_pages: Stop after downloading this many pages (default: 50)

    Returns:
        Combined text from all downloaded pages, separated by URL markers
    """
    # Parse starting URL to extract domain and path
    parsed_start = urlparse(start_url)
    start_path = parsed_start.path.rstrip("/")  # Remove trailing slash

    # Queue of URLs to download (starts with just the first URL)
    urls_to_fetch = [start_url]

    # Track which URLs have been downloaded (prevents duplicates)
    fetched_urls = set()

    # Store text from all pages
    all_texts = []

    # Continue until queue is empty or max_pages reached
    while urls_to_fetch and len(fetched_urls) < max_pages:
        url = urls_to_fetch.pop(0)
        if url in fetched_urls:
            continue

        try:
            # Download the current page
            headers = {"User-Agent": "Odoo-RAG-Bot/1.0"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "lxml")

            # Extract links from this page to add to download queue
            if len(fetched_urls) < max_pages - 1:  # Leave room for current page
                for link in soup.find_all("a", href=True):
                    href = link["href"]

                    # Convert relative URLs to absolute (e.g., ../page.html → https://odoo.com/page.html)
                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)

                    # Filter 1: Skip non-HTTP links (mailto:, javascript:, etc.)
                    if parsed.scheme not in ("http", "https"):
                        continue

                    # Filter 2: Skip external domains (only odoo.com)
                    if parsed.netloc != parsed_start.netloc:
                        continue

                    # Filter 3: Skip duplicates
                    if full_url in fetched_urls or full_url in urls_to_fetch:
                        continue

                    # Filter 4: Stay within same documentation section
                    link_path = parsed.path.rstrip("/")

                    # Must be under /documentation/18.0/
                    if not link_path.startswith("/documentation/18.0/"):
                        continue

                    # Must share the same category (applications, developer, reference)
                    # Example: If starting at /documentation/18.0/applications/sales.html
                    #          Only follow links starting with /documentation/18.0/applications/
                    start_parts = start_path.split("/")
                    if len(start_parts) >= 4:
                        # Extract base path: /documentation/18.0/applications
                        base_path = "/".join(start_parts[:4])
                        if not link_path.startswith(base_path):
                            continue
                    else:
                        # For shorter paths, require same prefix
                        if not link_path.startswith(start_path):
                            continue

                    # Link passed all filters, add to queue
                    urls_to_fetch.append(full_url)

            # Clean HTML: remove navigation, scripts, styling
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Extract plain text
            text = soup.get_text(separator="\n", strip=True)
            if text:
                # Add URL marker and text to collection
                all_texts.append(f"\n\n=== URL: {url} ===\n\n{text}")

            # Mark as successfully downloaded
            fetched_urls.add(url)
            print(f"  Fetched: {url} ({len(text)} chars)")

        except Exception as e:
            # Log error and mark as fetched to avoid infinite retries
            print(f"  Error fetching {url}: {e}")
            fetched_urls.add(url)

    return "\n".join(all_texts)


def ingest_all(
    sources_path: str = "data/sources.json", output_dir: str = "data/raw"
) -> None:
    """Download all documentation sources and save as text files.

    Reads sources.json which contains a list of URLs to scrape.
    For each source, downloads pages and saves to data/raw/ as .txt files.

    Args:
        sources_path: Path to JSON config file (default: data/sources.json)
        output_dir: Directory to save text files (default: data/raw)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configuration: list of URLs to scrape
    with open(sources_path) as f:
        config = json.load(f)

    # Process each documentation source
    for source in config.get("sources", []):
        # Generate filename from source name (e.g., "Odoo ORM" → "odoo_orm.txt")
        name = source["name"].replace(" ", "_").lower()
        url = source["url"]
        max_pages = source.get("max_pages", 1)  # Default: download 1 page only

        print(f"Fetching: {source['name']} (max {max_pages} pages)")
        try:
            # Choose download method based on max_pages setting
            if max_pages > 1:
                # Recursive: follow links to download multiple pages
                text = fetch_pages_recursive(url, max_pages=max_pages)
            else:
                # Single page: download only the specified URL
                text = fetch_page(url)

            # Save to file: data/raw/odoo_orm.txt
            (output_path / f"{name}.txt").write_text(text)
            print(f"  Saved: {len(text)} chars total\n")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Entry point: Download all documentation sources.

    Workflow:
    1. Read sources.json (list of URLs)
    2. Download each URL (single page or recursive crawl)
    3. Extract plain text (remove HTML, navigation, etc.)
    4. Save to data/raw/*.txt

    Next step: Run indexer.py to build FAISS index
    """
    ingest_all()
    print("Done. Run `make generate-qa` next.")


if __name__ == "__main__":
    main()
