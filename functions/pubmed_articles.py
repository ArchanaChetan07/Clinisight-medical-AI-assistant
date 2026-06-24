"""
pubmed_articles.py
Fetches structured PubMed article metadata using NCBI E-utilities.
Includes retry logic, proper timeout handling, and clean structured output.
"""

import time
import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {
    "User-Agent": "Clinisight/1.0 (clinical-decision-support; contact@clinisight.ai)"
}


def _get_text(tag, separator=" ") -> str:
    return tag.get_text(separator=separator, strip=True) if tag else ""


def _parse_date(date_tag) -> str:
    if not date_tag:
        return "Unknown"
    year = _get_text(date_tag.find("year"))
    month = _get_text(date_tag.find("month"))
    day = _get_text(date_tag.find("day"))
    parts = [p for p in [day, month, year] if p]
    return " ".join(parts) if parts else "Unknown"


def _parse_authors(article) -> list[str]:
    authors = []
    for author in article.find_all("author"):
        last = _get_text(author.find("lastname"))
        fore = _get_text(author.find("forename"))
        name = f"{fore} {last}".strip() if fore or last else None
        if name:
            authors.append(name)
    return authors or ["Unknown authors"]


def fetch_pubmed_articles_with_metadata(
    query: str,
    max_results: int = 3,
    retries: int = 2,
) -> list[dict]:
    """
    Search PubMed and return structured article metadata.

    Args:
        query: Search string (symptom keywords or clinical terms)
        max_results: Max articles to fetch (default 3)
        retries: Number of retry attempts on network failure

    Returns:
        List of article dicts with keys: title, abstract, authors,
        publication_date, journal, pmid, article_url, mesh_terms
    """
    for attempt in range(retries + 1):
        try:
            # Step 1: Search
            search_resp = requests.get(
                f"{NCBI_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                },
                headers=HEADERS,
                timeout=12,
            )
            search_resp.raise_for_status()
            id_list = search_resp.json()["esearchresult"]["idlist"]

            if not id_list:
                logger.info(f"No PubMed results for query: {query!r}")
                return []

            # Step 2: Fetch full records
            fetch_resp = requests.get(
                f"{NCBI_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "xml",
                },
                headers=HEADERS,
                timeout=15,
            )
            fetch_resp.raise_for_status()

            soup = BeautifulSoup(fetch_resp.text, "lxml-xml")
            articles = []

            for article_el, pmid in zip(soup.find_all("PubmedArticle"), id_list):
                medline = article_el.find("MedlineCitation")
                article_info = medline.find("Article") if medline else None

                title = _get_text(article_info.find("ArticleTitle")) if article_info else "No title"
                abstract = _get_text(article_info.find("Abstract")) if article_info else "No abstract available"
                authors = _parse_authors(article_info) if article_info else ["Unknown"]
                
                journal_tag = article_info.find("Journal") if article_info else None
                journal = _get_text(journal_tag.find("Title")) if journal_tag else "Unknown journal"
                
                pub_date = _parse_date(
                    (article_info.find("Journal") or article_el).find("PubDate")
                    if article_info else None
                )

                # MeSH terms for better context
                mesh_terms = [
                    _get_text(m.find("DescriptorName"))
                    for m in article_el.find_all("MeshHeading")
                    if m.find("DescriptorName")
                ][:8]

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": journal,
                    "publication_date": pub_date,
                    "mesh_terms": mesh_terms,
                    "article_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })

            logger.info(f"Fetched {len(articles)} PubMed articles for query: {query!r}")
            return articles

        except requests.RequestException as e:
            logger.warning(f"PubMed fetch attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                logger.error(f"All PubMed fetch attempts failed for query: {query!r}")
                return []

        except Exception as e:
            logger.error(f"Unexpected error in PubMed fetch: {e}")
            return []
