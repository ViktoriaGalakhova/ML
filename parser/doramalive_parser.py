import argparse
import os
import re
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://50.doramalive.news"
SERIALS_URL = f"{BASE_URL}/dorama/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0 Safari/537.36",
}
SLEEP_BETWEEN_REQUESTS = 0.5
TIMEOUT = 15

DEFAULT_OUTPUT_DIR = "data"
HTML_SUBDIR = "html"
CSV_DEFAULT = "data/doramalive_dataset.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def slugify_url(url: str) -> str:
    p = urlparse(url)
    slug = p.path.strip('/').replace('/', '_')
    return f"{p.netloc}__{slug or 'root'}.html"


def save_html_file(output_dir: str, url: str, text: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fn = slugify_url(url)
    path = os.path.join(output_dir, fn)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def try_fetch(session: requests.Session, url: str) -> Optional[str]:
    try:
        resp = session.get(url, timeout=TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"Failed {url} ({resp.status_code})")
            return None
        return resp.text
    except requests.RequestException as e:
        logger.warning(f"Error {url}: {e}")
        return None


def get_drama_links_from_dorama(max_pages: int = 70) -> List[str]:
    session = new_session()
    links = set()

    EXCLUDE_PATHS = [
        'serials', 'mini', 'film', 'lacorns', 'tvshow', 'new', 'top', 'soon',
        'luchshie', 'collections', 'recommendation', 'calendar', 'last',
        'ongoing', 'waiting'
    ]

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/dorama/?PAGEN_1={page}"
        html = try_fetch(session, url)
        if not html:
            logger.warning(f"Page {page} could not be fetched")
            continue

        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if href.startswith('/dorama/') and href.count('/') == 3:
                if not any(href.startswith(f'/dorama/{p}/') for p in EXCLUDE_PATHS):
                    full_url = urljoin(BASE_URL, href)
                    links.add(full_url)

        logger.info(f"Page {page}: collected {len(links)} links so far")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return sorted(links)



def download_urls(urls: List[str], output_dir: str) -> List[str]:
    session = new_session()
    saved_files = []
    for url in tqdm(urls, desc="Downloading"):
        html = try_fetch(session, url)
        if html:
            path = save_html_file(output_dir, url, html)
            saved_files.append(path)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return saved_files


@dataclass
class Drama:
    id: str
    title: Optional[str] = None
    country: Optional[str] = None
    platform: Optional[str] = None
    premiere_year: Optional[int] = None
    duration_min: Optional[float] = None
    episodes_total: Optional[int] = None
    rating: Optional[float] = None
    rating_votes: Optional[int] = None
    recommend_percent: Optional[float] = None
    source_url: Optional[str] = None

    def to_record(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_soup(cls, soup: BeautifulSoup, source_url: str = None):
        def parse_int(text):
            if not text: return None
            m = re.search(r'(\d+)', text.replace('\xa0',''))
            return int(m.group(1)) if m else None

        def parse_float(text):
            if not text: return None
            m = re.search(r'(\d+[\.,]?\d*)', text)
            return float(m.group(1).replace(',', '.')) if m else None

        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else None

        text_all = soup.get_text(" ")

        country = None
        m = re.search(r'Страна:?\s*([А-Яа-яA-Za-z ]+)', text_all)
        if m: country = m.group(1).strip()

        platform = None
        m = re.search(r'Платформа:?\s*([А-Яа-яA-Za-z ]+)', text_all)
        if m: platform = m.group(1).strip()

        premiere_year = None
        m = re.search(r'(20\d{2}|19\d{2})', text_all, re.IGNORECASE)
        if m: premiere_year = int(m.group(1)) if m else None

        duration_min = None
        m = re.search(r'Продолжительность\s*(\d+)\s*мин', text_all, re.IGNORECASE)
        if m: duration_min = float(m.group(1))

        episodes_total = None
        m = re.search(r'(?:кол-во\s*серий[:\s]*)?(\d+)', text_all, re.IGNORECASE)
        if m: episodes_total = int(m.group(1))

        rating = None
        rating_votes = None
        m = re.search(r'Рейтинг\s*([0-9]+(?:[.,][0-9]+)?)\s+(\d+)\s*(?:оцен|голос)', text_all)
        if m:
            rating = float(m.group(1).replace(',', '.'))
            rating_votes = int(m.group(2))

        recommend_percent = None
        m = re.search(r'Рекомендуют\s*(\d+)%', text_all)
        if m: recommend_percent = float(m.group(1))

        return cls(
            id=source_url or '',
            title=title,
            country=country,
            platform=platform,
            premiere_year=premiere_year,
            duration_min=duration_min,
            episodes_total=episodes_total,
            rating=rating,
            rating_votes=rating_votes,
            recommend_percent = recommend_percent,
            source_url=source_url
        )


def parse_html_file(path: str) -> Optional[Drama]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'lxml')
        url = os.path.basename(path)
        return Drama.from_soup(soup, source_url=url)
    except Exception as e:
        logger.warning(f"Parse error {path}: {e}")
        return None


def run_crawl(args):
    links = get_drama_links_from_dorama(max_pages=args.max_pages)
    logger.info(f"Total drama links collected: {len(links)}")
    html_dir = os.path.join(args.out_dir, HTML_SUBDIR)
    download_urls(links, html_dir)


def run_parse(args):
    input_dir = os.path.join(args.out_dir, HTML_SUBDIR)
    out_csv = args.output_file or CSV_DEFAULT
    files = [os.path.join(input_dir, fn) for fn in os.listdir(input_dir) if fn.endswith(".html")]
    records = []
    for p in tqdm(files, desc="Parsing"):
        d = parse_html_file(p)
        if d:
            records.append(d.to_record())
    if not records:
        logger.warning("No records parsed")
        return
    df = pd.DataFrame.from_records(records)
    for col in ['rating', 'rating_votes', 'recommend_percent']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    int_columns = ['premiere_year', 'episodes_total', 'rating_votes', 'recommend_percent']
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype('Int64')
    if 'recommend_percent' in df.columns:
        df['recommend_percent'] = df['recommend_percent'].apply(
            lambda x: f"{int(x)}%" if pd.notna(x) else ""
        )
    if 'platform' in df.columns:
        df['platform'] = df['platform'].replace('', pd.NA).fillna('local TV')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    logger.info(f"Saved dataset {out_csv} ({len(df)} rows)")


def run_all(args):
    run_crawl(args)
    run_parse(args)


def build_argparser():
    p = argparse.ArgumentParser(description="Doramalive parser")
    p.add_argument("--mode", choices=["crawl", "parse", "all"], default="all")
    p.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-pages", type=int, default=70, help="Количество страниц с сериалами")
    p.add_argument("--output-file", default=CSV_DEFAULT)
    return p


def main():
    args = build_argparser().parse_args()
    if args.mode == "crawl":
        run_crawl(args)
    elif args.mode == "parse":
        run_parse(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
