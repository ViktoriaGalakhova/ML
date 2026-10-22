import csv
import hashlib
import time
import uuid
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

from src.collecting.config import Config


def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def valid_image_bytes(b):
    if len(b) < Config.MIN_BYTES:
        return False
    try:
        with Image.open(BytesIO(b)) as im:
            im.convert("RGB")
        return True
    except Exception:
        return False


def sleep_backoff(attempt):
    time.sleep(Config.BACKOFF_SECONDS[min(attempt, len(Config.BACKOFF_SECONDS) - 1)])


def atomic_write(out, data):
    out = Path(out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        if tmp.stat().st_size <= 0:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(out)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def list_by_prefix(prefix):
    if not Config.IMAGES_DIR.exists():
        return []
    return [p for p in Config.IMAGES_DIR.iterdir() if p.is_file() and p.name.startswith(prefix)]


def fetch_dog_urls(s):
    for attempt in range(Config.RETRIES):
        try:
            r = s.get(Config.DOG_API, timeout=Config.API_TIMEOUT)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    time.sleep(int(ra))
                else:
                    sleep_backoff(attempt)
                continue
            if 500 <= r.status_code < 600:
                sleep_backoff(attempt)
                continue
            r.raise_for_status()
            j = r.json()
            urls = j.get("message", [])
            return urls if isinstance(urls, list) else []
        except Exception:
            sleep_backoff(attempt)
    return []


def download_url_to_jpg(s, url, out):
    out = Path(out)
    if out.exists() and out.stat().st_size > 0:
        return True

    for attempt in range(Config.RETRIES):
        try:
            r = s.get(url, timeout=Config.IMG_TIMEOUT, stream=True)
            code = r.status_code

            if code == 429:
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    time.sleep(int(ra))
                else:
                    sleep_backoff(attempt)
                continue

            if 500 <= code < 600:
                sleep_backoff(attempt)
                continue

            if code != 200:
                return False

            ctype = (r.headers.get("Content-Type") or "").lower()
            if not ctype.startswith("image/"):
                return False

            data = r.content
            if not valid_image_bytes(data):
                return False

            return atomic_write(out, data)
        except Exception:
            sleep_backoff(attempt)

    return False


def download_dog(s, url):
    name = Config.DOG_PREFIX + hashlib.sha1(url.encode("utf-8")).hexdigest() + ".jpg"
    out = Config.IMAGES_DIR / name
    return download_url_to_jpg(s, url, out)


def download_duck(s):
    name = Config.DUCK_PREFIX + uuid.uuid4().hex + ".jpg"
    out = Config.IMAGES_DIR / name
    return download_url_to_jpg(s, Config.DUCK_API, out)


def rebuild_labels_csv():
    dogs = sorted(list_by_prefix(Config.DOG_PREFIX), key=lambda p: p.name)
    ducks = sorted(list_by_prefix(Config.DUCK_PREFIX), key=lambda p: p.name)

    rows = []
    for p in dogs:
        rows.append({"id": p.stem, "path": "data/images/" + p.name, "label_int": 0})
    for p in ducks:
        rows.append({"id": p.stem, "path": "data/images/" + p.name, "label_int": 1})

    mkdir(Config.LABELS_CSV.parent)
    with open(Config.LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "path", "label_int"])
        w.writeheader()
        w.writerows(rows)


def collect_dogs(s, target):
    while True:
        have = len(list_by_prefix(Config.DOG_PREFIX))
        if have >= target:
            return

        urls = fetch_dog_urls(s)
        if not urls:
            time.sleep(1)
            continue

        need = target - have
        urls = urls[:need]

        ok = 0
        with ThreadPoolExecutor(max_workers=Config.WORKERS) as pool:
            futures = [pool.submit(download_dog, s, u) for u in urls]
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1

        have2 = len(list_by_prefix(Config.DOG_PREFIX))
        print("dogs", have2, "/", target)

        if ok == 0:
            time.sleep(1)


def collect_ducks(s, target):
    have = len(list_by_prefix(Config.DUCK_PREFIX))
    while have < target:
        need = target - have
        batch = min(need, Config.WORKERS * 6)

        ok = 0
        with ThreadPoolExecutor(max_workers=Config.WORKERS) as pool:
            futures = [pool.submit(download_duck, s) for _ in range(batch)]
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1

        have += ok
        print("ducks", have, "/", target)

        if ok == 0:
            time.sleep(2)


def main():
    mkdir(Config.IMAGES_DIR)

    s = requests.Session()

    print("collect dogs")
    collect_dogs(s, Config.TARGET_DOGS)

    print("collect ducks")
    collect_ducks(s, Config.TARGET_DUCKS)

    rebuild_labels_csv()
    print("saved labels.csv")

    s.close()


if __name__ == "__main__":
    main()
