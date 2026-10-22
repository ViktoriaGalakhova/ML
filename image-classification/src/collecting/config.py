from pathlib import Path


class Config:
    LABELS_CSV = Path("data/labels.csv")
    SPLITS_DIR = Path("data/splits")
    IMAGES_DIR = Path("data/images")
    LABELS_CSV = Path("data/labels.csv")
    DOG_PREFIX = "dog_"
    DUCK_PREFIX = "duck_"
    DOG_API = "https://dog.ceo/api/breeds/image/random/50"
    DUCK_API = "https://random-d.uk/api/v2/randomimg"
    TARGET_DOGS = 1472
    TARGET_DUCKS = 1472
    API_TIMEOUT = 10
    IMG_TIMEOUT = 25
    WORKERS = 6
    RETRIES = 5
    BACKOFF_SECONDS = (1, 2, 4, 8, 16)
    MIN_BYTES = 10_000
