import pandas as pd
from pathlib import Path
from src.collecting.config import Config
import math

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1




def split_stratified(df):
    if "label_int" not in df.columns:
        raise ValueError("Missing required column 'label_int'.")

    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

    train_parts = []
    val_parts = []
    test_parts = []

    for cls, part in df.groupby("label_int", sort=True):
        part = part.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n = len(part)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
            n_test = n - n_train - n_val

            if n_test <= 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val = max(1, n_val - 1)

            n_test = n - n_train - n_val

        if n_train < 0 or n_val < 0 or n_test < 0 or (n_train + n_val + n_test != n):
            raise ValueError(
                f"Invalid split sizes for class {cls}: "
                f"n={n}, train={n_train}, val={n_val}, test={n_test}"
            )

        train_parts.append(part.iloc[:n_train])
        val_parts.append(part.iloc[n_train:n_train + n_val])
        test_parts.append(part.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return train_df, val_df, test_df



def main():
    if not Config.LABELS_CSV.exists():
        raise FileNotFoundError(f"Not found: {Config.LABELS_CSV}")

    df = pd.read_csv(Config.LABELS_CSV)

    if "path" in df.columns:
        missing = df[~df["path"].astype(str).apply(lambda p: Path(p).exists())]
        if len(missing) > 0:
            df = df.drop(missing.index).reset_index(drop=True)

    train_df, val_df, test_df = split_stratified(df)

    Config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Config.SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(Config.SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(Config.SPLITS_DIR / "test.csv", index=False)

    print("saved splits")


if __name__ == "__main__":
    main()
