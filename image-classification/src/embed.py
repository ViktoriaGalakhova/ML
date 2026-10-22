import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from src.config import Config


def _subsample(embeddings: np.ndarray, labels: np.ndarray, max_points: int, seed: int):
    n = embeddings.shape[0]
    if n <= max_points:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return embeddings[idx], labels[idx]


def get_embeddings(model, loader, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts_dir = Path(Config.ARTIFACTS_DIR)
    model_path = artifacts_dir / f"{model_name}_best.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_emb = []
    all_lbl = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model.get_embedding(x)
            all_emb.append(emb.detach().cpu())
            all_lbl.append(y.detach().cpu())

    embeddings = torch.cat(all_emb, dim=0).numpy()
    labels = torch.cat(all_lbl, dim=0).numpy().astype(int)

    embeddings, labels = _subsample(
        embeddings,
        labels,
        max_points=int(getattr(Config, "EMBED_MAX_POINTS", 1500)),
        seed=int(getattr(Config, "SEED", 42)),
    )

    out_dir = Path(Config.ARTIFACTS_DIR)
    np.save(out_dir / f"{model_name}_embeddings.npy", embeddings)
    np.save(out_dir / f"{model_name}_labels.npy", labels)

    return embeddings, labels


def visualize_embeddings(embeddings, labels, model_name):
    tsne = TSNE(n_components=2, random_state=int(getattr(Config, "SEED", 42)))
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for cls, name in [(0, "dog"), (1, "duck")]:
        mask = labels == cls
        if mask.sum() == 0:
            continue
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=10, alpha=0.7, label=name)

    plt.title(f"{model_name} embeddings (t-SNE), n={len(labels)}")
    plt.legend()
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.3)

    artifacts_dir = Path(Config.ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(artifacts_dir / f"{model_name}_embeddings_2d.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"saved plot: {model_name}_embeddings_2d.png")


def save_embeddings_plot(model, test_loader, model_name):
    embeddings, labels = get_embeddings(model, test_loader, model_name)
    visualize_embeddings(embeddings, labels, model_name)
