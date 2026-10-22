from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.data import get_data_loaders
from src.tl.model_tl import ResNetTransfer
from src.train import run_training
from src.train import test_model
from src.embed import save_embeddings_plot


def main():
    Path(Config.ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders()

    model = ResNetTransfer(
        hidden_dim=Config.TL_HIDDEN_DIM,
        dropout=Config.DROPOUT,
        embedding_dim=Config.EMBEDDING_DIM,
    )

    writer = SummaryWriter(log_dir=str(Path(Config.ARTIFACTS_DIR) / "tb"))
    info = run_training(model, train_loader, val_loader, model_name="tl", writer=writer)
    writer.close()

    print("\nTraining summary:", info)

    test_model(model, test_loader, model_name="tl")
    save_embeddings_plot(model, test_loader, model_name="tl")


if __name__ == "__main__":
    main()
