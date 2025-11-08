import os
import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.utils.helpers import load_config, set_seed, ensure_dir
from src.utils.dataset import ImageFolderDataset
from src.features.extract_features import extract_features
from src.models.classical import train_and_eval_classical
from src.models.transfer_learning import train_transfer_model
from src.evaluation.evaluate import summarize_results


def run():
    # Charger la configuration
    cfg = load_config()
    set_seed(cfg.get("seed", 42))

    # Charger le dataset brut (images PIL, sans transform)
    dataset = ImageFolderDataset(cfg["paths"]["data_raw"], transform=None)
    num_classes = len(dataset.class_to_idx)
    print(f"Detected classes: {num_classes} -> {dataset.class_to_idx}")

    # === Entraînement des modèles classiques sur les trois backbones ===
    all_results = {}
    for backbone in cfg["features"]["backbones"]:
        print(f"\nExtracting features with {backbone} ...")
        X, y = extract_features(backbone, dataset, batch_size=cfg["features"]["batch_size"])

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg["preprocessing"]["test_split"],
            stratify=y,
            random_state=cfg["seed"]
        )

        # Entraîner et évaluer les modèles classiques
        results = train_and_eval_classical(X_train, y_train, X_test, y_test, cfg)
        all_results[backbone] = results

        # Sauvegarder les modèles classiques
        out_dir = os.path.join(cfg["paths"]["models_classical"], backbone)
        ensure_dir(out_dir)
        for name, r in results.items():
            joblib.dump(r["model"], os.path.join(out_dir, f"{name}.joblib"))

    # Résumé comparatif
    summarize_results(all_results)

    # === Entraînement du CNN en transfer learning ===
    print("\nTraining transfer learning CNN ...")
    cnn_model = train_transfer_model(
        dataset,
        num_classes=num_classes,
        epochs=cfg["transfer"]["epochs"],
        batch_size=cfg["transfer"]["batch_size"],
        lr=cfg["transfer"]["lr"],
        backbone=cfg["transfer"]["backbone"]
    )

    # Sauvegarder le CNN
    ensure_dir(cfg["paths"]["models_transfer"])
    torch_path = os.path.join(
        cfg["paths"]["models_transfer"],
        f"{cfg['transfer']['backbone']}_{num_classes}classes.pth"
    )
    torch.save(cnn_model.state_dict(), torch_path)
    print(f"✅ Saved transfer model to: {torch_path}")
