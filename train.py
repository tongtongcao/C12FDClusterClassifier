import argparse
import os
import time
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from trainer import *
from plotter import Plotter

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster Track Prediction Training")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("inputs", type=str, nargs="+", help="Input CSV files")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--end_name", type=str, default="")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--enable_progress_bar", action="store_true")
    return parser.parse_args()

def load_dataset(csv_files, num_tracks=2):
    df_list = []
    offset = 0
    for f in csv_files:
        df = pd.read_csv(f)
        df["eventIdx"] = df["eventIdx"] + offset
        df_list.append(df)
        offset += df["eventIdx"].max() + 1
    df_all = pd.concat(df_list, ignore_index=True)
    dataset = EventGraphDataset(df_all, num_tracks=num_tracks)
    # 排除全噪声事件
    dataset.data_list = [d for d in dataset.data_list if d.y.sum() > 0]
    return dataset

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------
    # Load dataset
    print("Loading dataset...")
    start_time = time.time()
    dataset = load_dataset(args.inputs, num_tracks=2)
    dataset_size = len(dataset)
    val_size = max(1, int(0.2 * dataset_size))
    train_size = dataset_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False)
    print(f"Train size: {train_size}, Val size: {val_size}")
    print(f"Data loading took {time.time() - start_time:.2f}s")

    # -------------------------
    # Initialize model
    model = ClusterGNN(
        in_channels=3,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_tracks=2,
        lr=args.lr,
        dropout=args.dropout
    )
    loss_tracker = LossTracker()

    plotter = Plotter(print_dir=args.outdir, end_name=args.end_name)

    # -------------------------
    # Train
    if not args.no_train:
        if args.device == "cpu":
            accelerator, devices = "cpu", 1
        elif args.device == "gpu":
            accelerator, devices = ("gpu", 1) if torch.cuda.is_available() else ("cpu", 1)
        else:  # auto
            accelerator, devices = ("gpu", "auto") if torch.cuda.is_available() else ("cpu", 1)

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=args.max_epochs,
            enable_progress_bar=args.enable_progress_bar,
            logger=False,
            callbacks=[loss_tracker],
        )

        print("Start training...")
        t0 = time.time()
        trainer.fit(model, train_loader, val_loader)
        print(f"Training finished in {(time.time() - t0)/60:.2f} minutes")

        plotter.plotTrainLoss(loss_tracker)

        # Save TorchScript model
        model.to("cpu")
        wrapper = ClusterGNNWrapper(model)
        wrapper.eval()
        torchscript_model = torch.jit.script(wrapper)
        model_path = os.path.join(args.outdir, f"gnn_{args.end_name}.pt")
        torchscript_model.save(model_path)
        print(f"Saved model to {model_path}")

    else:
        model_path = os.path.join(args.outdir, f"gnn_{args.end_name}.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join("nets", "gnn_default.pt")

    # -------------------------
    # Inference
    print("Running inference...")
    model = torch.jit.load(model_path)
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    print(f"Predictions shape: {all_preds.shape}, Targets shape: {all_targets.shape}")

    plotter.plotLabelProbsPerTrack(all_preds, all_targets)

    plotter.plotTPR_TNR_vs_Threshold(all_preds, all_targets)

    plotter.plotPrecisionRecallF1_vs_Threshold(all_preds, all_targets)

    # -------------------------
    # Print sample cluster probabilities
    for i in range(min(10, all_preds.size(0))):
        print(f"Cluster {i}: Track1={all_preds[i,0]:.3f}, Track2={all_preds[i,1]:.3f}, Target={all_targets[i].tolist()}")

if __name__ == "__main__":
    main()
