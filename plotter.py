import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import norm

plt.rcParams.update({
    'font.size': 15,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth': 2,
    'lines.linewidth': 3
})


class Plotter:
    def __init__(self, print_dir='', end_name=''):
        self.print_dir = print_dir
        self.end_name = end_name

    def plotTrainLoss(self, tracker):
        train_losses = tracker.train_losses
        val_losses = tracker.val_losses

        plt.figure(figsize=(20, 20))
        plt.plot(train_losses, label='Train', color='royalblue')
        plt.plot(val_losses, label='Test', color='firebrick')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.print_dir}/loss_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    def plotLabelProbsPerTrack(self, preds, targets, bins=100):
        """
        分别绘制每条 track 的预测概率分布，正负样本分开显示。

        preds: (N, num_tracks) tensor, sigmoid 概率
        targets: (N, num_tracks) tensor, 0/1
        """
        num_tracks = preds.size(1)
        for track_idx in range(num_tracks):
            probs = preds[:, track_idx].cpu().numpy()
            labels = targets[:, track_idx].cpu().numpy()

            probs_1 = probs[labels == 1]
            probs_0 = probs[labels == 0]

            plt.figure(figsize=(10, 6))
            plt.hist(probs_1, bins=bins, alpha=0.6, color='green', label='Label=1')
            plt.hist(probs_0, bins=bins, alpha=0.6, color='red', label='Label=0')
            plt.xlabel("Predicted probability")
            plt.ylabel("Count")
            plt.title(f"Track {track_idx+1} probability distribution")
            plt.legend()
            plt.tight_layout()
            outname = f"{self.print_dir}/probs_track{track_idx+1}_{self.end_name}.png"
            plt.savefig(outname, dpi=300)
            plt.close()

    def plotTPR_TNR_vs_Threshold(self, preds, targets, num_points=100):
        """
        分别绘制每条 track 的 TPR 和 TNR 随概率阈值变化曲线。

        preds: (N, num_tracks) tensor, sigmoid 概率
        targets: (N, num_tracks) tensor, 0/1
        num_points: 阈值采样点数
        """
        num_tracks = preds.size(1)
        thresholds = np.linspace(0, 1, num_points)

        for track_idx in range(num_tracks):
            probs = preds[:, track_idx].cpu().numpy()
            labels = targets[:, track_idx].cpu().numpy()

            tpr_list = []
            tnr_list = []

            for th in thresholds:
                pred_labels = (probs >= th).astype(int)
                TP = ((pred_labels == 1) & (labels == 1)).sum()
                TN = ((pred_labels == 0) & (labels == 0)).sum()
                FP = ((pred_labels == 1) & (labels == 0)).sum()
                FN = ((pred_labels == 0) & (labels == 1)).sum()

                tpr = TP / (TP + FN + 1e-8)
                tnr = TN / (TN + FP + 1e-8)

                tpr_list.append(tpr)
                tnr_list.append(tnr)

            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, tpr_list, label='TPR', color='green', linewidth=2)
            plt.plot(thresholds, tnr_list, label='TNR', color='red', linewidth=2)
            plt.xlabel("Threshold")
            plt.ylabel("Rate")
            plt.title(f"Track {track_idx + 1} TPR/TNR vs Threshold")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            outname = f"{self.print_dir}/tpr_tnr_track{track_idx + 1}_{self.end_name}.png"
            plt.savefig(outname, dpi=300)
            plt.close()

    def plotPrecisionRecallF1_vs_Threshold(self, preds, targets, num_points=100):
        """
        分别绘制每条 track 的 Precision、Recall、F1-score 随概率阈值变化曲线。
        返回每条 track 的最佳阈值（F1-score 最大时）。

        preds: (N, num_tracks) tensor, sigmoid 概率
        targets: (N, num_tracks) tensor, 0/1
        num_points: 阈值采样点数
        """
        num_tracks = preds.size(1)
        thresholds = np.linspace(0, 1, num_points)
        best_thresholds = []

        for track_idx in range(num_tracks):
            probs = preds[:, track_idx].cpu().numpy()
            labels = targets[:, track_idx].cpu().numpy()

            precision_list = []
            recall_list = []
            f1_list = []

            for th in thresholds:
                pred_labels = (probs >= th).astype(int)
                precision = precision_score(labels, pred_labels, zero_division=0)
                recall = recall_score(labels, pred_labels, zero_division=0)
                f1 = f1_score(labels, pred_labels, zero_division=0)

                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            precision_list = np.array(precision_list)
            recall_list = np.array(recall_list)
            f1_list = np.array(f1_list)

            best_idx = f1_list.argmax()
            best_threshold = thresholds[best_idx]
            best_thresholds.append(best_threshold)

            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precision_list, label='Precision', color='blue', linewidth=2)
            plt.plot(thresholds, recall_list, label='Recall', color='orange', linewidth=2)
            plt.plot(thresholds, f1_list, label='F1-score', color='green', linewidth=2)
            plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold={best_threshold:.3f}')
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title(f"Track {track_idx + 1} Precision/Recall/F1 vs Threshold")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            outname = f"{self.print_dir}/prf_track{track_idx + 1}_{self.end_name}.png"
            plt.savefig(outname, dpi=300)
            plt.close()

        return best_thresholds
