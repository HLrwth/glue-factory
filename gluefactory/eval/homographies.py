from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..models.utils.metrics import matcher_metrics
from ..settings import EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..utils.tensor import map_tensor, batch_to_device
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args

from ..utils.tools import (
    AverageMetric,
    MedianMetric,
)

class HomographiesValPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "homographies",
            "data_dir": "revisitop1m",
            "train_size": 150000,
            "val_size": 2000,
            "batch_size": 1,
            "num_workers": 1,
            "homography": {
                "difficulty": 0.7,
                "max_angle": 45
            },
            "photometric": {
                "name": "lg"
            },
        },
        "model": {
            "name": "two_view_pipeline",
            "extractor": {
                "name": "gluefactory_nonfree.superpoint",
                "max_num_keypoints": 512,
                "force_num_keypoints": True,
                "detection_threshold": 0.0,
                "nms_radius": 3,
                "trainable": False
            },
            "ground_truth": {
                "name": "matchers.homography_matcher",
                "th_positive": 3,
                "th_negative": 3
            },
            "matcher": {
                "name": "matchers.simpleglue",
                "flash": False,
                "checkpointed": False
            },
            "run_gt_in_forward": True,
        },
        "eval": {
            "filter_threshold": [0, 0.1, 0.2],  # visual matching threshold
            "logvar_filter_threshold": [-1, -0.5, 0, 0.5, 1, 1.5, 2, 3],  # uncertainty threshold
        },
    }
    export_keys = [
        "p_rp_01",
        "p_rp_10",
        "logvar_01",
        "logvar_10",
        "gt_matches0",
        "gt_matches1"
    ]
    optional_export_keys = []

    def _init(self, conf):
        pass

    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset("homographies")(data_conf)  # with default seed 0
        return dataset.get_data_loader("val")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file
    
    def compute_matches(self, pred):
        p_rp_01 = pred["p_rp_01"]
        p_rp_10 = pred["p_rp_10"]
        scores = p_rp_01 * p_rp_10
        max0, max1 = scores.max(2), scores.max(1)
        m0, m1 = max0.indices, max1.indices
        indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
        indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
        mutual0 = indices0 == m1.gather(1, m0)
        mutual1 = indices1 == m0.gather(1, m1)
        max0_exp = max0.values
        zero = max0_exp.new_tensor(0)
        mscores0 = torch.where(mutual0, max0_exp, zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)

        logvar_01 = pred["logvar_01"]
        logvar_10 = pred["logvar_10"]

        logvar_01_value = logvar_01.max(-1).values
        logvar_10_value = logvar_10.max(-1).values

        return m0, mutual0, mscores0, logvar_01_value, m1, mutual1, mscores1, logvar_10_value
    
    def filter_matches(self, m0, mutual0, mscores0, lv_01_value, m1, mutual1, mscores1, lv_10_value, th, lv_th):
        valid0 = mutual0 & (mscores0 > th) & (lv_01_value < lv_th) & (lv_10_value.gather(1, m0) < lv_th)
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, -1)
        m1 = torch.where(valid1, m1, -1)
        pred_filtered = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
        }
        return pred_filtered


    def run_eval(self, loader, pred_file):
        assert pred_file.exists()
        results = {}

        conf = self.conf.eval

        score_th = (
            ([conf.filter_threshold] if conf.filter_threshold > 0 else [0, 0.1, 0.2])
            if not isinstance(conf.filter_threshold, Iterable)
            else conf.filter_threshold
        )
        logvar_th = (
            ([conf.logvar_filter_threshold] if conf.logvar_filter_threshold > 0 else [-2, -1, -0.5, 0, 0.5, 1, 2])
            if not isinstance(conf.logvar_filter_threshold, Iterable)
            else conf.logvar_filter_threshold
        )

        metrics_keys = ['match_recall', 'match_precision', 'accuracy']
        for th in score_th:
            results[th] = {}
            for lv_th in logvar_th:
                results[th][lv_th] = {k: [] for k in metrics_keys}
        results['names'] = []

        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i, data in enumerate(tqdm(loader)):
            data = batch_to_device(data, device, non_blocking=True)
            pred = cache_loader(data)
            # Add batch dimension
            pred = map_tensor(pred, lambda t: torch.unsqueeze(t, dim=0))

            m0, mutual0, mscores0, lv_01_value, m1, mutual1, mscores1, lv_10_value = self.compute_matches(pred)

            for th in score_th:
                for lv_th in logvar_th: 
                    pred_filtered = self.filter_matches(m0, mutual0, mscores0, lv_01_value, m1, mutual1, mscores1, lv_10_value, th, lv_th)
                    metrics = matcher_metrics(pred_filtered, {**pred, **data})
                    for k in metrics_keys:
                        results[th][lv_th][k].append(metrics[k])
                    
            # we also store the names for later reference
            # results_i["names"] = data["name"][0]
            results["names"].append(data["name"][0])

            
        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for th in score_th:
            summaries[th] = {}
            for lv_th in logvar_th:
                summaries[th][lv_th] = {}
                for m in metrics_keys:
                    summaries[th][lv_th][m] = {
                        "mean": AverageMetric(),
                        "median": MedianMetric(),
                    }

        # compute average and median of results
        for th in score_th:
            for lv_th in logvar_th:
                for m in metrics_keys:
                    for v in results[th][lv_th][m]:
                        summaries[th][lv_th][m]["mean"].update(v)
                        summaries[th][lv_th][m]["median"].update(v)

        for th in score_th:
            for lv_th in logvar_th:
                for m in metrics_keys:
                    summaries[th][lv_th][m]["mean"] = summaries[th][lv_th][m]["mean"].compute()
                    summaries[th][lv_th][m]["median"] = summaries[th][lv_th][m]["median"].compute()

        # plot pr cures for each threshold
        figures = {}
        
        for th in score_th:
            precision_i_mean = [summaries[th][lv_th]['match_precision']['mean'] for lv_th in logvar_th]
            recall_i_mean = [summaries[th][lv_th]['match_recall']['mean'] for lv_th in logvar_th]

            precision_i_median = [summaries[th][lv_th]['match_precision']['median'] for lv_th in logvar_th]
            recall_i_median = [summaries[th][lv_th]['match_recall']['median'] for lv_th in logvar_th]

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].plot(recall_i_mean, precision_i_mean)
            axs[0].scatter(recall_i_mean, precision_i_mean)
            for i, lv_th in enumerate(logvar_th):
                axs[0].annotate(lv_th, (recall_i_mean[i], precision_i_mean[i]))
            axs[0].set_xlabel("recall")
            axs[0].set_ylabel("precision")
            axs[0].set_title(f'Mean precision-recall curve for threshold {th}')

            axs[1].plot(recall_i_median, precision_i_median, label="median")
            axs[1].scatter(recall_i_median, precision_i_median)
            for i, lv_th in enumerate(logvar_th):
                axs[1].annotate(lv_th, (recall_i_median[i], precision_i_median[i]))
            axs[1].set_xlabel("recall")
            axs[1].set_ylabel("precision")
            axs[1].set_title(f'Median precision-recall curve for threshold {th}')
            
            # tight layout
            plt.tight_layout()

            # store figure to pred_file location
            figures[f'pr_curve_{th}'] = fig

        # convert summaries and results to dict of {k, numbers}
        results_flat = {}
        summaries_flat = {}
        for th in score_th:
            for lv_th in logvar_th:
                for m in metrics_keys:
                    summaries_flat[f'{th}_{lv_th}_{m}_mean'] = summaries[th][lv_th][m]['mean']
                    summaries_flat[f'{th}_{lv_th}_{m}_median'] = summaries[th][lv_th][m]['median']

                    results_flat[f'{th}_{lv_th}_{m}'] = torch.stack(results[th][lv_th][m], dim = 0).cpu().numpy()

        return summaries_flat, figures, summaries_flat


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(HomographiesValPipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = HomographiesValPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
