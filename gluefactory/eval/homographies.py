import json
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..models.utils.metrics import matcher_metrics
from ..models.matchers import simpleglue
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
        #  "model": {
        #     "name": "two_view_pipeline",
        #     "extractor": {
        #         "name": "gluefactory_nonfree.superpoint",
        #         "max_num_keypoints": 512,
        #         "force_num_keypoints": True,
        #         "detection_threshold": 0.0,
        #         "nms_radius": 3,
        #         "trainable": False
        #     },
        #     "ground_truth": {
        #         "name": "matchers.homography_matcher",
        #         "th_positive": 3,
        #         "th_negative": 3
        #     },
        #     "matcher": {
        #         "name": "matchers.lightglue",
        #         "flash": False,
        #         "checkpointed": False,
        #         "weights": "superpoint_lightglue"
        #     },
        #     "run_gt_in_forward": True,
        # },
        "eval": {
            "model_name": "simpleglue", 
            "filter_threshold": [0, 0.1, 0.2],  # visual matching threshold
            "logvar_filter_threshold": [-1, -0.5, 0, 0.5, 1, 1.5, 2, 3],  # uncertainty threshold
        },
    }
    export_keys = [
        "gt_matches0",
        "gt_matches1"
        ]
    optional_export_keys = [        
        "p_rp_01",
        "p_rp_10",
        "logvar_01",
        "logvar_10",
        "log_assignment"
        ]

    @classmethod
    def get_dataloader(cls, data_conf=None):
        data_conf = data_conf if data_conf else cls.default_conf["data"]
        dataset = get_dataset("homographies")(data_conf)  # with default shuffle_seed 0 to get same val set
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
    
    def compute_matches_sg(self, pred):
        p_rp_01 = pred["p_rp_01"]
        p_rp_10 = pred["p_rp_10"]
        scores = p_rp_01 * p_rp_10
        logvar_01 = pred["logvar_01"]
        logvar_10 = pred["logvar_10"]

        return simpleglue.compute_matches(scores, logvar_01, logvar_10)

    def filter_matches_sg(self, m, mscores, mutual, logvar_value, score_th, logvar_th, pred):
        valid0 = mutual[0] & (logvar_value[0] < logvar_th) \
           & (logvar_value[1].gather(1, m[0]) < logvar_th)
        mscores0_test = torch.where(valid0, mscores[0], 10000)
        # print(f"with logvar_th {logvar_th} minimal mscores0_test value {mscores0_test.min().item()}")
        if mscores0_test.min().item() < 0.01:
            idx = mscores0_test.min(-1).indices
            p_rp_01 = pred["p_rp_01"]
            p_rp_10 = pred["p_rp_10"]
            scores = p_rp_01 * p_rp_10
            val01, minm0_m1_idx = scores[0][idx].max(-1)
            print(f' val01 {val01.item()}, minm0_m1_idx {minm0_m1_idx.item()}')
            print(f'min mscores0_test {mscores0_test.min().item()} at idx {idx}')
            print(f'logvar_th {logvar_th}')
            minm0_m1_match = m[0][0][idx]
            val10, minm1_m0_idx = scores[0][:, minm0_m1_match].max(-2)
            print(f'minm0_m1_match {minm0_m1_match.item()}')
            print(f' val10 {val10.item()}, minm1_m0_idx {minm1_m0_idx.item()}')

        m0, m1, mscores0, mscores1 = simpleglue.filter_matches_with_th(m, mscores, mutual, logvar_value, score_th, logvar_th)
        pred_filtered = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
        }
        return pred_filtered
    
    def compute_matches_lg(self, scores):
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
        indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
        mutual0 = indices0 == m1.gather(1, m0)
        mutual1 = indices1 == m0.gather(1, m1)
        max0_exp = max0.values.exp()
        zero = max0_exp.new_tensor(0)
        mscores0 = torch.where(mutual0, max0_exp, zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)

        return (m0, m1), (mscores0, mscores1), (mutual0, mutual1)
    
    def filter_matches_lg(self, m, mscores, mutual, score_th):
        valid0 = mutual[0] & (mscores[0] > score_th)
        valid1 = mutual[1] & valid0.gather(1, m[1])
        m0 = torch.where(valid0, m[0], -1)
        m1 = torch.where(valid1, m[1], -1)
        pred_filtered = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores[0],
            "matching_scores1": mscores[1],
        }
        return pred_filtered


    def run_eval(self, loader, pred_file):
        assert pred_file.exists()
        results = {}

        conf = self.conf.eval

        score_ths = (
            ([conf.filter_threshold] if conf.filter_threshold >= 0 else [0.1])
            if not isinstance(conf.filter_threshold, Iterable)
            else conf.filter_threshold
        )
        logvar_ths = (
            ([conf.logvar_filter_threshold])
            if not isinstance(conf.logvar_filter_threshold, Iterable)
            else conf.logvar_filter_threshold
        )

        metrics_keys = ('match_recall', 'match_precision', 'accuracy')
        results = {'names': []}
        if conf.model_name == "simpleglue":
            for th in score_ths:
                results[th] = {}
                for lv_th in logvar_ths:
                    results[th][lv_th] = {k: {"mean": AverageMetric(), "median": MedianMetric()} for k in metrics_keys}
        elif conf.model_name == "lightglue":
            for th in score_ths:
                results[th] = {k: {"mean": AverageMetric(), "median": MedianMetric()} for k in metrics_keys}

        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i, data in enumerate(tqdm(loader)):
            data = batch_to_device(data, device, non_blocking=True)
            pred = cache_loader(data)
            # Add batch dimension
            pred = map_tensor(pred, lambda t: torch.unsqueeze(t, dim=0))

            if conf.model_name == "simpleglue":
                m, mscores, mutual, logvar_value = self.compute_matches_sg(pred)

                for th in score_ths:
                    for lv_th in logvar_ths:
                        pred_filtered = self.filter_matches_sg( m, mscores, mutual, logvar_value, th, lv_th, pred)
                        metrics = matcher_metrics(pred_filtered, {**pred, **data})
                        for k in metrics_keys:
                            results[th][lv_th][k]['mean'].update(metrics[k])
                            results[th][lv_th][k]['median'].update(metrics[k])
            elif conf.model_name == "lightglue":
                scores = pred["log_assignment"]
                m, mscores, mutual = self.compute_matches_lg(scores)
                for th in score_ths:
                    pred_filtered = self.filter_matches_lg(m, mscores, mutual, th)
                    metrics = matcher_metrics(pred_filtered, {**pred, **data})
                    for k in metrics_keys:
                        results[th][k]['mean'].update(metrics[k])
                        results[th][k]['median'].update(metrics[k])
                    
            results["names"].append(data["name"][0])

        # compute mean and
        for th, th_results in results.items():
            if th == "names":
                continue
            for k, v in th_results.items():
                if conf.model_name == "simpleglue":
                    for kk, vv in v.items():
                        results[th][k][kk]['mean'] = vv['mean'].compute()
                        results[th][k][kk]['median'] = vv['median'].compute()
                elif conf.model_name == "lightglue":
                    results[th][k]['mean'] = v['mean'].compute()
                    results[th][k]['median'] = v['median'].compute()

        # plot pr cures for each threshold
        figures = {}
        if conf.model_name == "simpleglue":
            for th in score_ths:
                precision_i_mean = [results[th][lv_th]['match_precision']['mean'] for lv_th in logvar_ths]
                recall_i_mean = [results[th][lv_th]['match_recall']['mean'] for lv_th in logvar_ths]

                precision_i_median = [results[th][lv_th]['match_precision']['median'] for lv_th in logvar_ths]
                recall_i_median = [results[th][lv_th]['match_recall']['median'] for lv_th in logvar_ths]

                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].plot(recall_i_mean, precision_i_mean)
                axs[0].scatter(recall_i_mean, precision_i_mean)
                for i, lv_th in enumerate(logvar_ths):
                    axs[0].annotate(lv_th, (recall_i_mean[i], precision_i_mean[i]))
                axs[0].set_xlabel("recall")
                axs[0].set_ylabel("precision")
                axs[0].set_title(f'Mean precision-recall curve for threshold {th}')

                axs[1].plot(recall_i_median, precision_i_median, label="median")
                axs[1].scatter(recall_i_median, precision_i_median)
                for i, lv_th in enumerate(logvar_ths):
                    axs[1].annotate(lv_th, (recall_i_median[i], precision_i_median[i]))
                axs[1].set_xlabel("recall")
                axs[1].set_ylabel("precision")
                axs[1].set_title(f'Median precision-recall curve for threshold {th}')
                
                # tight layout
                plt.tight_layout()

                # store figure to pred_file location
                figures[f'pr_curve_{th}'] = fig

        # convert summaries and results to dict of {k, numbers}
        summaries = {}
        for th in score_ths:
            if conf.model_name == "simpleglue":
                for lv_th in logvar_ths:
                    for m in metrics_keys:
                        summaries[f'{th}_{lv_th}_{m}_mean'] = results[th][lv_th][m]['mean']
                        summaries[f'{th}_{lv_th}_{m}_median'] = results[th][lv_th][m]['median']
            elif conf.model_name == "lightglue":
                for m in metrics_keys:
                    summaries[f'{th}_{m}_mean'] = results[th][m]['mean']
                    summaries[f'{th}_{m}_median'] = results[th][m]['median']

        return summaries, figures, {}

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
    # save summary to json
    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(s, f)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
