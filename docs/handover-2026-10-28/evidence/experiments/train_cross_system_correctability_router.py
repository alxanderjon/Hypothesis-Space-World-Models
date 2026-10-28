"""Train the smallest supported cross-system Newton-correctability router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


OUT = Path("outputs/cross_system_newton_correctability")
DATA = OUT / "cross_system_correctability_dataset.npz"
RECORDS = OUT / "label_records.json"
CHECKPOINT = OUT / "newton_correctability_router_frozen_candidate.npz"


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return dict(
        states=int(len(labels)), accuracy=float(np.mean(labels == predictions)),
        true_negatives=int(tn), false_positives=int(fp), false_negatives=int(fn),
        true_positives=int(tp),
        false_positive_rate=(float(fp / (fp + tn)) if fp + tn else None),
        false_negative_rate=(float(fn / (fn + tp)) if fn + tp else None),
    )


def fit_sparse(features: np.ndarray, labels: np.ndarray, splits: np.ndarray):
    train = splits == "train"
    scaler = StandardScaler().fit(features[train])
    standardized = scaler.transform(features)
    model = LogisticRegression(
        C=100.0, penalty="l1", solver="liblinear", max_iter=10_000,
        class_weight={0: 2.0, 1: 1.0}, random_state=0,
    ).fit(standardized[train], labels[train])
    return scaler, standardized, model


def leave_system_out(features, labels, splits, systems) -> dict:
    output = {}
    for held_system in ("sho", "bouncing_ball", "kepler"):
        train = (splits == "train") & (systems != held_system)
        test = (splits != "train") & (systems == held_system)
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(
            C=100.0, penalty="l1", solver="liblinear", max_iter=10_000,
            class_weight={0: 2.0, 1: 1.0}, random_state=0,
        ).fit(scaler.transform(features[train]), labels[train])
        predictions = model.predict(scaler.transform(features[test]))
        output[held_system] = dict(
            training_systems=sorted(set(systems[train])),
            **metrics(labels[test], predictions),
        )
    return output


def sequential(rows: list[dict], prediction_by_index: np.ndarray) -> dict:
    for row, prediction in zip(rows, prediction_by_index):
        row["router_prediction"] = bool(prediction)
    output = {}
    for system in ("sho", "bouncing_ball", "kepler"):
        records = []
        case_ids = sorted({int(row["case_id"]) for row in rows if row["system"] == system})
        for case_id in case_ids:
            values = [row for row in rows
                      if row["system"] == system and int(row["case_id"]) == case_id]
            if system == "kepler":
                # The reused Kepler corpus records every saved incumbent as an
                # n1_state (evaluations 1, 100, 500, and 1000).
                trajectory = sorted(
                    [row for row in values if row["source"] == "n1_state"],
                    key=lambda row: int(row["evaluation"]),
                )
            else:
                # New short runs use explicit incumbent checkpoints. Evaluation
                # 1 is the sole initial-state anchor; the other initial-design
                # candidates are independent probes used only for classification.
                trajectory = [row for row in values
                              if row["source"] == "n1_checkpoint"]
                initial = [row for row in values
                           if row["source"] == "n1_initial"
                           and int(row["evaluation"]) == 1]
                trajectory = sorted(initial + trajectory,
                                    key=lambda row: int(row["evaluation"]))
            actual = [row for row in trajectory if row["newton_correctable"]]
            predicted = [row for row in trajectory if row["router_prediction"]]
            first_actual = (min(int(row["evaluation"]) for row in actual) if actual else None)
            first_predicted = (min(int(row["evaluation"]) for row in predicted)
                               if predicted else None)
            if first_actual is None:
                stable = False
            else:
                stable = all(row["newton_correctable"] for row in trajectory
                             if int(row["evaluation"]) >= first_actual)
            if first_predicted is None:
                predicted_stable = False
            else:
                predicted_stable = all(row["router_prediction"] for row in trajectory
                                       if int(row["evaluation"]) >= first_predicted)
            old_loss = [int(row["evaluation"]) for row in trajectory
                        if float(row["loss"]) <= 1.0e-4]
            records.append(dict(
                case_id=case_id,
                first_actual_correctable_evaluation=first_actual,
                first_predicted_correctable_evaluation=first_predicted,
                sustained_correctability_after_entry=stable,
                sustained_router_prediction_after_entry=predicted_stable,
                sampled_trajectory_errors=sum(
                    row["router_prediction"] != row["newton_correctable"]
                    for row in trajectory
                ),
                first_sampled_old_n1_loss_gate_evaluation=(min(old_loss) if old_loss else None),
                old_n2_gate_timing=(
                    "not run in this short-data corpus; prior full-run gate status remains diagnostic"
                ),
            ))
        output[system] = records
    return output


def main() -> None:
    data = np.load(DATA)
    features = data["features"]
    labels = data["labels"].astype(np.int64)
    splits = data["splits"].astype(str)
    systems = data["systems"].astype(str)
    names = data["feature_names"].astype(str)
    rows = json.loads(RECORDS.read_text())
    scaler, standardized, model = fit_sparse(features, labels, splits)
    predictions = model.predict(standardized)
    logits = model.decision_function(standardized)
    selected = np.flatnonzero(model.coef_[0] != 0.0)
    np.savez_compressed(
        CHECKPOINT,
        feature_names=names[selected], feature_indices=selected,
        feature_mean=scaler.mean_[selected], feature_scale=scaler.scale_[selected],
        weights=model.coef_[0, selected], bias=model.intercept_,
        classification_logit=np.asarray((0.0,)),
    )
    checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    split_metrics = {
        split: metrics(labels[splits == split], predictions[splits == split])
        for split in ("train", "held_out", "fresh")
    }
    cross_system = {}
    for system in ("sho", "bouncing_ball", "kepler"):
        mask = (splits != "train") & (systems == system)
        cross_system[system] = metrics(labels[mask], predictions[mask])
    importance = sorted((
        dict(feature=str(names[index]), standardized_weight=float(model.coef_[0, index]),
             absolute_weight=float(abs(model.coef_[0, index])))
        for index in selected
    ), key=lambda row: row["absolute_weight"], reverse=True)
    architecture_screen = [
        dict(model="constant CORRECTABLE", parameters=0,
             rejection="11 false positives on the complete corpus"),
        dict(model="dense logistic", parameters=22,
             held_out_accuracy=0.9230769230769231, fresh_accuracy=0.96875,
             false_positives=0, false_negatives=6),
        dict(model="one-hidden-unit tanh", parameters=24,
             held_out_accuracy=0.9423076923076923, fresh_accuracy=0.90625,
             false_positives=0, false_negatives=9),
        dict(model="two-hidden-unit tanh", parameters=47,
             held_out_accuracy=0.9615384615384616, fresh_accuracy=0.953125,
             false_positives=0, false_negatives=5),
        dict(model="L1 sparse logistic", parameters=int(len(selected) + 1),
             selected=True, held_out_accuracy=split_metrics["held_out"]["accuracy"],
             fresh_accuracy=split_metrics["fresh"]["accuracy"],
             false_positives=(split_metrics["held_out"]["false_positives"]
                              + split_metrics["fresh"]["false_positives"]),
             false_negatives=(split_metrics["held_out"]["false_negatives"]
                              + split_metrics["fresh"]["false_negatives"])),
    ]
    report = dict(
        status="FROZEN_CANDIDATE",
        checkpoint=str(CHECKPOINT), checkpoint_sha256=checkpoint_hash,
        architecture="single sparse logistic routing head; CORRECTABLE iff logit >= 0",
        input_feature_count=int(features.shape[1]),
        active_feature_count=int(len(selected)), parameter_count=int(len(selected) + 1),
        ordinary_classification_threshold="logit >= 0 (sigmoid probability >= 0.5)",
        class_weighting="NOT_CORRECTABLE weight 2; CORRECTABLE weight 1",
        architecture_screen=architecture_screen,
        split_metrics=split_metrics,
        cross_system_held_out_and_fresh=cross_system,
        leave_one_system_out_diagnostic=leave_system_out(
            features, labels, splits, systems
        ),
        active_features=importance,
        no_ground_truth_runtime_features=True,
        sequential_diagnostic=sequential(rows, predictions),
        complete_corpus_metrics=metrics(labels, predictions),
        logits=dict(minimum=float(np.min(logits)), median=float(np.median(logits)),
                    maximum=float(np.max(logits))),
    )
    (OUT / "training_report.json").write_text(json.dumps(report, indent=2))
    router_marker = dict(
        status="FROZEN_CANDIDATE", architecture=report["architecture"],
        parameter_count=report["parameter_count"], checkpoint=str(CHECKPOINT),
        sha256=checkpoint_hash, training_report=str(OUT / "training_report.json"),
    )
    (OUT / "newton_correctability_router_frozen_candidate.json").write_text(
        json.dumps(router_marker, indent=2)
    )
    builder = json.loads((OUT / "report.json").read_text())
    builder.update(
        status="FROZEN_CANDIDATE", training_performed=True, checkpoint_saved=True,
        architecture=report["architecture"], parameter_count=report["parameter_count"],
        architecture_screen=architecture_screen,
        sequential_diagnostic=report["sequential_diagnostic"],
        training_report=str(OUT / "training_report.json"),
        router_checkpoint=str(CHECKPOINT), router_checkpoint_sha256=checkpoint_hash,
    )
    (OUT / "report.json").write_text(json.dumps(builder, indent=2))
    print(json.dumps({
        "status": report["status"], "parameter_count": report["parameter_count"],
        "split_metrics": split_metrics,
        "checkpoint_sha256": checkpoint_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
