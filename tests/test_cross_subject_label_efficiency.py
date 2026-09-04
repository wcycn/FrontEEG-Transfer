import numpy as np
from run_cross_subject_lowshot_matb import MODEL_CONFIG, nested_indices, parameter_count
from run_matb_ot_gnn_paper import GcnClassifier
from summarize_cross_subject_lowshot import summarize


def test_nested_indices_support_64_non_overlapping_windows_per_class() -> None:
    labels = np.repeat(np.arange(3), 148)
    choices = nested_indices(labels, maximum=64, seed=123)

    assert set(choices[1]).issubset(choices[2])
    assert set(choices[32]).issubset(choices[64])
    assert len(choices[64]) == 3 * 64
    for label in range(3):
        selected = sorted(index for index in choices[64] if labels[index] == label)
        assert all(
            right - left >= 2 for left, right in zip(selected, selected[1:], strict=False)
        )


def test_current_gcn_parameter_counts() -> None:
    model = GcnClassifier(MODEL_CONFIG)
    assert parameter_count(model) == 1_858

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.linear_layers[-1].parameters():
        parameter.requires_grad = True
    assert parameter_count(model, trainable_only=True) == 162


def _metrics(accuracy: float) -> dict:
    return {
        "accuracy": {"mean": accuracy, "std": 0.0},
        "balanced_accuracy": {"mean": accuracy, "std": 0.0},
        "macro_f1": {"mean": accuracy, "std": 0.0},
    }


def test_population_label_equivalence_uses_larger_scratch_budgets() -> None:
    subjects = []
    for index in range(2):
        subjects.append(
            {
                "subject": f"sub-{index:02d}",
                "source_only": {
                    "accuracy": 0.50,
                    "balanced_accuracy": 0.50,
                    "macro_f1": 0.50,
                },
                "source_training": {},
                "source_only_inference": {},
                "methods": {
                    "scratch": {"1": _metrics(0.35), "64": _metrics(0.55)},
                    "linear": {"1": _metrics(0.48), "64": _metrics(0.52)},
                    "full": {"1": _metrics(0.49), "64": _metrics(0.53)},
                },
            }
        )

    aggregate = summarize(subjects, [1, 64])

    assert aggregate["population_label_equivalence"]["source_only"][
        "minimum_observed_scratch_budget_per_class"
    ] == 64
    assert aggregate["population_label_equivalence"]["linear@1"][
        "minimum_observed_scratch_budget_per_class"
    ] == 64
