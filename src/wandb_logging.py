from typing import Any, Optional, Sequence

from src.wandb_config import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT


def init_wandb_run(
    enabled: bool,
    *,
    job_type: str,
    config: dict[str, Any],
    name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
):
    """Create a W&B run only when requested, keeping library imports out of normal tests."""
    if not enabled:
        return None

    import wandb

    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        mode=WANDB_MODE,
        name=name,
        group=group,
        tags=list(tags or ()),
        job_type=job_type,
        config=config,
    )


def flatten_verification_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float | bool]:
    """Reduce verification comparisons to scalar metrics that are compact in W&B."""
    flattened: dict[str, float | bool] = {}

    logits = metrics.get("logits", {})
    for metric_name in ("max_abs", "mean_abs", "rel_l2"):
        if metric_name in logits:
            flattened[f"{prefix}/logits_{metric_name}"] = logits[metric_name]

    if "next_token_match" in metrics:
        flattened[f"{prefix}/next_token_match"] = metrics["next_token_match"]

    hidden_states = metrics.get("hidden_states", [])
    if hidden_states:
        hidden_rel_l2 = [entry["rel_l2"] for entry in hidden_states if "rel_l2" in entry]
        hidden_max_abs = [entry["max_abs"] for entry in hidden_states if "max_abs" in entry]
        if hidden_rel_l2:
            flattened[f"{prefix}/hidden_rel_l2_max"] = max(hidden_rel_l2)
            flattened[f"{prefix}/hidden_rel_l2_mean"] = sum(hidden_rel_l2) / len(hidden_rel_l2)
        if hidden_max_abs:
            flattened[f"{prefix}/hidden_max_abs_max"] = max(hidden_max_abs)

    module_outputs = metrics.get("module_outputs", {})
    if module_outputs:
        module_rel_l2 = [entry["rel_l2"] for entry in module_outputs.values() if "rel_l2" in entry]
        if module_rel_l2:
            flattened[f"{prefix}/module_rel_l2_max"] = max(module_rel_l2)

    return flattened


def flatten_verification_layer_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    """Log detailed layer and module comparison stats without printing large tensors."""
    flattened: dict[str, float] = {}

    for hidden_metrics in metrics.get("hidden_states", []):
        layer_index = hidden_metrics.get("layer_index")
        if layer_index is None:
            continue
        for metric_name in ("max_abs", "mean_abs", "rel_l2"):
            if metric_name in hidden_metrics:
                flattened[f"{prefix}/hidden_layer_{layer_index}/{metric_name}"] = hidden_metrics[metric_name]

    for module_name, module_metrics in metrics.get("module_outputs", {}).items():
        safe_module_name = module_name.replace(".", "_")
        for metric_name in ("max_abs", "mean_abs", "rel_l2"):
            if metric_name in module_metrics:
                flattened[f"{prefix}/module/{safe_module_name}/{metric_name}"] = module_metrics[metric_name]

    return flattened


def compact_verification_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep CLI output focused on top-line verification metrics only."""
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"hidden_states", "module_outputs"}
    }


def flatten_rotation_summary(prefix: str, summary: dict[str, Any]) -> dict[str, float | int | str]:
    """Expose the useful rotation-state diagnostics without logging the matrices."""
    flattened: dict[str, float | int | str] = {}

    r1 = summary.get("R1", {})
    if "mode" in r1:
        flattened[f"{prefix}/R1_mode"] = r1["mode"]
    if "orthogonality_error" in r1:
        flattened[f"{prefix}/R1_orthogonality_error"] = r1["orthogonality_error"]

    r2 = summary.get("R2", {})
    for metric_name in (
        "mode",
        "count",
        "head_dim",
        "mean_orthogonality_error",
        "min_orthogonality_error",
        "max_orthogonality_error",
    ):
        if metric_name in r2:
            flattened[f"{prefix}/R2_{metric_name}"] = r2[metric_name]

    return flattened


def log_pipeline_results(run, results: dict[str, Any]) -> None:
    """Log one inference or verification pipeline result as scalar W&B summaries."""
    if run is None:
        return

    metrics: dict[str, Any] = {
        "pipeline/rotation_backend": results["rotation_backend"],
        "pipeline/rotate_mode": results["rotate_mode"],
        "pipeline/r2_mode": results["r2_mode"],
        "pipeline/model_name": results["model_name"],
        "pipeline/analog_target_count": len(results.get("analog_targets", [])),
    }
    if results.get("hardware_preset") is not None:
        metrics["pipeline/hardware_preset"] = results["hardware_preset"]

    metrics.update(flatten_rotation_summary("rotation", results["rotation_summary"]))
    metrics.update(flatten_verification_metrics("prep", results["prep_equivalence"]))
    metrics.update(flatten_verification_layer_metrics("prep/layers", results["prep_equivalence"]))
    metrics.update(flatten_verification_metrics("rotation", results["rotation_equivalence"]))
    metrics.update(flatten_verification_layer_metrics("rotation/layers", results["rotation_equivalence"]))
    metrics.update(flatten_verification_metrics("overall", results["float_equivalence"]))
    metrics.update(flatten_verification_layer_metrics("overall/layers", results["float_equivalence"]))

    if "baseline_to_analog_comparison" in results:
        metrics.update(
            flatten_verification_metrics("analog/baseline_to_analog", results["baseline_to_analog_comparison"])
        )
        metrics.update(
            flatten_verification_layer_metrics(
                "analog/baseline_to_analog/layers",
                results["baseline_to_analog_comparison"],
            )
        )
    if "rotated_float_to_analog_comparison" in results:
        metrics.update(
            flatten_verification_metrics("analog/rotated_float_to_analog", results["rotated_float_to_analog_comparison"])
        )
        metrics.update(
            flatten_verification_layer_metrics(
                "analog/rotated_float_to_analog/layers",
                results["rotated_float_to_analog_comparison"],
            )
        )

    run.log(metrics)
