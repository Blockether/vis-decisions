"""Vis bindings for the ordinary decision-model training library."""

from dataclasses import asdict

import blockether.vis.extension as vis

from blockether.vis_decisions.tools import DecisionTools


def _presentation(label):
    """Preserve model identities, validation metrics, digests, paths and errors."""

    def render(*, phase, result=None, error=None, **_):
        if phase == "start":
            return vis.ActivityPresentation(label, "Running")
        if phase == "failure":
            return vis.ActivityPresentation(
                label, "Failed", (vis.ActivityText(str(error)),)
            )
        if phase == "cancelled":
            return vis.ActivityPresentation(label, "Cancelled")
        if phase != "success":
            return None
        if isinstance(result, list):
            text = "\n".join(
                f"{row.model_id}: training {row.training_bytes:,} bytes, inference {row.inference_bytes:,} bytes"
                for row in result
            )
            return vis.ActivityPresentation(
                label, f"{len(result)} models", (vis.ActivityText(text or "No models"),)
            )
        fields = asdict(result)
        text = "\n".join(
            f"{key.replace('_', ' ').capitalize()}: {value}"
            for key, value in fields.items()
        )
        return vis.ActivityPresentation(label, "Completed", (vis.ActivityText(text),))

    return render


for name, label, tag in (
    ("models", "Read decision model catalog", "observation"),
    ("fetch", "Download training checkpoint", "mutation"),
    ("inspect_checkpoint", "Verify training checkpoint", "observation"),
    ("train", "Train decision model", "mutation"),
    ("prepare", "Export decision model", "mutation"),
    ("package", "Package decision model", "mutation"),
):
    setattr(
        DecisionTools,
        name,
        vis.method(
            tag=tag,
            activity=vis.Activity(
                label=label,
                show_start=name not in {"models", "inspect_checkpoint"},
                render=_presentation(label),
            ),
        )(getattr(DecisionTools, name)),
    )

vis.register_extension(
    vis.Extension(
        name="vis-decisions",
        version="0.3.0",
        description="Train, resume, validate and export Laya, GLiNER and Decision 2.0 models.",
        alias="decisions",
        symbols=[vis.Symbol(DecisionTools(), name="decisions")],
        prompt=(
            "decisions trains Laya, GLiNER and Decision 2.0 with one runtime. fetch downloads only a verified checkpoint. "
            "train accepts local labeled rows and disjoint held-out rows, and resumes matching partial checkpoints. "
            "prepare exports without training; package includes inference files only. "
            "Training does not activate a gateway alias. Publishing and activation use the Decisions Python client."
        ),
    )
)
