"""Canonical Open-in-Colab URLs for madrl_citylearn_v3_tutorial.ipynb.

Keep these constants aligned with REPO_BRANCH / CITYLEARN_BRANCH in the notebook.
The notebook file is committed on Mac-Tapia/CityLearn; Colab opens that copy directly.
"""

from __future__ import annotations

COLAB_NOTEBOOK_OWNER = "Mac-Tapia"
COLAB_NOTEBOOK_REPO = "CityLearn"
COLAB_NOTEBOOK_BRANCH = "codex/iquitos-distillation-madrl-docs"
COLAB_NOTEBOOK_PATH = "examples/madrl_citylearn_v3_tutorial.ipynb"

PARENT_REPO_OWNER = "Mac-Tapia"
PARENT_REPO_NAME = "MADRLCitytleranflexresdr"
PARENT_REPO_BRANCH = "codex/fix-madrl-traceability-docs"
PARENT_NOTEBOOK_PATH = "CityLearn/examples/madrl_citylearn_v3_tutorial.ipynb"

COLAB_BADGE_SVG = "https://colab.research.google.com/assets/colab-badge.svg"


def github_colab_url(*, owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/{path}"


def open_in_colab_url() -> str:
    """Primary badge URL — notebook on CityLearn live branch."""
    return github_colab_url(
        owner=COLAB_NOTEBOOK_OWNER,
        repo=COLAB_NOTEBOOK_REPO,
        branch=COLAB_NOTEBOOK_BRANCH,
        path=COLAB_NOTEBOOK_PATH,
    )


def open_in_colab_parent_url() -> str:
    """Alternate URL via parent repo (submodule pointer; may lag CityLearn HEAD)."""
    return github_colab_url(
        owner=PARENT_REPO_OWNER,
        repo=PARENT_REPO_NAME,
        branch=PARENT_REPO_BRANCH,
        path=PARENT_NOTEBOOK_PATH,
    )


def markdown_badge_line() -> str:
    url = open_in_colab_url()
    return f"[![Open In Colab]({COLAB_BADGE_SVG})]({url})"


def badge_must_contain() -> str:
    """Substring required in notebook markdown for validation."""
    return f"github/{COLAB_NOTEBOOK_OWNER}/{COLAB_NOTEBOOK_REPO}/blob/{COLAB_NOTEBOOK_BRANCH}/{COLAB_NOTEBOOK_PATH}"
