"""Canonical Open-in-Colab URLs for madrl_citylearn_v3_tutorial.ipynb.

Keep these constants aligned with REPO_BRANCH in the notebook.
The notebook lives at repo-root examples_madrl_v3/ on Mac-Tapia/MADRLCitytleranflexresdr.
"""

from __future__ import annotations

COLAB_NOTEBOOK_OWNER = "Mac-Tapia"
COLAB_NOTEBOOK_REPO = "MADRLCitytleranflexresdr"
COLAB_NOTEBOOK_BRANCH = "codex/fix-madrl-traceability-docs"
COLAB_NOTEBOOK_PATH = "examples_madrl_v3/madrl_citylearn_v3_tutorial.ipynb"

PARENT_REPO_OWNER = COLAB_NOTEBOOK_OWNER
PARENT_REPO_NAME = COLAB_NOTEBOOK_REPO
PARENT_REPO_BRANCH = COLAB_NOTEBOOK_BRANCH
PARENT_NOTEBOOK_PATH = COLAB_NOTEBOOK_PATH

COLAB_BADGE_SVG = "https://colab.research.google.com/assets/colab-badge.svg"


def github_colab_url(*, owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/{path}"


def open_in_colab_url() -> str:
    """Primary badge URL — notebook on parent repo branch."""
    return github_colab_url(
        owner=COLAB_NOTEBOOK_OWNER,
        repo=COLAB_NOTEBOOK_REPO,
        branch=COLAB_NOTEBOOK_BRANCH,
        path=COLAB_NOTEBOOK_PATH,
    )


def open_in_colab_parent_url() -> str:
    """Same as primary URL (notebook lives on the parent repo)."""
    return open_in_colab_url()


def markdown_badge_line() -> str:
    url = open_in_colab_url()
    return f"[![Open In Colab]({COLAB_BADGE_SVG})]({url})"


def badge_must_contain() -> str:
    """Substring required in notebook markdown for validation."""
    return f"github/{COLAB_NOTEBOOK_OWNER}/{COLAB_NOTEBOOK_REPO}/blob/{COLAB_NOTEBOOK_BRANCH}/{COLAB_NOTEBOOK_PATH}"
