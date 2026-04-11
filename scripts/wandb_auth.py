"""W&B login for Kaggle (secret) or local (environment variable)."""

from __future__ import annotations

import os

import wandb

from . import config


def login() -> None:
    try:
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret("wandb_api_key")
        wandb.login(key=key)
    except Exception as e:
        env = os.environ.get("WANDB_API_KEY")
        if env:
            wandb.login(key=env)
        else:
            raise RuntimeError(
                "Add Kaggle secret 'wandb_api_key' or set WANDB_API_KEY in the environment."
            ) from e


WANDB_PROJECT = config.WANDB_PROJECT
