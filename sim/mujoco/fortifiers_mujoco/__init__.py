"""MuJoCo embodiment slice for the Fortifiers apprenticeship.

This package is deliberately separate from the browser teaching UI.  It owns
physics, camera observations, scene randomization, and a small action boundary
that can later be called by a VLA policy or a TypeScript process bridge.
"""

from .env import (
    OFFICIAL_SO101_MJCF_PATH,
    FortifiersMuJoCoEnv,
    PerturbationConfig,
)

__all__ = ["FortifiersMuJoCoEnv", "PerturbationConfig", "OFFICIAL_SO101_MJCF_PATH"]
