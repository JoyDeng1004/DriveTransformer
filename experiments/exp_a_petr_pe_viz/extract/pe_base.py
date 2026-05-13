from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class PEExtractor(ABC):
    """Interface for geometry-to-PE extraction used by experiment scripts."""

    @abstractmethod
    def encode_image_tokens(
        self,
        lidar2img: np.ndarray,
        cam_intrinsic: np.ndarray,
        feature_hw: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        """Return PE fields for image tokens and the intermediate 3D coordinates."""

    @abstractmethod
    def encode_points(
        self,
        points_3d: np.ndarray,
        lidar2img: np.ndarray,
        cam_intrinsic: np.ndarray,
        image_hw: Tuple[int, int],
        camera_index: int = 0,
        feature_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return PE vectors sampled at point projections."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return extractor settings that should be stored with outputs."""
