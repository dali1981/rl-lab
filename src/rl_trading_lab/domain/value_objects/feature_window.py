"""
Feature Window Value Object - Immutable window of observation features.

Used for passing feature data from infrastructure to domain without
exposing DataFrame or numpy internals.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class FeatureWindow:
    """
    Immutable window of features for agent observation.

    Stores features as nested tuples for complete immutability.
    No numpy or pandas dependencies in the domain layer.

    Attributes:
        values: 2D tuple of feature values (rows x features)
        feature_names: Tuple of feature column names

    Example:
        >>> window = FeatureWindow(
        ...     values=((1.0, 2.0), (1.1, 2.1), (1.2, 2.2)),
        ...     feature_names=("close", "volume")
        ... )
        >>> window.flatten()
        (1.0, 2.0, 1.1, 2.1, 1.2, 2.2)
        >>> len(window)
        3
    """

    values: Tuple[Tuple[float, ...], ...]
    feature_names: Tuple[str, ...]

    def __post_init__(self):
        """Validate window data."""
        if len(self.values) == 0:
            return

        expected_cols = len(self.feature_names)
        for i, row in enumerate(self.values):
            if len(row) != expected_cols:
                raise ValueError(
                    f"Row {i} has {len(row)} values, expected {expected_cols} "
                    f"(matching feature_names)"
                )

    def __len__(self) -> int:
        """Number of rows (time steps) in the window."""
        return len(self.values)

    @property
    def n_features(self) -> int:
        """Number of features per row."""
        return len(self.feature_names)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the window (rows, features)."""
        return (len(self.values), len(self.feature_names))

    def flatten(self) -> Tuple[float, ...]:
        """
        Flatten to 1D tuple for observation vector.

        Returns:
            Flat tuple of all values in row-major order
        """
        return tuple(v for row in self.values for v in row)

    def get_column(self, feature_name: str) -> Tuple[float, ...]:
        """
        Get all values for a specific feature.

        Args:
            feature_name: Name of the feature column

        Returns:
            Tuple of values for that feature across all rows

        Raises:
            KeyError: If feature_name not found
        """
        try:
            idx = self.feature_names.index(feature_name)
        except ValueError:
            raise KeyError(f"Feature '{feature_name}' not found. Available: {self.feature_names}")
        return tuple(row[idx] for row in self.values)

    def get_row(self, index: int) -> Tuple[float, ...]:
        """
        Get all features for a specific row.

        Args:
            index: Row index (0-based, negative indexing supported)

        Returns:
            Tuple of feature values for that row
        """
        return self.values[index]

    def last_row(self) -> Tuple[float, ...]:
        """Get the most recent row (last row)."""
        return self.values[-1] if self.values else ()

    def first_row(self) -> Tuple[float, ...]:
        """Get the oldest row (first row)."""
        return self.values[0] if self.values else ()
