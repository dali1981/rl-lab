"""
Domain Exceptions - Business rule violations.

These exceptions represent domain invariant violations and should be
caught and handled appropriately by the application layer.
"""


class DomainError(Exception):
    """Base class for domain exceptions."""

    pass


class InsufficientFundsError(DomainError):
    """Raised when attempting to debit more than available balance."""

    def __init__(self, requested: float, available: float):
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient funds: requested ${requested:.2f}, available ${available:.2f}"
        )


class InvalidPositionError(DomainError):
    """Raised when an invalid position operation is attempted."""

    pass


class InvalidOrderError(DomainError):
    """Raised when an invalid order is submitted."""

    pass
