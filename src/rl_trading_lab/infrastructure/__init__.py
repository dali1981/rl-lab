"""
Infrastructure Layer - External system integrations.

This layer contains:
- Adapters: Implementations of domain ports (MarketDataAdapter, etc.)
- External APIs: Binance, MLflow, etc.
- Persistence: File I/O, databases

Infrastructure depends on the domain layer (implements ports).
Domain never depends on infrastructure.
"""
