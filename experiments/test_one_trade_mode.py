#!/usr/bin/env python
"""Deprecated compatibility wrapper for runtime one-trade-mode checks."""

import warnings

from rl_trading_lab.runtime.one_trade_mode_entrypoint import main


if __name__ == "__main__":
    warnings.warn(
        "experiments/test_one_trade_mode.py is deprecated compatibility-only and "
        "scheduled for removal by June 30, 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(main())
