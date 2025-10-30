"""
Real-time trading dashboard using Rich library.

Displays live portfolio metrics, positions, recent trades, and predictions.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

logger = logging.getLogger(__name__)


class TradingDashboard:
    """
    Real-time trading dashboard with Rich library.

    Displays:
    - Portfolio summary (balance, PnL, returns)
    - Active positions per symbol
    - Recent trades
    - Model predictions
    - Safety guard status

    Example:
        >>> dashboard = TradingDashboard()
        >>> with dashboard.live():
        ...     # Update dashboard in trading loop
        ...     dashboard.update(
        ...         portfolio_stats=portfolio.get_stats(),
        ...         predictions={"BTCUSDT": ("BUY", 0.85)},
        ...         recent_trades=trades[-5:]
        ...     )
    """

    def __init__(
        self,
        refresh_rate: float = 1.0,
        show_predictions: bool = True,
        show_features: bool = False,
    ):
        """
        Initialize dashboard.

        Args:
            refresh_rate: Screen refresh rate in seconds
            show_predictions: Show model predictions
            show_features: Show feature values (debugging)
        """
        self.console = Console()
        self.refresh_rate = refresh_rate
        self.show_predictions = show_predictions
        self.show_features = show_features

        # Data to display
        self.portfolio_stats: Dict = {}
        self.predictions: Dict = {}
        self.recent_trades: List = []
        self.safety_stats: Dict = {}
        self.feature_values: Dict = {}

        # Live display
        self._live: Optional[Live] = None

    def update(
        self,
        portfolio_stats: Optional[Dict] = None,
        predictions: Optional[Dict] = None,
        recent_trades: Optional[List] = None,
        safety_stats: Optional[Dict] = None,
        feature_values: Optional[Dict] = None,
    ):
        """
        Update dashboard data.

        Args:
            portfolio_stats: Portfolio statistics
            predictions: Model predictions per symbol
            recent_trades: Recent trade list
            safety_stats: Safety guard statistics
            feature_values: Feature values per symbol (for debugging)
        """
        if portfolio_stats:
            self.portfolio_stats = portfolio_stats
        if predictions:
            self.predictions = predictions
        if recent_trades is not None:
            self.recent_trades = recent_trades
        if safety_stats:
            self.safety_stats = safety_stats
        if feature_values:
            self.feature_values = feature_values

    def render(self) -> Layout:
        """Render the complete dashboard layout."""
        layout = Layout()

        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Split body into left and right
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        # Split left into sections
        layout["left"].split_column(
            Layout(name="portfolio", size=12),
            Layout(name="positions"),
        )

        # Split right into sections
        layout["right"].split_column(
            Layout(name="predictions", size=12),
            Layout(name="trades"),
        )

        # Render sections
        layout["header"].update(self._render_header())
        layout["portfolio"].update(self._render_portfolio())
        layout["positions"].update(self._render_positions())
        layout["predictions"].update(self._render_predictions())
        layout["trades"].update(self._render_trades())
        layout["footer"].update(self._render_footer())

        return layout

    def _render_header(self) -> Panel:
        """Render header with title and time."""
        title = Text("RL Trading Lab - Live Trading Dashboard", style="bold cyan")
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_text = Text.assemble(
            title,
            "\n",
            (time_str, "dim"),
        )

        return Panel(header_text, box=box.DOUBLE)

    def _render_portfolio(self) -> Panel:
        """Render portfolio summary."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        if self.portfolio_stats:
            balance = self.portfolio_stats.get("total_value", 0)
            cash = self.portfolio_stats.get("cash_balance", 0)
            position_value = self.portfolio_stats.get("position_value", 0)
            pnl = self.portfolio_stats.get("total_pnl", 0)
            returns = self.portfolio_stats.get("returns", 0) * 100
            drawdown = self.portfolio_stats.get("drawdown", 0) * 100

            # Color code PnL and returns
            pnl_color = "green" if pnl >= 0 else "red"
            returns_color = "green" if returns >= 0 else "red"

            table.add_row("Balance", f"${balance:,.2f}")
            table.add_row("Cash", f"${cash:,.2f}")
            table.add_row("Positions", f"${position_value:,.2f}")
            table.add_row("PnL", f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]")
            table.add_row("Returns", f"[{returns_color}]{returns:+.2f}%[/{returns_color}]")
            table.add_row("Drawdown", f"[red]{drawdown:.2f}%[/red]")
            table.add_row("Trades", str(self.portfolio_stats.get("total_trades", 0)))
        else:
            table.add_row("Status", "[dim]Waiting for data...[/dim]")

        return Panel(table, title="Portfolio Summary", border_style="blue")

    def _render_positions(self) -> Panel:
        """Render active positions."""
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("PnL", justify="right")

        positions = self.portfolio_stats.get("positions", {})
        if positions:
            for symbol, pos in positions.items():
                pnl = pos.get("unrealized_pnl", 0)
                pnl_color = "green" if pnl >= 0 else "red"

                table.add_row(
                    symbol,
                    f"{pos.get('quantity', 0):.8f}",
                    f"${pos.get('entry_price', 0):,.2f}",
                    f"${pos.get('current_price', 0):,.2f}",
                    f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]",
                )
        else:
            table.add_row("—", "—", "—", "—", "[dim]No positions[/dim]")

        return Panel(table, title="Active Positions", border_style="blue")

    def _render_predictions(self) -> Panel:
        """Render model predictions."""
        if not self.show_predictions:
            return Panel("[dim]Predictions disabled[/dim]", title="Model Predictions")

        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Symbol", style="cyan")
        table.add_column("Action", justify="center")
        table.add_column("Confidence", justify="right")

        if self.predictions:
            for symbol, (action, confidence) in self.predictions.items():
                # Color code actions
                action_colors = {
                    "BUY": "green",
                    "SELL": "red",
                    "HOLD": "yellow",
                }
                action_color = action_colors.get(action, "white")

                # Confidence bar
                conf_pct = int(confidence * 100)
                bar_length = int(confidence * 10)
                bar = "█" * bar_length + "░" * (10 - bar_length)

                table.add_row(
                    symbol,
                    f"[{action_color}]{action}[/{action_color}]",
                    f"{bar} {conf_pct}%",
                )
        else:
            table.add_row("—", "[dim]No predictions[/dim]", "—")

        return Panel(table, title="Model Predictions", border_style="magenta")

    def _render_trades(self) -> Panel:
        """Render recent trades."""
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Time", style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Action")
        table.add_column("Price", justify="right")
        table.add_column("PnL", justify="right")

        if self.recent_trades:
            for trade in self.recent_trades[-5:]:  # Show last 5
                time_str = trade.get("timestamp", "")[:19]
                if "T" in time_str:
                    time_str = time_str.split("T")[1]  # Just show time

                action = trade.get("action", "")
                action_color = "green" if action == "BUY" else "red" if action == "SELL" else "white"

                pnl = trade.get("pnl")
                if pnl is not None:
                    pnl_color = "green" if pnl >= 0 else "red"
                    pnl_str = f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]"
                else:
                    pnl_str = "—"

                table.add_row(
                    time_str,
                    trade.get("symbol", ""),
                    f"[{action_color}]{action}[/{action_color}]",
                    f"${trade.get('price', 0):,.2f}",
                    pnl_str,
                )
        else:
            table.add_row("—", "—", "[dim]No trades yet[/dim]", "—", "—")

        return Panel(table, title="Recent Trades", border_style="yellow")

    def _render_footer(self) -> Panel:
        """Render footer with safety status."""
        # Safety guard status
        state = self.safety_stats.get("state", "unknown")
        state_colors = {
            "closed": "green",
            "open": "red",
            "manual": "yellow",
        }
        state_color = state_colors.get(state, "white")

        footer_text = Text.assemble(
            "Safety: ",
            (f"{state.upper()}", f"bold {state_color}"),
            " | ",
            f"Drawdown: {self.safety_stats.get('drawdown_pct', 0):.1f}% | ",
            f"Trades/hr: {self.safety_stats.get('trades_last_hour', 0)}",
        )

        return Panel(footer_text, border_style="white")

    def live(self) -> Live:
        """
        Get Live context manager for continuous updates.

        Returns:
            Rich Live context manager
        """
        self._live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=1 / self.refresh_rate,
            screen=True,
        )
        return self._live

    def refresh(self):
        """Manually refresh the display."""
        if self._live:
            self._live.update(self.render())

    def print_summary(self):
        """Print a static summary (non-live)."""
        self.console.print(self.render())
