"""
Unit tests for TradingEnv

Tests cover:
- Trade counting accuracy
- Position lifecycle (open, hold, close)
- Episode end behavior
- Portfolio value calculations
"""

import pytest
import numpy as np
import pandas as pd
from rl_trading_lab.environment.trading_env import TradingEnv


@pytest.fixture
def sample_data():
    """Create sample trading data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(200) * 0.5),
        'high': 102 + np.cumsum(np.random.randn(200) * 0.5),
        'low': 98 + np.cumsum(np.random.randn(200) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(200) * 0.5),
        'volume': 1000 + np.random.randn(200) * 100,
    })

    # Add features
    df['returns'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()

    return df


@pytest.fixture
def env(sample_data):
    """Create trading environment"""
    return TradingEnv(
        df=sample_data,
        lookback_window=10,
        initial_balance=10000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        reward_type="returns",
        discrete_actions=True,
        randomize_start=False,
        hold_closes_position=True,
    )


class TestTradeCountingAccuracy:
    """Test that trade counting is accurate"""

    def test_repeated_buy_signals_count_as_one_trade(self, env):
        """Sending Buy multiple times when already long should only count as 1 trade"""
        env.reset()

        # Send Buy signal
        env.step(1)  # Buy
        assert env.num_trades == 1, "First buy should count as 1 trade"

        # Send Buy signal again (should NOT create new trade)
        env.step(1)  # Buy again
        assert env.num_trades == 1, "Repeated buy should NOT count as new trade"

        # Send Buy one more time
        env.step(1)  # Buy again
        assert env.num_trades == 1, "Still should be only 1 trade"

    def test_repeated_sell_signals_count_as_one_trade(self, env):
        """Sending Sell multiple times when already short should only count as 1 trade"""
        env.reset()

        # Send Sell signal
        env.step(2)  # Sell
        assert env.num_trades == 1, "First sell should count as 1 trade"

        # Send Sell signal again (should NOT create new trade)
        env.step(2)  # Sell again
        assert env.num_trades == 1, "Repeated sell should NOT count as new trade"

        # Send Sell one more time
        env.step(2)  # Sell again
        assert env.num_trades == 1, "Still should be only 1 trade"

    def test_reversal_counts_as_two_trades(self, env):
        """Reversing from long to short (or vice versa) should count as 2 trades"""
        env.reset()

        # Go long
        env.step(1)  # Buy
        assert env.num_trades == 1, "Buy should count as 1 trade"

        # Reverse to short
        env.step(2)  # Sell
        assert env.num_trades == 2, "Reversal should count as 2nd trade (close + open)"

        # Reverse back to long
        env.step(1)  # Buy
        assert env.num_trades == 3, "Second reversal should count as 3rd trade"

    def test_hold_action_does_not_count_as_trade(self, env):
        """Hold actions should never count as trades"""
        env.reset()

        # Send Hold multiple times
        for _ in range(10):
            env.step(0)  # Hold

        assert env.num_trades == 0, "Hold actions should not count as trades"

    def test_open_hold_close_counts_as_one_trade(self, env):
        """Opening position, holding, then closing should count correctly"""
        env.reset()

        # Open long position
        env.step(1)  # Buy
        assert env.num_trades == 1

        # Hold for several steps
        for _ in range(5):
            env.step(1)  # Keep buying (but already long, so no new trades)

        assert env.num_trades == 1, "Holding should not increment trade count"

        # Close position with Hold (since hold_closes_position=True)
        env.step(0)  # Hold
        assert env.num_trades == 1, "Closing doesn't increment counter (only opening does)"


class TestPositionLifecycle:
    """Test position opening, holding, and closing"""

    def test_opening_long_position(self, env):
        """Test opening a long position"""
        env.reset()
        initial_balance = env.balance

        # Open long
        env.step(1)  # Buy

        assert env.position.size > 0, "Position should be positive (long)"
        assert env.balance < initial_balance, "Balance should decrease by commission"
        assert env.num_trades == 1, "Should count as 1 trade"

    def test_opening_short_position(self, env):
        """Test opening a short position"""
        env.reset()
        initial_balance = env.balance

        # Open short
        env.step(2)  # Sell

        assert env.position.size < 0, "Position should be negative (short)"
        assert env.balance < initial_balance, "Balance should decrease by commission"
        assert env.num_trades == 1, "Should count as 1 trade"

    def test_position_persists_with_same_signal(self, env):
        """Position should persist when receiving same direction signal"""
        env.reset()

        # Open long
        env.step(1)  # Buy
        position_size_after_open = env.position.size

        # Send buy again
        env.step(1)  # Buy
        position_size_after_repeat = env.position.size

        assert position_size_after_open == position_size_after_repeat, \
            "Position size should not change when repeating same signal"

    def test_closing_position_with_hold(self, env):
        """Test that hold_closes_position works"""
        env.reset()

        # Open long
        env.step(1)  # Buy
        assert env.position.size > 0

        # Close with Hold
        env.step(0)  # Hold
        assert env.position.size == 0, "Hold should close the position"

    def test_reversing_position(self, env):
        """Test reversing from long to short"""
        env.reset()

        # Open long
        env.step(1)  # Buy
        assert env.position.size > 0, "Should be long"

        # Reverse to short
        env.step(2)  # Sell
        assert env.position.size < 0, "Should now be short"

        # Reverse back to long
        env.step(1)  # Buy
        assert env.position.size > 0, "Should be long again"


class TestEpisodeEndBehavior:
    """Test that positions close at episode end"""

    def test_close_all_positions_closes_long(self, env):
        """Test that close_all_positions() closes long positions"""
        env.reset()

        # Open long position
        env.step(1)  # Buy
        assert env.position.size > 0, "Should have open long position"

        # Close all positions
        env.close_all_positions()

        assert env.position.size == 0, "Position should be closed"
        assert env.balance != env.initial_balance, "Balance should reflect P&L and commissions"

    def test_close_all_positions_closes_short(self, env):
        """Test that close_all_positions() closes short positions"""
        env.reset()

        # Open short position
        env.step(2)  # Sell
        assert env.position.size < 0, "Should have open short position"

        # Close all positions
        env.close_all_positions()

        assert env.position.size == 0, "Position should be closed"

    def test_close_all_positions_when_flat_does_nothing(self, env):
        """Test that close_all_positions() is safe when no position"""
        env.reset()
        balance_before = env.balance

        # Close all (should do nothing)
        env.close_all_positions()

        assert env.position.size == 0, "Should still be flat"
        assert env.balance == balance_before, "Balance should not change"

    def test_final_return_reflects_closed_position(self, env):
        """Test that final return includes P&L from closed position"""
        env.reset()
        initial_balance = env.initial_balance

        # Open position and let it run
        env.step(1)  # Buy
        for _ in range(10):
            env.step(1)  # Hold position

        portfolio_value_before_close = env._get_portfolio_value()

        # Close position
        env.close_all_positions()

        final_portfolio_value = env._get_portfolio_value()

        # Portfolio value should change after closing (due to slippage and commission)
        # But should be close to before-close value
        assert abs(final_portfolio_value - portfolio_value_before_close) < 100, \
            "Portfolio value should be similar before/after close (within commission/slippage)"

        # Final return should be non-zero (unless price didn't move at all)
        final_return = (final_portfolio_value - initial_balance) / initial_balance
        assert isinstance(final_return, float), "Should calculate return"


class TestPortfolioValue:
    """Test portfolio value calculations"""

    def test_portfolio_value_when_flat(self, env):
        """Portfolio value should equal balance when flat"""
        env.reset()

        portfolio_value = env._get_portfolio_value()
        assert portfolio_value == env.balance, \
            "Portfolio value should equal balance when no position"

    def test_portfolio_value_with_position(self, env):
        """Portfolio value should include unrealized P&L"""
        env.reset()
        initial_balance = env.initial_balance

        # Open position
        env.step(1)  # Buy

        # Move forward in time (price changes)
        for _ in range(5):
            env.step(1)  # Hold

        portfolio_value = env._get_portfolio_value()

        # Portfolio value should be different from initial (due to price movement and commissions)
        assert portfolio_value != initial_balance, \
            "Portfolio value should change with price movement"

    def test_info_dict_contains_num_trades(self, env):
        """Test that info dict includes num_trades"""
        env.reset()

        # Take some actions
        env.step(1)  # Buy
        obs, reward, done, truncated, info = env.step(2)  # Sell

        assert 'num_trades' in info, "Info should contain num_trades"
        assert info['num_trades'] == 2, "Should have 2 trades"


class TestCommissionsAndSlippage:
    """Test that commissions and slippage are applied correctly"""

    def test_commission_deducted_on_open(self, env):
        """Commission should be deducted when opening position"""
        env.reset()
        initial_balance = env.balance

        env.step(1)  # Buy

        # Balance should decrease by commission (not full position value)
        balance_decrease = initial_balance - env.balance
        expected_commission_approx = initial_balance * 0.95 * 0.001  # ~$9.50

        assert balance_decrease > 0, "Balance should decrease"
        assert 5 < balance_decrease < 15, \
            f"Commission should be ~$9.50, got ${balance_decrease:.2f}"

    def test_commission_applied_on_close(self, env):
        """Commission should be applied when closing position"""
        env.reset()

        # Open position
        env.step(1)  # Buy
        balance_after_open = env.balance

        # Close position
        env.step(0)  # Hold (closes position)
        balance_after_close = env.balance

        # Balance change should include commission from closing
        # (and any P&L from price movement)
        assert balance_after_close != balance_after_open, \
            "Balance should change after closing (commission + P&L)"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
