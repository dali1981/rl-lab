#!/usr/bin/env python
import pandas as pd

df = pd.read_csv('debug_episode_results.csv')

print('Looking for position lifecycle: flat → position → flat')
print()

# Find when position goes from 0 to non-zero
for i in range(len(df)-1):
    if abs(df.loc[i, 'position']) < 0.001 and abs(df.loc[i+1, 'position']) > 0.001:
        # Found opening
        print(f'Position OPENED at step {i+1}:')
        print(f'  Before (step {i}): balance=${df.loc[i, "balance"]:.2f}, position={df.loc[i, "position"]:.4f}, portfolio=${df.loc[i, "portfolio_value"]:.2f}')
        print(f'  After  (step {i+1}): balance=${df.loc[i+1, "balance"]:.2f}, position={df.loc[i+1, "position"]:.4f}, portfolio=${df.loc[i+1, "portfolio_value"]:.2f}')
        print(f'  Balance change: ${df.loc[i+1, "balance"] - df.loc[i, "balance"]:.2f}')
        print(f'  Portfolio change: ${df.loc[i+1, "portfolio_value"] - df.loc[i, "portfolio_value"]:.2f}')
        print()

        # Look for when it closes
        for j in range(i+1, min(i+20, len(df))):
            if abs(df.loc[j, 'position']) < 0.001:
                print(f'Position CLOSED at step {j}:')
                print(f'  Before (step {j-1}): balance=${df.loc[j-1, "balance"]:.2f}, position={df.loc[j-1, "position"]:.4f}, portfolio=${df.loc[j-1, "portfolio_value"]:.2f}')
                print(f'  After  (step {j}): balance=${df.loc[j, "balance"]:.2f}, position={df.loc[j, "position"]:.4f}, portfolio=${df.loc[j, "portfolio_value"]:.2f}')
                print(f'  Balance change: ${df.loc[j, "balance"] - df.loc[j-1, "balance"]:.2f}')
                print(f'  Portfolio change: ${df.loc[j, "portfolio_value"] - df.loc[j-1, "portfolio_value"]:.2f}')
                print()
                print(f'ROUND TRIP:')
                print(f'  Initial portfolio (step {i}): ${df.loc[i, "portfolio_value"]:.2f}')
                print(f'  Final portfolio (step {j}): ${df.loc[j, "portfolio_value"]:.2f}')
                print(f'  Net P&L: ${df.loc[j, "portfolio_value"] - df.loc[i, "portfolio_value"]:.2f}')
                break
        break
