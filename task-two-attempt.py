import pandas as pd

def price_gas_storage_auto(prices, injection_rate, withdrawal_rate, max_volume, storage_cost):
    current_volume = 0
    total_value = 0
    sorted_dates = sorted(prices.keys(), key=lambda x: pd.to_datetime(x, format='%m/%d/%y'))

    for i, date in enumerate(sorted_dates):
        price = prices[date]

        # Simple greedy strategy: inject if next price higher, withdraw if next price lower
        if i < len(sorted_dates) - 1:
            next_price = prices[sorted_dates[i + 1]]
            if next_price > price and current_volume < max_volume:  # inject
                inject = min(injection_rate, max_volume - current_volume)
                total_value -= inject * price
                current_volume += inject
            elif next_price < price and current_volume > 0:  # withdraw
                withdraw = min(withdrawal_rate, current_volume)
                total_value += withdraw * price
                current_volume -= withdraw
        
        # Storage cost for remaining gas
        total_value -= current_volume * storage_cost

    return total_value

# Load CSV
df = pd.read_csv("Nat_Gas.csv")
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
prices = {date.strftime('%-m/%-d/%y'): price for date, price in zip(df['Dates'], df['Prices'])}

# Calculate contract value automatically
value = price_gas_storage_auto(
    prices,
    injection_rate=100,
    withdrawal_rate=100,
    max_volume=500,
    storage_cost=0.05
)

print(f"Total contract value: {value:.2f}")
