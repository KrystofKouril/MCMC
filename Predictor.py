import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay


# Adjust volatility window based on market conditions
def get_adaptive_window(returns, min_window=10, max_window=30):
    initial_vol = returns.std()

    recent_vol = returns.rolling(5, min_periods=1).std().fillna(initial_vol)
    long_vol = returns.rolling(30, min_periods=5).std().fillna(initial_vol)

    # Avoid division by zero and handle NaN values
    long_vol = long_vol.replace(0, initial_vol)
    vol_ratio = (recent_vol / long_vol).fillna(1)
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], 1)

    # Shorter windows during high volatility periods
    window_sizes = (max_window * vol_ratio.clip(0.5, 2))
    window_sizes = window_sizes.round().astype(int)
    window_sizes = window_sizes.clip(min_window, max_window)

    # For the first few periods, use min_window
    window_sizes.iloc[:5] = min_window

    return window_sizes

# Calculate adaptive thresholds
def calculate_adaptive_thresholds(returns, prices):
    # Ensure we're working with clean data
    returns = returns.ffill().bfill()

    # Calculate base volatility using EWMA for more responsiveness
    base_vol = returns.ewm(span=20, min_periods=5).std()

    # Add trend detection with proper handling of NaN values
    prices_clean = prices.ffill()
    trend = prices_clean.pct_change(periods=10).rolling(
        window=10, min_periods=1
    ).mean()

    # Create trend factor with proper NaN handling
    trend_factor = (1 + trend.abs().fillna(0)).clip(0.8, 1.2)

    # Calculate final volatility adjustment
    vol_adjusted = base_vol * trend_factor

    # Ensure vol_adjusted is a Series
    if isinstance(vol_adjusted, pd.DataFrame):
        vol_adjusted = vol_adjusted.iloc[:, 0]

    # Calculate thresholds with minimum values
    big_threshold = (vol_adjusted * 0.5).clip(lower=0.0020, upper=0.0080)
    small_threshold = (vol_adjusted * 0.25).clip(lower=0.0010, upper=0.0040)

    # Final NaN cleanup
    big_threshold = big_threshold.fillna(0.0020)
    small_threshold = small_threshold.fillna(0.0010)

    return {
        'big': big_threshold,
        'small': small_threshold
    }

# Categorize returns
def categorize_next_open_state(ret, threshold_big, threshold_small):

    if ret > threshold_big:
        return 'big_up'
    elif ret > threshold_small:
        return 'small_up'
    elif ret < -threshold_big:
        return 'big_down'
    elif ret < -threshold_small:
        return 'small_down'
    else:
        return 'flat'

# MCMC

def log_likelihood(data, state, current_row):
    # Count transitions to the given state
    state_count = np.sum(data['Next_Open_State'] == state)

    # Convert current_row values to float numpy arrays
    close_return = current_row['Close_to_Close_Return'].item()
    volume_change = current_row['Volume_Change'].item()
    volatility = current_row['Volatility'].item()

    # Calculate means and stds for the given state
    state_data = data[data['Next_Open_State'] == state]
    close_return_mean = state_data['Close_to_Close_Return'].astype(float).mean()
    close_return_std = state_data['Close_to_Close_Return'].astype(float).std()
    volume_mean = state_data['Volume_Change'].astype(float).mean()
    volume_std = state_data['Volume_Change'].astype(float).std()
    volatility_mean = state_data['Volatility'].astype(float).mean()
    volatility_std = state_data['Volatility'].astype(float).std()

    # Handle zero standard deviations
    close_return_std = max(close_return_std, 1e-8)
    volume_std = max(volume_std, 1e-8)
    volatility_std = max(volatility_std, 1e-8)

    # Calculate probabilities
    close_return_prob = norm.pdf(close_return, loc=close_return_mean, scale=close_return_std)
    volume_prob = norm.pdf(volume_change, loc=volume_mean, scale=volume_std)
    volatility_prob = norm.pdf(volatility, loc=volatility_mean, scale=volatility_std)

    return (np.log(state_count + 1) + np.log(close_return_prob + 1e-10) +
            np.log(volume_prob + 1e-10) + np.log(volatility_prob + 1e-10))

def log_prior():
    return np.log(1/5)

def adaptive_proposal(current_state, iteration, accepted, burn_in, states):
    current_index = states.index(current_state)

    if iteration > burn_in:
        acceptance_rate = sum(accepted[-burn_in:]) / burn_in
        if acceptance_rate > 0.234: # Optimal acceptance rate for many MCMC applications
            probs = [0.1 if i != current_index else 0.6 for i in range(5)]
        else:
            probs = [0.05 if i != current_index else 0.8 for i in range(5)]
    else:
        probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    return np.random.choice(states, p=probs)

def adaptive_metropolis_hastings(data, current_row, n_iterations=10000, burn_in=1000):
    states = ['big_up', 'small_up', 'flat', 'small_down', 'big_down']
    current_state = np.random.choice(states)
    samples = []
    accepted = []

    for i in range(n_iterations):
        proposed_state = adaptive_proposal(current_state, i, accepted, burn_in, states)

        current_log_posterior = log_likelihood(data, current_state, current_row) + log_prior()
        proposed_log_posterior = log_likelihood(data, proposed_state, current_row) + log_prior()

        log_alpha = proposed_log_posterior - current_log_posterior

        if np.log(np.random.random()) < log_alpha:
            current_state = proposed_state
            accepted.append(1)
        else:
            accepted.append(0)

        if i >= burn_in:
            samples.append(current_state)

    return samples

def get_next_market_day(current_date):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=current_date.strftime('%Y-%m-%d'), end=(current_date + timedelta(days=10)).strftime('%Y-%m-%d'))
    next_day = pd.Timestamp(current_date.strftime('%Y-%m-%d')) + BDay(1)
    while next_day in holidays or next_day.weekday() >= 5:
        next_day = next_day + BDay(1)
    return next_day

def calculate_state_metrics(states, returns):
    if len(states) == 0:
        return {
            'state_distribution': pd.Series(),
            'state_stability': np.nan,
            'extreme_capture': np.nan,
            'false_signals': np.nan
        }

    metrics = {
        'state_distribution': states.value_counts(normalize=True),
        'state_stability': 1 - (states != states.shift()).mean(),
        'extreme_capture': (
            (states.isin(['big_up', 'big_down'])) &
            (returns.abs() > returns.std() * 2)
        ).mean(),
        'false_signals': (
            (states.isin(['big_up', 'big_down'])) &
            (returns.abs() < returns.std())
        ).mean()
    }
    return metrics

# Apply
def analyze_market_data(start_date='2010-01-01'):
    current_date = datetime.now()

    # Fetch S&P 500 data
    sp500 = yf.download('^GSPC', start=start_date,
                        end=(current_date + timedelta(days=1)).strftime('%Y-%m-%d'))

    # Calculate returns
    sp500['Next_Open_Return'] = sp500['Open'] / sp500['Close'].shift(1) - 1
    sp500['Close_to_Close_Return'] = sp500['Close'].pct_change()
    sp500['Log_Return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
    sp500['Volume_Change'] = sp500['Volume'].pct_change()
    sp500['Volatility'] = sp500['Log_Return'].ewm(span=20, min_periods=5).std()

    # Calculate adaptive thresholds
    thresholds = calculate_adaptive_thresholds(sp500['Log_Return'], sp500['Close'])

    # Apply state classification
    states = []
    for i in range(len(sp500)):
        threshold_big = float(thresholds['big'].iat[i])
        threshold_small = float(thresholds['small'].iat[i])

        state = categorize_next_open_state(
            float(sp500['Next_Open_Return'].iat[i]),
            threshold_big,
            threshold_small
        )
        states.append(state)

    sp500['Next_Open_State'] = states

    print(sp500.tail())
    print("\nDate Range:")
    print("First Date:", sp500.index[0])
    print("Last Date:", sp500.index[-1])

    # Convert relevant columns to float
    columns_to_convert = ['Next_Open_Return', 'Close_to_Close_Return',
                          'Volume_Change', 'Volatility']
    for col in columns_to_convert:
        sp500[col] = pd.to_numeric(sp500[col], errors='coerce')

    # Remove NaN rows
    sp500 = sp500.dropna()

    # Calculate performance metrics
    metrics = calculate_state_metrics(sp500['Next_Open_State'], sp500['Next_Open_Return'])

    # Make prediction for next market day
    current_row = sp500.iloc[-1:].copy()
    mcmc_samples = adaptive_metropolis_hastings(sp500, current_row)
    prediction = max(set(mcmc_samples), key=mcmc_samples.count)
    next_market_day = get_next_market_day(sp500.index[-1])

    return {
        'prediction': prediction,
        'next_market_day': next_market_day,
        'metrics': metrics,
        'thresholds': {
            'big': float(thresholds['big'].iloc[-1]),
            'small': float(thresholds['small'].iloc[-1])
        },
        'data': sp500
    }


if __name__ == "__main__":
    results = analyze_market_data()

    print(f"\nPredicted state for {results['next_market_day'].strftime('%A, %B %d, %Y')}: "
          f"{results['prediction']}")
    print("\nCurrent thresholds:")
    print(f"Big move: ±{results['thresholds']['big'] * 100:.2f}%")
    print(f"Small move: ±{results['thresholds']['small'] * 100:.2f}%")
    print("\nPerformance metrics:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value}")
