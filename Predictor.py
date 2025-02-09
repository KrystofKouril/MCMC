import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd

# Preprocessing data

# Fetch S&P 500 data
sp500 = yf.download('^GSPC', start='2010-01-01', end='2025-02-07')

# Calculate the next day's opening return based on previous day's close
sp500['Next_Open_Return'] = sp500['Open'] / sp500['Close'].shift(1) - 1

# Calculate close to close returns
sp500['Close_to_Close_Return'] = sp500['Close'].pct_change()

# Calculate logarithmic returns
sp500['Log_Return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))

# Add volume feature
sp500['Volume_Change'] = sp500['Volume'].pct_change()

# Calculate volatility (20-day rolling standard deviation of returns)
sp500['Volatility'] = sp500['Log_Return'].rolling(window=20).std()

# Categorize returns
def categorize_next_open_state(ret):
    if ret > 0.002:
        return 'big_up'
    elif 0.002 >= ret >= 0.001:
        return 'small_up'
    elif ret < -0.002:
        return 'big_down'
    elif -0.002 <= ret <= -0.001:
        return 'small_down'
    else:
        return 'flat'

sp500['Next_Open_State'] = sp500['Next_Open_Return'].apply(categorize_next_open_state)

# Convert relevant columns to float
columns_to_convert = ['Next_Open_Return', 'Close_to_Close_Return', 'Volume_Change', 'Volatility']
for col in columns_to_convert:
    sp500[col] = pd.to_numeric(sp500[col], errors='coerce')

# Remove the first row (NaN due to return calculation)
sp500 = sp500.dropna()

print(sp500.tail())
print("\nDataset Shape:", sp500.shape)
print("\nColumns:", sp500.columns.tolist())
print("\nDate Range:")
print("First Date:", sp500.index[0])
print("Last Date:", sp500.index[-1])

# MCMC

states = ['big_up', 'small_up', 'flat', 'small_down', 'big_down']

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

def log_prior(state):
    return np.log(1/5)

def adaptive_proposal(current_state, iteration, accepted):
    current_index = states.index(current_state)

    if iteration > 1000:
        acceptance_rate = sum(accepted[-1000:]) / 1000
        if acceptance_rate > 0.234: # Optimal acceptance rate for many MCMC applications
            probs = [0.1 if i != current_index else 0.6 for i in range(5)]
        else:
            probs = [0.05 if i != current_index else 0.8 for i in range(5)]
    else:
        probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    return np.random.choice(states, p=probs)

def adaptive_metropolis_hastings(data, current_row, n_iterations=100000, burn_in=10000):
    current_state = np.random.choice(states)
    samples = []
    accepted = []

    for i in range(n_iterations):
        proposed_state = adaptive_proposal(current_state, i, accepted)

        current_log_posterior = log_likelihood(data, current_state, current_row) + log_prior(current_state)
        proposed_log_posterior = log_likelihood(data, proposed_state, current_row) + log_prior(proposed_state)

        log_alpha = proposed_log_posterior - current_log_posterior

        if np.log(np.random.random()) < log_alpha:
            current_state = proposed_state
            accepted.append(1)
        else:
            accepted.append(0)

        if i >= burn_in:
            samples.append(current_state)

    return samples

# Apply

def make_prediction(data):
    # Use the last row of the data as the current row
    current_row = data.iloc[-1]

    # Run the adaptive Metropolis-Hastings MCMC
    mcmc_samples = adaptive_metropolis_hastings(data, current_row, n_iterations=10000, burn_in=1000)

    # Determine the most frequent state in the MCMC samples
    prediction = max(set(mcmc_samples), key=mcmc_samples.count)

    next_day = data.index[-1] + pd.Timedelta(days=1)

    return prediction, next_day

# Assuming 'sp500' is your preprocessed DataFrame
prediction, next_day = make_prediction(sp500)
print(f"Predicted opening state for {next_day.strftime('%A, %B %d, %Y')}: {prediction}")
