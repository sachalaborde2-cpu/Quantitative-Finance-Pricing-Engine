from math import log, sqrt, exp, isclose
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Pricing call and put according to Black-Scholes model (T expressed in years)
# Auxiliary functions d1 and d2
def d1(S,K,T,r,v):
    return (log(S/K)+(r+(v**2)/2)*T)/(v*sqrt(T))
def d2(S,K,T,r,v):
    return d1(S,K,T,r,v)-v*sqrt(T)

# Vanilla Option Pricing Functions
def pricingcall(S,K,T,r,v):
    C=S*norm.cdf(d1(S, K, T, r,v))-K*exp(-r*T)*norm.cdf(d2(S, K, T, r, v))
    return C

def pricingput(S,K,T,r,v):
    P = K*exp(-r*T)*norm.cdf(-d2(S, K, T, r, v))-S*norm.cdf(-d1(S, K, T, r, v))
    return P

# Unit Tests for Pricing functions
expected=0
actual = pricingcall(1,1000,1,0,0.2)
assert isclose(actual,expected,abs_tol=0.0001)

expected=0
actual = pricingput(1000,1,1,0,0.2)
assert isclose(actual,expected,abs_tol=0.0001)


# Greeks Computation using Finite Difference Methods

# 1. Delta: First derivative w.r.t. Underlying Price (S)
def delta_approx(S,K,T,r,v,h=0.0001):
    price_up=pricingcall(S+h,K,T,r,v)
    price_down=pricingcall(S-h,K,T,r,v)
    return (price_up-price_down)/(2*h)

# Check: Delta of a Deep ITM Call should be approx 1
expected=1
actual= delta_approx(1000,1,1,0,0.2)
assert isclose(actual,expected,abs_tol=0.0001)

# Check: Delta of a Deep OTM Call should be approx 0
expected=0
actual= delta_approx(1,1000,1,0,0.2)
assert isclose(actual,expected,abs_tol=0.0001)


# 2. Gamma: Second derivative w.r.t. Underlying Price (S)
def gamma_approx(S,K,T,r,v,h=0.0001):
    price_up = pricingcall(S + h, K, T, r, v)
    price_down = pricingcall(S - h, K, T, r, v)
    price_actual= pricingcall(S,K,T,r,v)
    G= (price_up+price_down-2*price_actual)/(h**2)
    return G

# Check: Gamma of a Deep ITM Call should be 0 (No convexity)
expected=0
actual= gamma_approx(1000,1,1,0,0.2)
assert isclose(actual,expected,abs_tol=0.0001)

# Check: Gamma of a Deep OTM Call should be 0
expected=0
actual= gamma_approx(1,1000,1,0,0.2)
assert isclose(actual,expected,abs_tol=0.0001)


# 3. Vega: First derivative w.r.t. Volatility (v)
def vega_approx(S,K,T,r,v,h=0.0001):
    vol_up=pricingcall(S,K,T,r,v+h)
    vol_down = pricingcall(S, K, T, r, v - h)
    return (vol_up-vol_down)/(2*h)

# Check: Vega of a Deep ITM Call should be 0 (No uncertainty)
expected=0
actual= vega_approx(1000,1,1,0,0.2)
assert isclose(actual,expected,abs_tol=0.0001)

# Check: Vega close to expiration should be 0 (Time crush)
expected=0
actual= vega_approx(10,10,0.000000000001,0,0.2)
assert isclose(actual,expected,abs_tol=0.0001)


# 4. Theta: First derivative w.r.t. Time (T)
# Note: Uses a forward difference (T - h) to reflect time decay
def theta_approx(S,K,T,r,v,h=(1/365)):
    theta_down=pricingcall(S,K,T-h,r,v)
    theta_actual= pricingcall(S,K,T,r,v)
    return (theta_down-theta_actual)/(h)

# Check: Long Call Theta must be strictly negative (Time decay)
actual=theta_approx(100,100,1,0.05,0.2)
assert actual<0

# Check: Theta Magnitude (ATM vs OTM)
# Due to interest rates, ITM decay is skewed. Comparing ATM vs slightly OTM confirms the uncertainty peak.
C_ATM_theta=theta_approx(100,100,1,0.05,0.2)
C_ITM_theta=theta_approx(90,100,1,0.05,0.2)
assert abs(C_ITM_theta)<abs(C_ATM_theta)


# 5. Rho (Kho): First derivative w.r.t. Risk-Free Rate (r)
def kho_approx(S,K,T,r,v,h=(0.0001)):
    kho_up=pricingcall(S,K,T,r+h,v)
    kho_down=pricingcall(S, K, T, r-h, v)
    return (kho_up-kho_down)/(h*2)

# Check: Rho of a Long Call must be positive
actual=kho_approx(100,100,1,0.05,0.2)
assert actual>0

# Check: Rho increases with Maturity (Time Value of Money)
C_largematurity=kho_approx(100,100,10,0.05,0.2)
C_lowmaturity=kho_approx(90,100,1/12,0.05,0.2)
assert C_lowmaturity<C_largematurity





# Black-Scholes Sensitivity Analysis
# 1. Market and Option Parameters
K_sim = 100       # Strike Price
T_sim = 1         # Time to Maturity
r_sim = 0.05      # Risk-free Rate
v_sim = 0.2       # Volatility

# Spot Price domain definition
S_vals = np.linspace(50, 150, 100)

# 2. Initialize storage lists
P_vals = []
Delta_vals = []
gamma_vals = []
vega_vals = []
theta_vals = []
kho_vals = []

# 3. Compute Greeks for each spot price
for k in range(len(S_vals)):
    P_vals.append(pricingcall(S_vals[k], K_sim, T_sim, r_sim, v_sim))
    Delta_vals.append(delta_approx(S_vals[k], K_sim, T_sim, r_sim, v_sim))
    gamma_vals.append(gamma_approx(S_vals[k], K_sim, T_sim, r_sim, v_sim))
    vega_vals.append(vega_approx(S_vals[k], K_sim, T_sim, r_sim, v_sim))
    theta_vals.append(theta_approx(S_vals[k], K_sim, T_sim, r_sim, v_sim))
    kho_vals.append(kho_approx(S_vals[k], K_sim, T_sim, r_sim, v_sim))

# 4. Visualization setup
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Black-Scholes Option Profile (K={K_sim}, T={T_sim}, r={r_sim}, v={v_sim})', fontsize=16)

# Plotting Price
ax[0,0].plot(S_vals, P_vals, label='Pricing Curve', color="blue")
ax[0,0].axvline(x=K_sim, linestyle='--', color='black', alpha=0.75)
ax[0,0].set(title='Option Price', xlabel='Spot Price', ylabel='Price')

# Plotting Delta
ax[0,1].plot(S_vals, Delta_vals, label='Delta Curve', color="red")
ax[0,1].axvline(x=K_sim, linestyle='--', color='black', alpha=0.75)
ax[0,1].set(title='Delta', xlabel='Spot Price', ylabel='Delta')

# Plotting Gamma
ax[0,2].plot(S_vals, gamma_vals, label='Gamma Curve', color="green")
ax[0,2].axvline(x=K_sim, linestyle='--', color='black', alpha=0.75)
ax[0,2].set(title='Gamma', xlabel='Spot Price', ylabel='Gamma')

# Plotting Vega
ax[1,0].plot(S_vals, vega_vals, label='Vega Curve', color="orange")
ax[1,0].axvline(x=K_sim, linestyle='--', color='black', alpha=0.75)
ax[1,0].set(title='Vega', xlabel='Spot Price', ylabel='Vega')

# Plotting Theta
ax[1,1].plot(S_vals, theta_vals, label='Theta Curve', color="purple")
ax[1,1].axvline(x=K_sim, linestyle='--', color='black', alpha=0.75)
ax[1,1].set(title='Theta', xlabel='Spot Price', ylabel='Theta')

# Plotting Rho (Kho)
ax[1,2].plot(S_vals, kho_vals, label='Kho Curve', color="yellow")
ax[1,2].axvline(x=K_sim, linestyle='--', color='black', alpha=0.75)
ax[1,2].set(title='Rho (Kho)', xlabel='Spot Price', ylabel='Rho')

# Final layout adjustment
plt.tight_layout()
plt.show()


# Newton-Raphson method to estimate Implied Volatility
def implied_volatility(market_price, S, K, T, r):
    sigma = 0.5
    precision = 0.00001
    # Loop until the model price matches the market price within precision
    while abs(pricingcall(S, K, T, r, sigma) - market_price) > precision:
        vega = vega_approx(S, K, T, r, sigma)
        # Safety check to avoid division by zero
        if vega < 0.0000001:
            break
        # Update sigma using Newton-Raphson formula
        sigma = sigma - (pricingcall(S, K, T, r, sigma) - market_price) / vega
    return sigma

# Unit Test

# Test parameters
S_Iv = 100
K_Iv = 110
T_Iv = 1
r_Iv = 0.05
sigma_Iv = 0.2
# 1. Generate theoretical market price
market_price_Iv = pricingcall(S_Iv, K_Iv, T_Iv, r_Iv, sigma_Iv)
# 2. Reverse-engineer volatility from price
actual = implied_volatility(market_price_Iv, S_Iv, K_Iv, T_Iv, r_Iv)
# 3. Validate result
assert isclose(actual, sigma_Iv, abs_tol=0.00001)



#Live Market Testing: Implied Volatility Skew

# 1. Data Acquisition: Fetching live market data for the underlying asset
asset = yf.Ticker("^SPX")
history_1d = asset.history(period="1d")
S_market = history_1d['Close'].iloc[0]

# Extracting the option chain for the targeted expiration date
target_date = "2026-04-10"
options_calls = asset.option_chain(target_date).calls

# 2. Data Cleansing: Filtering illiquid options to ensure model stability
# We only keep contracts with actual trading volume and valid bid prices
options_calls = options_calls[options_calls["volume"] > 0]
options_calls = options_calls[options_calls["bid"] > 0]

# Computing the mid-price to eliminate the bid-ask spread bias
options_calls["market_price"] = (options_calls["bid"] + options_calls["ask"]) / 2

# 3. Parameter Initialization: Standardizing inputs for the Black-Scholes model
r_market = 0.04  # Assuming a static 4% risk-free rate
today = datetime.now()
expiration_date = datetime.strptime(target_date, "%Y-%m-%d")

# Annualizing the time to maturity (T)
T_market = (expiration_date - today).days / 365

# 4. Iterative Computation: Reverse-engineering the Implied Volatility
computed_ivs = []

# Looping through the dataset to compute IV for each strike
options_calls = options_calls[options_calls["strike"] >= S_market * 0.98]
for index, row in options_calls.iterrows():
    target_price = row["market_price"]
    K = row["strike"]

    # Calling our custom Newton-Raphson algorithm
    iv = implied_volatility(target_price, S_market, K, T_market, r_market)
    computed_ivs.append(iv)

# Storing our calculated results back into the dataframe
options_calls["My_IV"] = computed_ivs

# 5. Data Visualization: Plotting the Volatility Smile
plt.figure(figsize=(10, 6))

# Plotting the empirical volatility curve
plt.plot(options_calls["strike"], options_calls["My_IV"], label="Implied Volatility Curve", color="red", marker='o')

# Formatting the chart layout
plt.title("Implied Volatility Smile / Skew")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")

# Highlighting the At-The-Money (ATM) threshold
plt.axvline(x=S_market, linestyle='--', color='black', alpha=0.75, label="Current Spot Price (ATM)")

plt.legend()
plt.show()

# Monte Carlo Simulation for Vanilla Call Pricing

# 1. Parameters and Stochastic Component setup
M = 100000
S_sim = 100
Z = np.random.standard_normal(size=M)

# 2. Geometric Brownian Motion (GBM) Function
def simulation_MBG(S, T, r, v, V):
    St = S * np.exp((r - ((v**2) / 2)) * T + v * np.sqrt(T) * V)
    return St

# Simulating M possible future spot prices at maturity
S_t = simulation_MBG(S_sim, T_sim, r_sim, v_sim, Z)

# 3. Payoff Calculation and Discounting
# Computing the vectorized payoff at maturity
Payoffs = np.maximum(S_t - K_sim, 0)

# Averaging the payoffs and applying the continuous discount factor
OptionPrice = np.mean(Payoffs) * np.exp((-r_sim) * T_sim)

# 4. Convergence Check: Comparing Monte Carlo with Exact Black-Scholes
print(f"Monte Carlo Price ({M} simulations): {OptionPrice:.4f}")
print(f"Black-Scholes Price (Exact): {pricingcall(S_sim, K_sim, T_sim, r_sim, v_sim):.4f}")




# 1. Path Generation: Discretizing time into trading days
jours = 252
Dt = T_sim / jours

# Generating a matrix of stochastic shocks for 100 paths over 252 days
Z_matrice = np.random.standard_normal((jours, 100))

# Initializing the price paths matrix and setting the initial spot price
S_paths = np.zeros((jours + 1, 100))
S_paths[0] = S_sim

# Iterative simulation of the Geometric Brownian Motion
for t in range(1, jours + 1):
    S_paths[t] = simulation_MBG(S_paths[t - 1], Dt, r_sim, v_sim, Z_matrice[t - 1])

# 2. Data Visualization: Plotting the Spaghetti Chart
plt.figure(figsize=(10, 6))

# Plotting all 100 simulated trajectories simultaneously
plt.plot(S_paths, linewidth=1, alpha=0.8)

# Chart layout and formatting
plt.title("Monte Carlo Simulation: 100 Price Paths (1 Year)")
plt.xlabel("Trading Days")
plt.ylabel("Asset Price")

# Strike price reference line
plt.axhline(y=K_sim, color='black', linestyle='--', label=f"Strike Price (K={K_sim})", linewidth=2)

plt.legend()
plt.show()


# Path-Dependent Exotic Options: Pricing an Asian Call

# 1. Computing the arithmetic mean of the asset price along each simulated path
S_moyen = np.mean(S_paths, axis=0)

# 2. Calculating the payoff based on the average price (Path-dependency)
Payoffs_AC = np.maximum(S_moyen - K_sim, 0)

# 3. Averaging the payoffs and applying the continuous discount factor
AC_price = np.mean(Payoffs_AC) * np.exp((-r_sim) * T_sim)

# Output the final expected price
print(f"Monte Carlo Price (Asian Call): {AC_price:.4f}")

# Market Reality Check: Realized vs. Implied Volatility

# 1. Fetching one year of historical market data for the underlying asset
history_1y = asset.history(period="1y")

# 2. Computing daily logarithmic returns
log_returns = np.log(history_1y['Close'] / history_1y['Close'].shift(1))

# 3. Calculating daily volatility (standard deviation of returns)
day_volatility = log_returns.std()

# 4. Annualizing the volatility (assuming 252 trading days)
historical_volatility = day_volatility * np.sqrt(252)

# Output the final metric
print(f"Historical Volatility (1 Year): {historical_volatility:.4f}")

# 5. Extracting the At-The-Money (ATM) Implied Volatility for direct comparison
iv_atm = options_calls.loc[(options_calls['strike'] - S_market).abs().idxmin(), 'My_IV']

# 6. Final Market Sentiment Output
print(f"Market Comparison -> Implied Volatility (ATM): {iv_atm:.4f} vs Historical Volatility: {historical_volatility:.4f}")



