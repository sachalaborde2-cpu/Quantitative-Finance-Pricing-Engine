# 📈 Quantitative Finance Option Pricing Engine: From Black-Scholes to Monte Carlo

**Overview**
This repository contains a comprehensive, Python-based quantitative finance engine designed to price financial derivatives, compute risk sensitivities (Greeks), and bridge theoretical mathematical models with empirical live market data. 

**Core Objective**
The primary goal of this project is to demonstrate the evolution and practical implementation of option pricing methodologies. It transitions from traditional analytical frameworks (Black-Scholes) to advanced numerical methods (Monte Carlo simulations), proving its real-world applicability by extracting and analyzing live market sentiment. 

**What the Code Returns & Outputs**
When executed, this script performs a full pipeline of quantitative analysis and outputs the following elements:
* **Theoretical Pricing & Risk Metrics:** Computes Vanilla Call/Put prices and their associated Greeks (Delta, Gamma, Vega, Theta, Rho) using finite difference methods, accompanied by a 6-panel sensitivity chart.
* **Live Volatility Skew (Market Inversion):** Fetches real-time S&P 500 (`^SPX`) options data via `yfinance`, applies a liquidity filter, and uses a custom Newton-Raphson algorithm to reverse-engineer and plot the empirical Implied Volatility Smile.
* **Stochastic Trajectories:** Generates a "Spaghetti Chart" illustrating 100 simulated price paths over a 252-day trading year using Geometric Brownian Motion.
* **Exotic Option Pricing:** Leverages the path-dependency of the Monte Carlo simulation to price an Asian Call Option (based on the arithmetic average of the underlying asset).
* **Market Reality Check:** Downloads a full year of historical market data, computes the annualized realized volatility, and compares it directly against the At-The-Money (ATM) Implied Volatility to assess current market risk premiums.
