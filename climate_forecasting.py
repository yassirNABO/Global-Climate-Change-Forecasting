import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# 1. Scaling: Generate 10 years of daily environmental data (3,650 rows)
print("Simulating 10 years of global air quality data...")
dates = [datetime(2015, 1, 1) + timedelta(days=i) for i in range(3650)]
n_rows = len(dates)

# Generate synthetic CO2 levels with a steady upward trend + seasonal noise
trend = np.linspace(390, 420, n_rows) # Steady increase
seasonal_noise = 5 * np.sin(np.linspace(0, 20 * np.pi, n_rows)) # Annual fluctuations
noise = np.random.normal(0, 1, n_rows) # Daily random noise

co2_levels = trend + seasonal_noise + noise

df = pd.DataFrame({'Date': dates, 'CO2_Level': co2_levels})
df['Day_Index'] = np.arange(n_rows)

# 2. Moving Average (Smoothing technique used in Big Data)
df['Moving_Avg_30d'] = df['CO2_Level'].rolling(window=30).mean()

# 3. Machine Learning: Forecasting for the next 2 years (730 days)
X = df[['Day_Index']]
y = df['CO2_Level']

model = LinearRegression()
model.fit(X, y)

future_indices = np.arange(n_rows, n_rows + 730).reshape(-1, 1)
future_forecast = model.predict(future_indices)

# 4. Professional Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['CO2_Level'], label='Actual Daily Levels', alpha=0.3, color='gray')
plt.plot(df['Date'], df['Moving_Avg_30d'], label='30-Day Moving Average', color='blue', linewidth=2)

# Plot Forecast
future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 731)]
plt.plot(future_dates, future_forecast, label='2-Year Forecast Trend', color='red', linestyle='--', linewidth=2)

plt.title('Global CO2 Levels: Historical Data & 2-Year Trend Forecast', fontsize=14)
plt.xlabel('Year')
plt.ylabel('CO2 Concentration (ppm)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig('climate_forecast_plot.png')
print("Forecast complete. Analysis visualization saved.")
