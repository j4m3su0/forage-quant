import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as spstats
import datetime as dt

import sys

print(sys.version)

def load_data(file_path: str) -> list:
    data = []

    with open(file_path, 'r', newline = '') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)  # Skips header

        for row in csv_reader:
            data.append(row)

    return data

def apply_linear_func(x: int, slope: float, intercept: float) -> int:
    return slope * x + intercept

def apply_sinusoidal_linear_func(x: int, slope: float, intercept: float, amplitude: float, period: float, shift: float):
    return amplitude * np.sin(((2 * np.pi) / period) * (x + shift)) + slope * (x + shift) + intercept

data = load_data('Nat_Gas.csv')
data_array = np.array(data)

x_data_str = data_array[:, 0]
# Converts dates from strings to datetime objects to be displayed
x_data_datetime = np.array([dt.datetime.strptime(date_str, '%m/%d/%y') for date_str in x_data_str])
# Converts dates from datetime objects to numerical values (to be able to fit a regression line)
x_data_numerical = mdates.date2num(x_data_datetime)
y_data = data_array[:, 1].astype(float)

slope, intercept, r, p, std_err = spstats.linregress(x_data_numerical, y_data)
model1 = apply_linear_func(x_data_numerical, slope, intercept)
model2 = apply_sinusoidal_linear_func(x_data_numerical, slope, intercept, 0.7, 365, 45)

expected_price1 = apply_linear_func(x_data_numerical[-1] + 365, slope, intercept)
expected_price2 = apply_sinusoidal_linear_func(x_data_numerical[-1] + 365, slope, intercept, 0.7, 365, 45)
print(f'Expected Purchase Price of Natural Gas (One Year in the Future) using Linear Regression: {expected_price1.round(1)}')
print(f'Expected Purchase Price of Natural Gas (One Year in the Future) using Sinusoidal Linear Regression: {expected_price2.round(1)}')

plt.figure(figsize = (12, 6))  # Adjusts size of display when rendered
plt.scatter(x_data_datetime, y_data)
plt.plot(x_data_numerical, model1, color = 'red')
plt.plot(x_data_numerical, model2, color = 'blue')
plt.xticks(rotation = 45, ha = 'right')  # Rotates the values on the x-axis
plt.title('Purchase Price of Natural Gas at the End of the Month, from 31st Oct. 2020 to 30th Sep. 2024')
plt.xlabel('Date')
plt.ylabel('Purchase Price')
plt.show()
