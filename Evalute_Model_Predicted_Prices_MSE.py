import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


window_size = 1
def find_matching_files(directory, pattern):
    # List to hold all matching files
    matching_files = []

    # Compile the regular expression for matching filenames
    regex = re.compile(pattern)

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        if regex.match(filename):
            matching_files.append(filename)

    return matching_files


# Specify the directory to search in
directory = 'C:/Users/oc21380/PycharmProjects/Parameter_Calibration/H1/'

# Define the pattern to match filenames
# r'^bse_ICAART24_II_H1_RWMM_s01_d005_i01_(\d{4})_U0\.2_L0\.2_K0\.9_P00\.6_transactions\.csv$'
#  r'^bse_ICAART_H3_RWMM_s01_d005_i01_(\d{4})_U0\.6_L1\.0_K1\.0_P00\.1_transactions\.csv$'
# r'^bse_ICAART24_II_H2_RWMM_s01_d005_i05_\d{4}_transactions\.csv$'

# bse_ICAART24_II_H2_H3_RWMMs01_d005_i05_(\d{4})_transactions.csv$
# This pattern assumes that the varying part is like 0035 and it's surrounded by similar structure

# bse_ICAART24_II_H1_H2_RWMM_s01_d005_i01_(\d{4})_U0.2_L0.2_K0.9_P00.6_transactions.csv$
# bse_ICAART24_II_H1_H3_RWMM_s01_d005_i01_(\d{4})_transactions.csv$
# bse_ICAART24_II_H2_H3_RWMMs01_d005_i05_(\d{4})_transactions.csv$
# bse_ICAART24_II_H1_H2_H3_corr_RWMM_s01_d005_i05_(\d{4})_transactions.csv$
# bse_ICAART24_II_H1_H2_RWMM_s01_d005_i01_(\d{4})_U0.2_L0.2_K0.9_P00.6_transactions.csv$

# r'^bse_ICAART25_II_H1_RWMM_s01_d005_i01_0001_U0.2_L0.2_K0.9_P00.6_Noisy_ratio_0.1_transactions.csv$'
# r'^bse_ICAART25_II_H1_RWMM_s01_d005_i01_0001_U0.2_L0.2_K0.9_P00.6_Informed_ratio_0.8_transactions.csv$'
# r'^bse_ICAART24_II_H2_H3_RWMMs01_d005_i05_(\d{4})_transactions.csv$'


pattern = r'^bse_ICAART24_II_H1_RWMM_s01_d005_i01_(\d{4})_U0\.2_L0\.2_K0\.9_P00\.6_transactions\.csv$'


# Call the function and print the results
matching_files = find_matching_files(directory, pattern)
print(matching_files)
all_averages = []
for file in matching_files:
    if os.path.getsize(file) > 0:
        data = pd.read_csv(file, header=None, names=['trd', 'Date', 'timestamp', 'price'])
        # Check if DataFrame is empty after loading (no rows)
        if not data.empty:

            # Convert 'Date' to datetime format (if not already)
            data['Date'] = pd.to_datetime(data['Date'])

            # Group by 'Date' and calculate the average price
            daily_avg = data.groupby('Date')['price'].mean().reset_index()
            daily_avg['filename'] = file  # Track which file this data came from

            # Append the grouped data to the list
            all_averages.append(daily_avg)

        else:
            print(f"File {file} is empty after loading.")
    else:
        print(f"File {file} is empty (0 bytes).")






# Concatenate all dataframes in the list into one dataframe
combined_averages = pd.concat(all_averages)

# Group by 'Date' again to calculate the overall average across all files
final_averages = combined_averages.groupby('Date')['price'].mean().reset_index()

print(final_averages)
transactions_df = final_averages
transactions_df.to_csv('Average_H1_H2_H3_RWMM.csv', index=False)

transactions_df['Moving_Average'] = transactions_df['price'].rolling(window=window_size, min_periods=1).mean()




daily_prices_aliged = transactions_df['Moving_Average'] .to_numpy().reshape(-1,1)
daily_prices_aliged_log = np.log(daily_prices_aliged)


"""=================================================================================="""
bitcoin_df = pd.read_csv('dataset.csv', header=0)
#bitcoin_df = bitcoin_df.iloc[:-14]
#print(bitcoin_df.describe())
# Extract 'Date' and 'BTC_Closing' columns
bitcoin_closing_prices = bitcoin_df[['Date', 'BTC_Closing']]

bitcoin_closing_prices = bitcoin_closing_prices.copy()
#bitcoin_closing_prices.drop(columns=bitcoin_closing_prices.columns[0], inplace=True)



btc_prices_aligned = bitcoin_closing_prices['BTC_Closing'].to_numpy().reshape(-1,1)
btc_prices_aligned_log = np.log(btc_prices_aligned)
#print(btc_prices_aligned)
"""=================================================================================="""


# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()



daily_prices_norm = scaler.fit_transform(daily_prices_aliged_log)
btc_prices_norm = scaler.fit_transform(btc_prices_aligned_log)


daily_prices_series = pd.Series(daily_prices_norm.flatten(), name='Daily_Price')
btc_prices_series = pd.Series(btc_prices_norm.flatten(), name='BTC_Price')


# Compute the Pearson correlation
correlation = daily_prices_series.corr(btc_prices_series)

print("Pearson Correlation Coefficient:", correlation)

# Truncate daily_prices_norm to match the length of btc_prices_norm
#if len(daily_prices_norm) > len(btc_prices_norm):
 #   daily_prices_norm = daily_prices_norm[:len(btc_prices_norm)]
#elif len(btc_prices_norm) > len(daily_prices_norm):
 #   btc_prices_norm = btc_prices_norm[:len(daily_prices_norm)]

# Now that they're the same length, compute MSE
MSE = mean_squared_error(daily_prices_norm, btc_prices_norm)
print("MSE:", round(MSE,4))
import matplotlib.pyplot as plt

# Assuming 'transactions_df' and 'bitcoin_closing_prices' are already loaded and prepared
# Convert 'Date' in 'bitcoin_closing_prices' to datetime to ensure alignment
bitcoin_closing_prices['Date'] = pd.to_datetime(bitcoin_closing_prices['Date'])

# Merge the datasets on 'Date'
combined_data = pd.merge(transactions_df, bitcoin_closing_prices, on='Date', how='inner')

# Normalize the combined data for plotting
scaler = MinMaxScaler()
combined_data[['price_norm', 'BTC_Closing_norm']] = scaler.fit_transform(combined_data[['price', 'BTC_Closing']])

# Plotting
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(14, 7))
plt.plot(combined_data['Date'], daily_prices_norm, label='Daily Average Transaction Price')
plt.plot(combined_data['Date'], btc_prices_norm, label='Bitcoin Closing Price', linestyle='--')
#plt.title('Comparison of Normalized Prices')
plt.xlabel('Year')
plt.ylabel('log(standardized Price)')
plt.legend()
plt.grid(True)
plt.show()
