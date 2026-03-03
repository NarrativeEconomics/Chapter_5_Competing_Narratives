import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# bse_AAMAS24_H2_RMM_CorrectedMarket_s01_d078
# bse_ICAART24_H2_Case1_s01_d007_i50_000
# Prepare to capture all Opinion and Gamma-Delta data

opinion_data = []
gamma_data = []
delta_data = []
#

# bse_ICAART24_H2_Case1_s01_d007_i05_000
for i in range(1,2):
    if i > 9:
        trial_id = 'bse_ICAART24_H2_Case3_s01_d007_i05_00' + str(i)
    else:
        trial_id = 'bse_ICAART24_H2_Case3_s01_d007_i05_000' + str(i)

    # File location
    file_path = 'C:/Users/oc21380/PycharmProjects/paper3Corrected/H2_Case3/' + trial_id + '_opinions.csv'
    data_df = pd.read_csv(file_path, header=None)

    attributes = ['Time', 'Trader', 'Opinion', 'Input', 'Attention', 'Gamma', 'Delta']
    offsets = [1, 3, 6, 8, 10, 12, 14]

    data_list = []
    for _, row in data_df.iterrows():
        # Assuming 'Time' conversion from the raw data is correct
        time = float(row[1]) / (60 * 60 * 24)  # Convert time to days
        k = 3  # Adjust if starting index for the first data set is different
        while k < len(row) - 8:
            data_entry = [time]
            # Adjust the loop to collect each data field using the correct offsets
            for offset in offsets[1:]:  # Starting from 'Trader' to 'Delta'
                data_entry.append(row[k + offset - 3])
            data_list.append(data_entry)
            k += 13  # Move to the start of the next trader's data in the same row

    # Create a DataFrame from the collected data entries
    processed_df = pd.DataFrame(data_list, columns=attributes)
    

   

    processed_df['Attention'] = processed_df['Attention'].fillna(0)  # Fill NaN with 0
    # Determine color based on average opinion for the first two days for each trader
    two_days_data = processed_df[processed_df['Time'] <= processed_df['Time'].min() + 2]
    avg_opinions = two_days_data.groupby('Trader')['Opinion'].mean()
    colors = {trader: 'blue' if opinion > 0 else 'red' for trader, opinion in avg_opinions.items()}

    # Collecting Opinion Data
    for trader, color in colors.items():
        trader_data = processed_df[processed_df['Trader'] == trader]
        opinion_data.append((trader_data['Time'], trader_data['Opinion'], color))

    # Collecting Gamma and Delta Dynamics
    for trader_id in processed_df['Trader'].unique():
        trader_data = processed_df[processed_df['Trader'] == trader_id]
        gamma_data.append((trader_data['Time'], trader_data['Gamma'], 'purple'))
        delta_data.append((trader_data['Time'], trader_data['Delta'], 'orange'))

# Plotting Opinion Dynamics
plt.figure(figsize=(10, 6))
for time, opinion, color in opinion_data:
    plt.plot(time, opinion, color=color, alpha=0.5)

max_day = int(processed_df['Time'].max()) + 2
plt.xticks(list(range(max_day)), [str(i) for i in range(0, max_day)])
plt.ylim(-5, 5)
plt.ylabel('Opinion Dynamics', fontsize=16)
legend_handles_opinion = [
    mpatches.Patch(color='blue', label='Positive Group'),
    mpatches.Patch(color='red', label='Negative Group')
]
#plt.xlabel('Time (days)', fontsize=16)
plt.legend(handles=legend_handles_opinion,  fontsize =16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()

# Plotting Gamma and Delta Dynamics
plt.figure(figsize=(10, 4))
for time, value, color in gamma_data:
    plt.plot(time, value, color=color, alpha=0.5, label=r'$\gamma$' if r'$\gamma$' not in plt.gca().get_legend_handles_labels()[1] else "")
for time, value, color in delta_data:
    plt.plot(time, value, color=color, alpha=0.5, label=r'$\delta$' if r'$\delta$' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xticks(list(range(max_day)), [str(i) for i in range(0, max_day)])
#plt.xlabel('Time (days)', fontsize=16)
plt.ylabel(r'$\gamma$ and $\delta$ Dynamics', fontsize=16)
plt.legend( fontsize =16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()
