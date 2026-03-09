import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.dates as mdates

# Provided trial_id and file path
trial_id = "bse_ICAART24_II_H2_RWMM_s01_d005_i05"
file_path = f'C:/Users/oc21380/PycharmProjects/Parameter_Calibration/H2/{trial_id}_opinions.csv'

# Load data
data_df = pd.read_csv(file_path, header=None)
attributes = ['Time', 'Trader', 'Opinion', 'Input', 'Attention', 'Gamma', 'Delta']
offsets = [1, 3, 6, 8, 10, 12, 14]

# Process data
data_list = []
for _, row in data_df.iterrows():
    time = float(row[1])
    scaled_time = np.datetime64('2018-02-01') + np.timedelta64(int(time/ 229), 'D')
    i = 3
    while i < len(row) - 8:
        data_entry = [scaled_time]
        for offset in offsets[1:]:
            data_entry.append(row[i + offset - 3])
        data_list.append(data_entry)
        i += 13

processed_df = pd.DataFrame(data_list, columns=attributes)

# First Figure for Opinion Dynamics
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def opinion_group(first_opinion):
    # Classify based on the first opinion
    if first_opinion > 0:
        return 'Positive'
    else:
        return 'Negative'

# Retrieve the first opinion for each trader
first_opinions = processed_df.groupby('Trader')['Opinion'].first()
trader_groups = first_opinions.apply(opinion_group).to_dict()

# Create custom legend for opinion visualization
legend_handles_opinion = [
    mpatches.Patch(color='red', label='Positive Narrative'),
    mpatches.Patch(color='blue', label='Negative Narrative')
]

# Plotting Opinion Dynamics
plt.figure(figsize=(14, 7))
colors = {'Positive': 'blue', 'Negative': 'red'}
for group, color in colors.items():
    for trader_id in [k for k, v in trader_groups.items() if v == group]:
        trader_data = processed_df[processed_df['Trader'] == trader_id]
        plt.plot(trader_data['Time'], trader_data['Opinion'], color=color, linestyle='-')

plt.legend(handles=legend_handles_opinion,fontsize=18)
plt.tick_params(axis='both', which='both', labelsize=18)
#plt.title('Opinion Dynamics by Trader Group')
#plt.xlabel('Time')
plt.ylabel('Opinion Dynamics', fontsize = 18 )
plt.savefig("H2_RWMM_OD.pdf", bbox_inches="tight") # zero inital conditions
plt.show()   # optional
plt.close()  # recommended


# Second Figure for Gamma and Delta Dynamics
plt.figure(figsize=(14, 7))
for trader_id in processed_df['Trader'].unique():
    trader_data = processed_df[processed_df['Trader'] == trader_id]
    plt.plot(trader_data['Time'], trader_data['Gamma'], 'purple', label=r'$\sigma$' if trader_id == processed_df['Trader'].unique()[0] else "")

#plt.xlabel('Year', fontsize=18)
plt.ylabel('$\sigma$', fontsize=18)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#plt.xticks(rotation=45)
plt.tick_params(axis='both', which='both', labelsize=18)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicate labels/handles
plt.legend(by_label.values(), by_label.keys(), fontsize=18)
plt.savefig("H2_RWMM_Sigma.pdf", bbox_inches="tight") # zero inital conditions
plt.show()   # optional
plt.close()  # recommended
