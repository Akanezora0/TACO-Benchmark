import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creating an updated DataFrame to hold the accuracy data for models in the required format
data_updated = {
    'Model': [
        'GPT-4', 'GPT-4o', 'GPT-o1', 'Llama3-70b', 'Qwen2-72b', 'Deepseek-v2.5',
        'MAC-SQL', 'DIN-SQL+GPT-4', 'codes-7b', 'qwen2.5 coder-7b',
        'granite-34b', 'deepseek coder6.7b', 'CHESS', 'DAIL-SQL+GPT-4',
        'Zero-NL2SQL', 'CHASE', 'OUR'
    ],
    '12345 Accuracy (%)': [
        None, None, None, 7.9, 9.2, 12.8, None, None, 16.9, None, None, None, None, None, None, None, None
    ],
    'Beijing Accuracy (%)': [
        32.6, 38.1, None, 21.9, 26.8, 32.4, 26.7, 29.0, 42.2, None, None, None,
        25.2, 39.0, None, None, 54.4
    ],
    'Chicago Accuracy (%)': [
        34.8, 37.6, None, 28.1, 24.7, 34.3, 31.2, 28.5, 45.7, None, 47.8, None,
        32.7, 41.9, None, None, 55.8
    ]
}

df_updated = pd.DataFrame(data_updated)

# Setting up the figure for plotting in the desired format
plt.figure(figsize=(16, 8))

# Extracting valid data (ignoring None values)
valid_data_updated = df_updated.dropna()

bar_width = 0.25
bar_positions = np.arange(len(df_updated))

# Plotting the updated bar chart with side-by-side comparison of three accuracies
for i, model in enumerate(df_updated['Model']):
    accuracy_12345 = df_updated.loc[df_updated['Model'] == model, '12345 Accuracy (%)'].values[0]
    chinese_accuracy = df_updated.loc[df_updated['Model'] == model, 'Beijing Accuracy (%)'].values[0]
    english_accuracy = df_updated.loc[df_updated['Model'] == model, 'Chicago Accuracy (%)'].values[0]
    
    if accuracy_12345 is not None:
        plt.bar(i - bar_width, accuracy_12345, width=bar_width, color='lightgreen', label='12345' if i == 0 else "")
    if chinese_accuracy is not None:
        plt.bar(i, chinese_accuracy, width=bar_width, color='skyblue', label='Beijing' if i == 0 else "")
    if english_accuracy is not None:
        plt.bar(i + bar_width, english_accuracy, width=bar_width, color='salmon', label='Chicago' if i == 0 else "")

# Adding labels and title
plt.xticks(bar_positions, df_updated['Model'], rotation=45, ha='right', fontsize=10)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy of Different Models on NL2SQL Task', fontsize=14)

# Adding legend
plt.legend()

# Adding gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Tight layout for better spacing
plt.tight_layout()

# Save the updated plot as an image for use in the paper
plt.savefig('../../data/visualizations/accuracy.png', dpi=300)

# Display the updated plot
plt.show()