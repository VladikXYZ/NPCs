import os

import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)  # Forces it to show all columns
pd.set_option('display.width', 1000)
MODEL_DIR = '../models/'
# Load your future log file (which will have 20 rows per model)
df = pd.read_csv('bench_logs/CACHY/Vulkan_AMD_Radeon_RX_9070.csv')

# 1. Figure out which Model it is (Every 20 rows is a new model)
models = sorted([os.path.basename(x)[:-5] for x in os.listdir(MODEL_DIR) if x.endswith(".gguf")])
model_indices = df.index // 20
df['Model'] = [models[i] for i in model_indices]

# 2. Figure out if it is a Short or Long prompt
# df.index % 20 gives a repeating sequence from 0 to 19.
# If it is less than 10, it's the first half (Short). Otherwise, it's Long.
prompt_types = np.where((df.index % 20) < 10, 'Short Prompt', 'Long Prompt')
df['Prompt Type'] = prompt_types

# 3. Group by BOTH the Model and the Prompt Type, then calculate the average!
# We drop the 'USER PROMPT' and 'AI RESPONSE' text columns before averaging
numeric_cols = df.drop(columns=['USER PROMPT', 'AI RESPONSE'], errors='ignore')
avg_df = numeric_cols.groupby(['Model', 'Prompt Type']).mean()

# Print it nicely
print(avg_df)

# Save the beautifully formatted split averages to a new Excel/CSV file!
# avg_df.to_csv('Vulkan_AMD_Radeon_RX_9070_Detailed_Averages.csv')