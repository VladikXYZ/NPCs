import os
import pandas as pd
import numpy as np

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Model list
models = ["Gemma 4 E2B", "Gemma 4 E4B", "Llama 3.1", "Mistral Nemo", "Phi 4 mini", "Phi 4", "Qwen 3.5"]

LOG_DIR = 'bench_logs/'
systems = ["CACHY/", "NOTEBOOK/"]

for system in systems:
    # Ensure the directory exists before trying to read it
    if not os.path.exists(LOG_DIR + system):
        continue

    logs = os.listdir(LOG_DIR + system)
    for log in logs:
        # Load the CSV
        df = pd.read_csv(LOG_DIR + system + log)

        # Print a header so you know which hardware file this is
        print(f"\n{'=' * 50}")
        print(f"📊 HARDWARE: {system+log}")
        print(f"{'=' * 50}")

        # 1. Assign Model Names (Every 10 rows is a new model)
        model_indices = df.index // 10
        # Failsafe: just in case a log has too many rows, prevent an IndexError
        df['Model'] = [models[i] if i < len(models) else f"Unknown {i}" for i in model_indices]

        # 2. Assign Prompt Type (First 5 are Short, next 5 are Long)
        prompt_types = np.where((df.index % 10) < 5, 'Short Prompt', 'Long Prompt')
        df['Prompt Type'] = prompt_types

        # 3. Calculate Mean, Min, and Max for TTFT ONLY
        # We group by Model and Prompt Type, select the 'TTFT' column, and aggregate
        ttft_stats = df.groupby(['Model', 'Prompt Type'])['TTFT'].agg(['mean', 'min', 'max'])

        # Rename the columns to make them look nice
        ttft_stats.columns = ['Avg TTFT (s)', 'Min TTFT (s)', 'Max TTFT (s)']

        print(ttft_stats)