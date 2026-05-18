import os
import pandas as pd
import numpy as np

# Display settings to ensure the massive table prints cleanly
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)  # Made extra wide for the transposed columns

# Use the updated proper model names!
models = ["Gemma 4 E2B (2B)", "Gemma 4 E4B (4B)", "Llama 3.1 (8B)", "Mistral Nemo (12B)", "Phi 4 mini (3.8B)",
          "Phi 4 (14B)", "Qwen 3.5 (9B)"]

LOG_DIR = 'bench_logs/'
systems = ["CACHY/", "Jakub/"]

all_results = []

for system in systems:
    if not os.path.exists(LOG_DIR + system):
        continue

    logs = os.listdir(LOG_DIR + system)
    for log in logs:
        if not log.endswith('.csv'):
            continue

        df = pd.read_csv(LOG_DIR + system + log)

        # Clean up the hardware name (e.g., "CACHY | Vulkan_AMD_Radeon_RX_9070")
        hw_name = f"{system.replace('/', '')} | {log.replace('.csv', '')}"

        # 1. Assign Model Names
        model_indices = df.index // 10
        df['Model'] = [models[i] if i < len(models) else f"Unknown {i}" for i in model_indices]

        # 2. Assign Prompt Type
        df['Prompt Type'] = np.where((df.index % 10) < 5, 'Short', 'Long')
        df['Hardware'] = hw_name

        # 3. Calculate Mean for TTFT and flatten the index
        mean_ttft = df.groupby(['Hardware', 'Model', 'Prompt Type'])['TTFT'].mean().reset_index()
        all_results.append(mean_ttft)

if all_results:
    # 4. Combine all the hardware files into one massive dataframe
    combined_df = pd.concat(all_results, ignore_index=True)

    # 5. Create a combined string for our new column headers (e.g., "Gemma 4 E2B Short")
    combined_df['Condition'] = combined_df['Model'] + " " + combined_df['Prompt Type']

    # 6. PIVOT! Flip the table: Hardware as Rows, Conditions as Columns, TTFT means as Values
    pivot_df = combined_df.pivot(index='Hardware', columns='Condition', values='TTFT')

    # 7. Force the columns to sort exactly in the order of your 'models' list
    ordered_cols = []
    for m in models:
        ordered_cols.extend([f"{m} Short", f"{m} Long"])

    # Failsafe: Only keep columns that actually generated data
    ordered_cols = [col for col in ordered_cols if col in pivot_df.columns]
    pivot_df = pivot_df[ordered_cols]

    # 8. Print out the beautiful matrix!
    print(f"\n{'=' * 120}")
    print("🚀 MEAN TTFT (Seconds) ACROSS ALL HARDWARE")
    print(f"{'=' * 120}")

    # We use .round(3) to keep it readable, and fillna('-') in case a model crashed on a specific GPU
    print(pivot_df.round(3).fillna('-'))

    # Optional: Save this master matrix to a CSV for easy sharing/graphing!
    pivot_df.round(3).to_excel("Master_TTFT_Comparison.xlsx")
else:
    print("No CSV files found in bench_logs!")