""" TEST """


# import os
# import pandas as pd
# import numpy as np
#
# # Display settings
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 1000)
#
# # Model list
# models = ["Gemma 4 E2B", "Gemma 4 E4B", "Llama 3.1", "Mistral Nemo", "Phi 4 mini", "Phi 4", "Qwen 3.5"]
#
# LOG_DIR = 'bench_logs/'
# systems = ["CACHY/", "NOTEBOOK/"]
#
# for system in systems:
#     # Ensure the directory exists before trying to read it
#     if not os.path.exists(LOG_DIR + system):
#         continue
#
#     logs = os.listdir(LOG_DIR + system)
#     for log in logs:
#         # Load the CSV
#         df = pd.read_csv(LOG_DIR + system + log)
#
#         # Print a header so you know which hardware file this is
#         print(f"\n{'=' * 50}")
#         print(f"📊 HARDWARE: {system+log}")
#         print(f"{'=' * 50}")
#
#         # 1. Assign Model Names (Every 10 rows is a new model)
#         model_indices = df.index // 10
#         # Failsafe: just in case a log has too many rows, prevent an IndexError
#         df['Model'] = [models[i] if i < len(models) else f"Unknown {i}" for i in model_indices]
#
#         # 2. Assign Prompt Type (First 5 are Short, next 5 are Long)
#         prompt_types = np.where((df.index % 10) < 5, 'Short Prompt', 'Long Prompt')
#         df['Prompt Type'] = prompt_types
#
#         # 3. Calculate Mean, Min, and Max for TTFT ONLY
#         # We group by Model and Prompt Type, select the 'TTFT' column, and aggregate
#         stats = df.groupby(['Model', 'Prompt Type'])[['TTFT', 'T/S']].agg(['mean', 'min', 'max'])
#
#         # When you do multiple columns, Pandas creates a "MultiIndex" (stacked headers).
#         # We can flatten them out into a single row of nice, clean column names:
#         stats.columns = [
#             'Avg TTFT (s)', 'Min TTFT (s)', 'Max TTFT (s)',
#             'Avg T/S', 'Min T/S', 'Max T/S'
#         ]
#
#         print(stats)


""" TTFT """

# import os
# import pandas as pd
# import numpy as np
#
# # Display settings to ensure the massive table prints cleanly
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 2000)  # Made extra wide for the transposed columns
#
# # Use the updated proper model names!
# models = ["Gemma 4 E2B (2B)", "Gemma 4 E4B (4B)", "Llama 3.1 (8B)", "Mistral Nemo (12B)", "Phi 4 mini (3.8B)",
#           "Phi 4 (14B)", "Qwen 3.5 (9B)"]
#
# LOG_DIR = 'bench_logs/'
# systems = ["CACHY/", "Jakub/"]
#
# all_results = []
#
# for system in systems:
#     if not os.path.exists(LOG_DIR + system):
#         continue
#
#     logs = os.listdir(LOG_DIR + system)
#     for log in logs:
#         if not log.endswith('.csv'):
#             continue
#
#         df = pd.read_csv(LOG_DIR + system + log)
#
#         # Clean up the hardware name (e.g., "CACHY | Vulkan_AMD_Radeon_RX_9070")
#         hw_name = f"{system.replace('/', '')} | {log.replace('.csv', '')}"
#
#         # 1. Assign Model Names
#         model_indices = df.index // 10
#         df['Model'] = [models[i] if i < len(models) else f"Unknown {i}" for i in model_indices]
#
#         # 2. Assign Prompt Type
#         df['Prompt Type'] = np.where((df.index % 10) < 5, 'Short', 'Long')
#         df['Hardware'] = hw_name
#
#         # 3. Calculate Mean for TTFT and flatten the index
#         mean_ttft = df.groupby(['Hardware', 'Model', 'Prompt Type'])['TTFT'].mean().reset_index()
#         all_results.append(mean_ttft)
#
# if all_results:
#     # 4. Combine all the hardware files into one massive dataframe
#     combined_df = pd.concat(all_results, ignore_index=True)
#
#     # 5. Create a combined string for our new column headers (e.g., "Gemma 4 E2B Short")
#     combined_df['Condition'] = combined_df['Model'] + " " + combined_df['Prompt Type']
#
#     # 6. PIVOT! Flip the table: Hardware as Rows, Conditions as Columns, TTFT means as Values
#     pivot_df = combined_df.pivot(index='Hardware', columns='Condition', values='TTFT')
#
#     # 7. Force the columns to sort exactly in the order of your 'models' list
#     ordered_cols = []
#     for m in models:
#         ordered_cols.extend([f"{m} Short", f"{m} Long"])
#
#     # Failsafe: Only keep columns that actually generated data
#     ordered_cols = [col for col in ordered_cols if col in pivot_df.columns]
#     pivot_df = pivot_df[ordered_cols]
#
#     # 8. Print out the beautiful matrix!
#     print(f"\n{'=' * 120}")
#     print("🚀 MEAN TTFT (Seconds) ACROSS ALL HARDWARE")
#     print(f"{'=' * 120}")
#
#     # We use .round(3) to keep it readable, and fillna('-') in case a model crashed on a specific GPU
#     print(pivot_df.round(3).fillna('-'))
#
#     # Optional: Save this master matrix to a CSV for easy sharing/graphing!
#     pivot_df.round(3).to_excel("Master_TTFT_Comparison.xlsx")
# else:
#     print("No CSV files found in bench_logs!")




""" TS """

# import os
# import pandas as pd
#
# # Display settings to ensure the massive table prints cleanly
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 2000)
#
# models = ["Gemma 4 E2B (2B)", "Gemma 4 E4B (4B)", "Llama 3.1 (8B)", "Mistral Nemo (12B)", "Phi 4 mini (3.8B)",
#           "Phi 4 (14B)", "Qwen 3.5 (9B)"]
#
# LOG_DIR = 'bench_logs/'
# systems = ["CACHY/", "Jakub/"]
#
# all_results = []
#
# for system in systems:
#     if not os.path.exists(LOG_DIR + system):
#         continue
#
#     logs = os.listdir(LOG_DIR + system)
#     for log in logs:
#         if not log.endswith('.csv'):
#             continue
#
#         df = pd.read_csv(LOG_DIR + system + log)
#
#         # Clean up the hardware name
#         hw_name = f"{system.replace('/', '')} | {log.replace('.csv', '')}"
#
#         # 1. Assign Model Names (10 rows per model)
#         model_indices = df.index // 10
#         df['Model'] = [models[i] if i < len(models) else f"Unknown {i}" for i in model_indices]
#         df['Hardware'] = hw_name
#
#         # 2. Safely parse T/S (fixes commas and 'ERROR' strings)
#         if 'T/S' in df.columns:
#             df['T/S'] = df['T/S'].astype(str).str.replace(',', '.')
#             df['T/S'] = pd.to_numeric(df['T/S'], errors='coerce')
#
#             # 3. Calculate Mean for T/S, grouped ONLY by Hardware and Model
#             mean_ts = df.groupby(['Hardware', 'Model'])['T/S'].mean().reset_index()
#             all_results.append(mean_ts)
#
# if all_results:
#     # 4. Combine all the hardware files into one massive dataframe
#     combined_df = pd.concat(all_results, ignore_index=True)
#
#     # 5. PIVOT! Flip the table: Hardware as Rows, Models as Columns, T/S means as Values
#     pivot_df = combined_df.pivot(index='Hardware', columns='Model', values='T/S')
#
#     # 6. Keep the ORIGINAL model order!
#     # Failsafe: Only keep columns that actually generated data
#     ordered_cols = [col for col in models if col in pivot_df.columns]
#     pivot_df = pivot_df[ordered_cols]
#
#     # 7. Print out the beautiful matrix!
#     print(f"\n{'=' * 120}")
#     print("🚀 OVERALL MEAN T/S (Tokens Per Second)")
#     print(f"{'=' * 120}")
#
#     # Print to terminal
#     pivot_df = pivot_df.iloc[[1, 3, 6, 0, 4, 2, 5]]
#     print(pivot_df.round(3).fillna('-'))
#
#
#
#     pivot_df.to_excel("Master_TS_Comparison.xlsx")
#     print("\n✅ Saved color-coded Excel file (Master_TS_Comparison.xlsx)!")
# else:
#     print("No CSV files found in bench_logs!")




""" TOKENS """


import pandas as pd
import numpy as np

# 1. Load your benchmark file
file_name = 'bench_logs/CACHY/tokens.csv'
# file_name = 'vlad/bench_logs/CACHY/Vulkan_AMD_Radeon_RX_9070.csv'
df = pd.read_csv(file_name)

# Clean up hidden spaces in headers
df.columns = df.columns.str.strip()

# 2. Convert USER TOKENS column to numeric safely (handling commas/errors)
if 'USER TOKENS' in df.columns:
    df['USER TOKENS'] = df['USER TOKENS'].astype(str).str.replace(',', '.')
    df['USER TOKENS'] = pd.to_numeric(df['USER TOKENS'], errors='coerce')

# 3. Filter out benchmark error codes (-1) so they don't break the Min or Mean
df = df.replace(-1.0, np.nan).replace(-1, np.nan)

# 4. Map rows to Short and Long categories (5 rows each in a 10-row loop)
df['Prompt Type'] = np.where((df.index % 10) < 5, 'Short', 'Long')

# 5. Calculate mean, min, and max strictly for USER TOKENS
stats = df.groupby('Prompt Type')['USER TOKENS'].agg(['mean', 'min', 'max'])

# Give the columns clean headers
stats.columns = ['Avg User Tokens', 'Min User Tokens', 'Max User Tokens']

# 6. Print out the clean, focused table
print(f"\n📊 USER TOKEN SIZES BY PROMPT TYPE ({file_name})")
print("=" * 60)
print(stats.round(1).to_string())
print("=" * 60)