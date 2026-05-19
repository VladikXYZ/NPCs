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
#
#     # 2. Duplicate each column name next to itself
#     duplicated_cols = []
#     for col in ordered_cols:
#         duplicated_cols.extend([col, col])  # Adds ['Model A', 'Model A', 'Model B', 'Model B'...]
#
#     # 3. Apply it to your DataFrame
#     pivot_df = pivot_df[duplicated_cols]
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

# # 1. Load your benchmark file
# file_name = 'bench_logs/CACHY/tokens.csv'
# # file_name = 'vlad/bench_logs/CACHY/Vulkan_AMD_Radeon_RX_9070.csv'
# df = pd.read_csv(file_name)
#
# # Clean up hidden spaces in headers
# df.columns = df.columns.str.strip()
#
# # 2. Convert USER TOKENS column to numeric safely (handling commas/errors)
# if 'USER TOKENS' in df.columns:
#     df['USER TOKENS'] = df['USER TOKENS'].astype(str).str.replace(',', '.')
#     df['USER TOKENS'] = pd.to_numeric(df['USER TOKENS'], errors='coerce')
#
# # 3. Filter out benchmark error codes (-1) so they don't break the Min or Mean
# df = df.replace(-1.0, np.nan).replace(-1, np.nan)
#
# # 4. Map rows to Short and Long categories (5 rows each in a 10-row loop)
# df['Prompt Type'] = np.where((df.index % 10) < 5, 'Short', 'Long')
#
# # 5. Calculate mean, min, and max strictly for USER TOKENS
# stats = df.groupby('Prompt Type')['USER TOKENS'].agg(['mean', 'min', 'max'])
#
# # Give the columns clean headers
# stats.columns = ['Avg User Tokens', 'Min User Tokens', 'Max User Tokens']
#
# # 6. Print out the clean, focused table
# print(f"\n📊 USER TOKEN SIZES BY PROMPT TYPE ({file_name})")
# print("=" * 60)
# print(stats.round(1).to_string())
# print("=" * 60)

import os
import pandas as pd
import numpy as np

# Display settings to ensure the tables print cleanly in your terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

# The master model list in your preferred default sorting order
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

        # Load the CSV
        df = pd.read_csv(LOG_DIR + system + log)

        # Clean up any trailing hidden spaces in headers
        df.columns = df.columns.str.strip()

        # Create a clean hardware identifier name
        hw_name = f"{system.replace('/', '')} | {log.replace('.csv', '')}"

        # 1. Assign Model Names (10 rows per model)
        model_indices = df.index // 10
        df['Model'] = [models[i] if i < len(models) else f"Unknown {i}" for i in model_indices]
        df['Hardware'] = hw_name

        # 2. Safely parse BOTH performance columns (swaps commas to dots, handles 'ERROR' text)
        for col in ['TTFT', 'T/S']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Filter out benchmark error codes (-1) so they don't break the Min or Mean
        df = df.replace(-1.0, np.nan).replace(-1, np.nan)

        # 4. Calculate Mean for both TTFT and T/S, grouped strictly by Hardware and Model
        if 'TTFT' in df.columns and 'T/S' in df.columns:
            mean_stats = df.groupby(['Hardware', 'Model'])[['TTFT', 'T/S']].mean().reset_index()
            all_results.append(mean_stats)

if all_results:
    # 5. Combine all processed hardware chunks into one master dataframe
    combined_df = pd.concat(all_results, ignore_index=True)


    # 6. Apply your grading logic to the model averages!
    def calculate_status(row):
        ttft = row['TTFT']
        ts = row['T/S']

        # If the model crashed or has empty data, it's an automatic failure
        if pd.isna(ttft) or pd.isna(ts):
            print("fail")
            return 'FAILED'

        # Your specific dual-conditional logic:
        if ttft <= 0.3 and ts >= 18:
            return 'EX'
        elif ttft <= 0.8 and ts >= 6:
            return 'GE'
        else:
            return 'F'


    combined_df['Status'] = combined_df.apply(calculate_status, axis=1)

    # 7. PIVOT! Build separate row/column matrices for each metric and status
    pivot_ttft = combined_df.pivot(index='Hardware', columns='Model', values='TTFT')
    pivot_ts = combined_df.pivot(index='Hardware', columns='Model', values='T/S')
    pivot_status = combined_df.pivot(index='Hardware', columns='Model', values='Status')

    # 8. Restructure columns back into your preferred 'models' list sequence
    ordered_cols = [col for col in models if col in pivot_ttft.columns]
    pivot_ttft = pivot_ttft[ordered_cols]
    pivot_ts = pivot_ts[ordered_cols]
    pivot_status = pivot_status[ordered_cols]

    # 9. PRINT THE TTFT TABLE
    print(f"\n{'=' * 120}")
    print("🚀 MEAN TTFT (Seconds) - SHORT & LONG COMBINED")
    print(f"{'=' * 120}")
    print(pivot_ttft.round(3).fillna('-'))

    # 10. PRINT THE TOKENS PER SECOND TABLE
    print(f"\n{'=' * 120}")
    print("🚀 MEAN T/S (Tokens Per Second) - SHORT & LONG COMBINED")
    print(f"{'=' * 120}")
    print(pivot_ts.round(2).fillna('-'))

    # 11. PRINT THE STATUS MATRIX TABLE
    pivot_status = pivot_status.iloc[[1, 3, 6, 0, 4, 2, 5]]
    print(f"\n{'=' * 120}")
    print("🏆 FINAL STATUS EVALUATION MATRIX")
    print(f"{'=' * 120}")
    print(pivot_status.fillna('FAILED'))

    # 12. SAVE ALL SHEETS INTO A SINGLE EXCEL WORKBOOK

    with pd.ExcelWriter("Master_Combined_Comparison.xlsx") as writer:
        pivot_ttft.round(3).to_excel(writer, sheet_name="Average_TTFT")
        pivot_ts.round(2).to_excel(writer, sheet_name="Average_TS")
        pivot_status.to_excel(writer, sheet_name="Model_Status")

    print("\n✅ Successfully generated all tables and saved to: Master_Combined_Comparison.xlsx")
else:
    print("No valid CSV benchmarks found in the target folders!")