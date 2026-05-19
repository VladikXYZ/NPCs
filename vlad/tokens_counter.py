import json

# import pandas as pd
#
# # Load log file
# df = pd.read_csv("bench_logs/CACHY/Vulkan_AMD_Radeon_RX_9070.csv")
#
# # Classify based on the row positions (blocks of 5)
# df['Batch Type'] = df.index.map(
#     lambda idx: 'Short Batch (0-4, 10-14, ...)' if (idx % 10) < 5 else 'Long Batch (5-9, 15-19, ...)'
# )
#
# # Aggregate min, max, count, and performance averages
# comparison = df.groupby('Batch Type').agg(
#     Count=('USER TOKENS', 'count'),
#     Min_Tokens=('USER TOKENS', 'min'),
#     Max_Tokens=('USER TOKENS', 'max'),
#     Avg_Tokens=('USER TOKENS', 'mean'),
#     Avg_TTFT=('TTFT', 'mean'),
#     Avg_TPS=('T/S', 'mean')
# )
#
# print("--- Benchmark Cycle Analysis ---")
# print(comparison.to_string())
messages = json.load(open('test.json'))
for message in messages:
    print(len(message.split(" ")))