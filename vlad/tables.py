import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==============================================================================
# CONFIGURATION & GLOBAL CONSTANTS
# ==============================================================================
LOG_DIR = 'bench_logs/'
systems = ["CACHY/", "Jakub/"]

models_list = [
    "Gemma 4 E2B", "Gemma 4 E4B", "Llama 3.1",
    "Mistral Nemo", "Phi 4 mini", "Phi 4", "Qwen 3.5"
]

# Clean Hardware Name Mapping
HARDWARE_MAP = {
    "CACHY\nVulkan_AMD_Radeon_RX_9070": "RX 9070 XT",
    "CACHY\nVulkan_NVIDIA_GeForce_RTX_3060": "RTX 3060",
    "Jakub\nVulkan_NVIDIA_GeForce_RTX_4070": "RTX 4070 (L)",
    "CACHY\nVulkan_AMD_Ryzen_5_7600X": "R5 7600X iGPU",
    "Jakub\nVulkan_Intel(R)_RaptorLake-S_Mobile_Graphics": "i9 14900HX iGPU",
    "CACHY\nCPU_CPU": "R5 7600X",
    "Jakub\nCPU_CPU": "i9 14900HX"
}

# Presentation Color Schemes (Cyberpunk Dark Mode Matching Martin's Script)
BG_COLOR = '#1e1e24'
TEXT_WHITE = 'white'
COLOR_CRASH_BLACK = '#111115'  # Pure pitch-black block for crashed models

# Strict 3-color map values:
COLOR_FAILED = '#a50026'  # Deep Intense Red
COLOR_GOOD_ENOUGH = '#fee08b'  # Clean Pastel Yellow/Gold
COLOR_EXCELLENT = '#1a9850'  # Rich Emerald Green

# Typography Engine
TITLE_FONT_SIZE = 28
MATRIX_CELL_FONT_SIZE = 12
AXIS_TICK_FONT_SIZE = 20

# Unified Font Fallbacks
GLOBAL_FONTS = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.sans-serif'] = GLOBAL_FONTS
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# DATA PROCESSING & AGGREGATION
# ==============================================================================
all_results = []

for system in systems:
    if not os.path.exists(LOG_DIR + system):
        continue
    for log in os.listdir(LOG_DIR + system):
        if not log.endswith('.csv'):
            continue

        df = pd.read_csv(LOG_DIR + system + log)
        df.columns = df.columns.str.strip()

        raw_hw_name = f"{system.replace('/', '')}\n{log.replace('.csv', '')}"

        df['Model'] = [models_list[i] if i < len(models_list) else f"Unknown {i}" for i in (df.index // 10)]
        df['Prompt Type'] = np.where((df.index % 10) < 5, 'Short', 'Long')
        df['Hardware'] = raw_hw_name

        for col in ['TTFT', 'T/S']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'TTFT' in df.columns and 'T/S' in df.columns:
            means = df.groupby(['Hardware', 'Model', 'Prompt Type'])[['TTFT', 'T/S']].mean().reset_index()
            all_results.append(means)

if not all_results:
    print("No results processed.")
    exit()

combined_df = pd.concat(all_results, ignore_index=True)

# 1. GENERATE BASE DATA MATRICES
matrix_short_ttft = combined_df[combined_df['Prompt Type'] == 'Short'].pivot(index='Hardware', columns='Model',
                                                                             values='TTFT').reindex(columns=models_list)
matrix_long_ttft = combined_df[combined_df['Prompt Type'] == 'Long'].pivot(index='Hardware', columns='Model',
                                                                           values='TTFT').reindex(columns=models_list)
matrix_overall_ts = combined_df.groupby(['Hardware', 'Model'])['T/S'].mean().unstack().reindex(columns=models_list)

# 2. EVALUATE COMBINED STATUS MATRIX
mean_combined = combined_df.groupby(['Hardware', 'Model'])[['TTFT', 'T/S']].mean().reset_index()


def evaluate_status(row):
    if pd.isna(row['TTFT']) or pd.isna(row['T/S']) or row['TTFT'] < 0 or row['T/S'] < 0:
        return 'ERROR'
    if row['TTFT'] <= 0.3 and row['T/S'] >= 18: return 'EX'
    if row['TTFT'] <= 0.8 and row['T/S'] >= 6: return 'GE'
    return 'F'


mean_combined['Status_Text'] = mean_combined.apply(evaluate_status, axis=1)

status_mapping = {'ERROR': 0, 'F': 1, 'GE': 2, 'EX': 3}
mean_combined['Status_Num'] = mean_combined['Status_Text'].map(status_mapping)

matrix_status_text = mean_combined.pivot(index='Hardware', columns='Model', values='Status_Text').reindex(
    columns=models_list).fillna('ERROR')
matrix_status_num = mean_combined.pivot(index='Hardware', columns='Model', values='Status_Num').reindex(
    columns=models_list).fillna(0)


# ==============================================================================
# UNIFIED FORMATTING ENGINE
# ==============================================================================
def apply_formatting(df):
    df = df.iloc[[1, 3, 6, 0, 4, 2, 5]]
    df.index = [HARDWARE_MAP.get(name, name) for name in df.index]
    return df.T


matrix_short_ttft = apply_formatting(matrix_short_ttft)
matrix_long_ttft = apply_formatting(matrix_long_ttft)
matrix_overall_ts = apply_formatting(matrix_overall_ts)
matrix_status_text = apply_formatting(matrix_status_text)
matrix_status_num = apply_formatting(matrix_status_num)


# ==============================================================================
# WIDESCREEN HEATMAP GENERATOR WITH STRICT THRESHOLDS
# ==============================================================================
def draw_heatmap(data_matrix, title, save_filename, metric_type, float_fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Convert errors to -1.0 so we can catch them cleanly
    plot_matrix = data_matrix.fillna(-1.0)

    # Extract valid max so our bounds scale safely
    valid_mask = plot_matrix >= 0
    v_max = plot_matrix[valid_mask].max().max() if plot_matrix[valid_mask].size > 0 else 1.0

    # Apply specific hardcoded bounds & palettes based on the chart metric
    if metric_type == "status":
        cmap = ListedColormap([COLOR_CRASH_BLACK, COLOR_FAILED, COLOR_GOOD_ENOUGH, COLOR_EXCELLENT])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        annot_matrix = matrix_status_text.values

    elif metric_type == "ttft":
        # TTFT: <= 0.3 (Green), <= 0.8 (Yellow), > 0.8 (Red)
        cmap = ListedColormap([COLOR_EXCELLENT, COLOR_GOOD_ENOUGH, COLOR_FAILED])
        cmap.set_under(COLOR_CRASH_BLACK)  # Handle errors
        upper_bound = max(v_max + 1, 2.0)
        norm = BoundaryNorm([0.0, 0.301, 0.801, upper_bound], cmap.N)

    elif metric_type == "ts":
        # T/S: < 6 (Red), < 18 (Yellow), >= 18 (Green)
        cmap = ListedColormap([COLOR_FAILED, COLOR_GOOD_ENOUGH, COLOR_EXCELLENT])
        cmap.set_under(COLOR_CRASH_BLACK)  # Handle errors
        upper_bound = max(v_max + 1, 25.0)
        norm = BoundaryNorm([0.0, 6.0, 18.0, upper_bound], cmap.N)

    # Build the precise string formatting matrix for TTFT and TS
    if metric_type != "status":
        annot_matrix = np.empty_like(plot_matrix.values, dtype=object)
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                val = plot_matrix.iloc[i, j]
                annot_matrix[i, j] = "ERROR" if val < 0 else float_fmt.format(val)

    # Render natively with absolute boundary normalization
    sns.heatmap(
        plot_matrix, cmap=cmap, norm=norm, annot=annot_matrix, fmt='', cbar=False,
        linewidths=3, linecolor='#111115', annot_kws={"size": MATRIX_CELL_FONT_SIZE, "weight": "bold"}, ax=ax
    )

    # Dynamically inject high-contrast text color logic
    for text_el in ax.texts:
        text_str = text_el.get_text()
        if text_str == 'ERROR':
            text_el.set_color('#ff4d6d')  # Neon Red Text for Errors
        elif text_str == 'GE':
            text_el.set_color('black')
        elif text_str in ['EX', 'F']:
            text_el.set_color('white')
        else:
            try:
                val = float(text_str)
                # Ensure black text is printed when cell color is Yellow
                if metric_type == "ttft" and 0.301 <= val < 0.801:
                    text_el.set_color('black')
                elif metric_type == "ts" and 6.0 <= val < 18.0:
                    text_el.set_color('black')
                else:
                    text_el.set_color('white')
            except ValueError:
                text_el.set_color('white')

    # Title & Axis Styling
    ax.set_title(title, color=TEXT_WHITE, fontsize=TITLE_FONT_SIZE, pad=35, weight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(colors=TEXT_WHITE, labelsize=AXIS_TICK_FONT_SIZE, which='both', length=0)

    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha('right')
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)

    plt.subplots_adjust(left=0.18, bottom=0.22)
    plt.tight_layout()
    plt.savefig(save_filename, format='png', dpi=300, transparent=True)
    plt.close()
    print(f"🖼️ Exported Custom Bound Matrix: {save_filename}")


# ==============================================================================
# EXECUTION COMMAND BLOCK
# ==============================================================================
draw_heatmap(matrix_short_ttft, "MEAN TTFT (SECONDS) - SHORT PROMPTS", "heatmap_short_ttft.png", metric_type="ttft",
             float_fmt="{:.3f}")
draw_heatmap(matrix_long_ttft, "MEAN TTFT (SECONDS) - LONG PROMPTS", "heatmap_long_ttft.png", metric_type="ttft",
             float_fmt="{:.3f}")
draw_heatmap(matrix_overall_ts, "MEAN TEXT GENERATION SPEED (TOKENS/SEC)", "heatmap_tokens_sec.png", metric_type="ts",
             float_fmt="{:.1f}")
draw_heatmap(matrix_status_num, "HARDWARE EVALUATION MATRIX", "heatmap_3color_evaluation.png", metric_type="status")

print("\n🎉 Matrices rendered with absolute threshold bounds! xd")