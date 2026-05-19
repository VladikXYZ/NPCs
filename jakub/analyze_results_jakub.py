import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ==============================================================================
# CONFIGURATION & STYLE DESIGN (Synced with tables.py)
# ==============================================================================
INPUT_FILE = r"evaluation_results.csv"

# Color Palette Variables
BG_COLOR = '#1e1e24'
TEXT_WHITE = 'white'
COLOR_FAILED = '#a50026'  # Deep Intense Red
COLOR_GOOD_ENOUGH = '#fee08b'  # Clean Pastel Yellow/Gold
COLOR_EXCELLENT = '#1a9850'  # Rich Emerald Green

TITLE_FONT_SIZE = 28
MATRIX_CELL_FONT_SIZE = 12
AXIS_TICK_FONT_SIZE = 20

GLOBAL_FONTS = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.sans-serif'] = GLOBAL_FONTS
plt.rcParams['axes.unicode_minus'] = False

# Strict Mapping & Axis Layout Structure
MODEL_NAME_MAP = {
    'gemma-4-E2B-it-Q4_K_M': 'Gemma 4 E2B',
    'gemma-4-E4B-it-Q4_K_M': 'Gemma 4 E4B',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M': 'Llama 3.1',
    'Mistral-Nemo-Instruct-2407.Q4_K_S': 'Mistral Nemo',
    'Phi-4-mini-instruct-Q4_K_M': 'Phi 4 mini',
    'phi-4-Q4_K_S': 'Phi 4',
    'Qwen3.5-9B-Q4_K_M': 'Qwen 3.5'
}
MODELS_ORDER = ["Gemma 4 E2B", "Gemma 4 E4B", "Llama 3.1", "Mistral Nemo", "Phi 4 mini", "Phi 4", "Qwen 3.5"]
TEST_TYPE_ORDER = ['baseline', 'collision', 'conflict', 'dissonance']


def create_16_9_heatmap(data, title, filename, cmap_colors, vmin, vmax, is_improvement=False):
    """Generates heatmaps explicitly constrained to a widescreen 16:9 aspect ratio."""
    # 16x9 aspect ratio size initialization (Width=16, Height=9)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Construct a continuous linear map across your signature hex choices
    cmap = LinearSegmentedColormap.from_list("custom_continuous_16_9", cmap_colors)

    # Generate the heatmap matrix
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=3.0,
        linecolor=BG_COLOR,
        cbar_kws={
            'label': 'Improvement Delta (Higher is Better)' if is_improvement else 'Average Grade (1=Best, 5=Worst)'
        },
        annot_kws={"size": MATRIX_CELL_FONT_SIZE, "weight": "bold"},
        ax=ax
    )

    # Recolor colorbar ticks and label text elements to white
    cbar = ax.collections[0].colorbar

    # 3. MAKE THE COLORBAR TEXT BIGGER & READABLE
    # Change colorbar side label font size and weight
    cbar.ax.yaxis.label.set_size(16)
    cbar.ax.yaxis.label.set_weight('bold')
    cbar.ax.yaxis.label.set_color(TEXT_WHITE)
    cbar.ax.yaxis.labelpad = 15

    # Change colorbar ticks numbers size (e.g., 1.0, 2.0, 3.0...)
    cbar.ax.tick_params(labelsize=14, color=TEXT_WHITE, labelcolor=TEXT_WHITE)

    # Title & Axis Text Formatting
    ax.set_title(title, color=TEXT_WHITE, fontsize=TITLE_FONT_SIZE, pad=35, weight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(colors=TEXT_WHITE, labelsize=AXIS_TICK_FONT_SIZE, which='both', length=0)
    ax.set_xticklabels([x.get_text().capitalize() for x in ax.get_xticklabels()], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Balanced layout spacing
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300, transparent=True)
    plt.close()
    print(f"🖼️ Exported 16:9 Aspect Ratio Heatmap: {filename}")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Target dataset file '{INPUT_FILE}' not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    df['Model_Clean'] = df['Model'].map(MODEL_NAME_MAP).fillna(df['Model'])

    # 1. Separate & Pivot No Reasoning Mode
    df_no = df[df['Reasoning_Mode'] == 'no_reasoning'].pivot_table(
        index='Model_Clean', columns='Test_Type', values='Grade', aggfunc='mean'
    ).reindex(index=MODELS_ORDER, columns=TEST_TYPE_ORDER)

    # 2. Separate & Pivot With Reasoning Mode
    df_with = df[df['Reasoning_Mode'] == 'with_reasoning'].pivot_table(
        index='Model_Clean', columns='Test_Type', values='Grade', aggfunc='mean'
    ).reindex(index=MODELS_ORDER, columns=TEST_TYPE_ORDER)

    # 3. Calculate Improvement Delta (Positive value means reasoning successfully dropped the score)
    df_imp = df_no - df_with

    # Establish Color Range Profiles
    grade_cmap = [COLOR_EXCELLENT, COLOR_GOOD_ENOUGH, COLOR_FAILED]  # Green (1) -> Yellow -> Red (5)
    imp_cmap = [COLOR_FAILED, COLOR_GOOD_ENOUGH, COLOR_EXCELLENT]  # Red (Worse) -> Yellow (0) -> Green (Better)

    # Render the 3 distinct 16:9 dashboard visualizations
    create_16_9_heatmap(df_no, "MODEL PERFORMANCE - NO REASONING", "heatmap_no_reasoning.png", grade_cmap,
                        vmin=1.0, vmax=5.0)
    create_16_9_heatmap(df_with, "MODEL PERFORMANCE - WITH REASONING", "heatmap_with_reasoning.png", grade_cmap,
                        vmin=1.0, vmax=5.0)
    create_16_9_heatmap(df_imp, "REASONING IMPROVEMENT DELTA PER TASK", "heatmap_reasoning_improvement.png",
                        imp_cmap, vmin=-1.7, vmax=1.7, is_improvement=True)


if __name__ == "__main__":
    main()