import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==============================================================================
# 🎛️ GLOBAL CONFIGURATION & TYPOGRAPHY
# ==============================================================================
# Master Ordering (Alphabetical) applied to rows across ALL charts globally!
MODELS_ORDER = [
    "Gemma 4 E2B", "Gemma 4 E4B", "Llama 3.1",
    "Mistral Nemo", "Phi 4 mini", "Phi 4", "Qwen 3.5"
]

MARTIN_ATTACK_ORDER = [
    "Context size (CTX window)",
    "Experimental (Time shift)",
    "Long multiturn (10+ turns)",
    "Semantic (Blackmail)",
    "Overload (Attention span)",
    "Privilege (Authority)",
    "Syntactic (Word Play)",
    "Psychological (Good/Bad Cop)",
    "Domain/Language shift",
    "Encoding (Malicious HEX)"
]

TITLE_FONT_SIZE = 28
AXIS_TICK_FONT_SIZE = 24
MATRIX_CELL_FONT_SIZE = 20
BAR_DATA_LABEL_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 24
LEGEND_FONT_SIZE = 24
DPI = 500

IMG_DIR = "report"
os.makedirs(IMG_DIR, exist_ok=True)

BG_COLOR = '#1e1e24'
TEXT_WHITE = 'black'
COLOR_CRASH_BLACK = '#111115'

COLOR_FAILED = '#a50026'  # Deep Intense Red
COLOR_GOOD_ENOUGH = '#fee08b'  # Clean Pastel Yellow/Gold
COLOR_EXCELLENT = '#1a9850'  # Rich Emerald Green

# Grouped Bar Chart specific colors
COLOR_BAR_BASE = '#3a86ff'  # Bright Cyber Blue (No Reasoning)
COLOR_BAR_UPGRADE = '#38b000'  # Neon Green (With Reasoning)

GLOBAL_FONTS = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.sans-serif'] = GLOBAL_FONTS
plt.rcParams['axes.unicode_minus'] = False

HARDWARE_MAP = {
    "CACHY\nVulkan_AMD_Radeon_RX_9070": "RX 9070 XT",
    "CACHY\nVulkan_NVIDIA_GeForce_RTX_3060": "RTX 3060",
    "Jakub\nVulkan_NVIDIA_GeForce_RTX_4070": "RTX 4070 (L)",
    "CACHY\nVulkan_AMD_Ryzen_5_7600X": "R5 7600X iGPU",
    "Jakub\nVulkan_Intel(R)_RaptorLake-S_Mobile_Graphics": "i9 14900HX iGPU",
    "CACHY\nCPU_CPU": "R5 7600X",
    "Jakub\nCPU_CPU": "i9 14900HX"
}


# ==============================================================================
# 🛠️ HELPER FUNCTIONS
# ==============================================================================
def _clean_model_names(df, column_name):
    """Aggressively fuzzy-matches raw CSV model names to the strict MODELS_ORDER."""

    def map_name(raw):
        val = str(raw).lower()
        if 'gemma' in val and '2b' in val: return "Gemma 4 E2B"
        if 'gemma' in val and '4b' in val: return "Gemma 4 E4B"
        if 'llama' in val: return "Llama 3.1"
        if 'nemo' in val: return "Mistral Nemo"
        if 'phi' in val and 'mini' in val: return "Phi 4 mini"
        if 'phi' in val and 'mini' not in val: return "Phi 4"
        if 'qwen' in val: return "Qwen 3.5"
        return raw

    if column_name in df.columns:
        df[column_name] = df[column_name].apply(map_name)
    return df


def _load_data(data):
    if isinstance(data, str):
        return pd.read_csv(data)
    return data.copy()


def _format_perf_matrix(df_pivot):
    """Permutes hardware rows, translates names, and transposes models to rows."""
    if len(df_pivot) >= 7:
        try:
            df_pivot = df_pivot.iloc[[1, 3, 6, 0, 4, 2, 5]]
        except IndexError:
            pass
    df_pivot.index = [HARDWARE_MAP.get(name, name) for name in df_pivot.index]
    return df_pivot.T


# ==============================================================================
# 🖼️ CORE RENDERING ENGINES
# ==============================================================================
def draw_heatmap(data_matrix, title, save_filename, mode="continuous",
                 float_fmt="{:.2f}", custom_bounds=None, reverse_cmap=False,
                 vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    if mode == "status":
        cmap = ListedColormap([COLOR_CRASH_BLACK, COLOR_FAILED, COLOR_GOOD_ENOUGH, COLOR_EXCELLENT])
        status_map = {'ERROR': 0, 'F': 1, 'GE': 2, 'EX': 3}
        numeric_matrix = data_matrix.replace(status_map).fillna(0).astype(float)

        # 1. Map the cell abbreviations to full words for display labels
        display_map = {'ERROR': 'ERROR', 'F': 'FAILED', 'GE': 'PASS', 'EX': 'IDEAL'}
        annot_matrix = data_matrix.replace(display_map)

        # 2. Draw the heatmap with full text annotations and no sidebar
        sns.heatmap(
            numeric_matrix,
            cmap=cmap,
            annot=annot_matrix,  # <--- Forces FAILED, GOOD ENOUGH, IDEAL inside cells
            fmt='',
            cbar=False,  # <--- Keeps sidebar completely hidden
            vmin=0, vmax=3,
            linewidths=3, linecolor='#111115',
            annot_kws={"size": MATRIX_CELL_FONT_SIZE, "weight": "bold"},
            ax=ax
        )

        # 3. Match text font contrast against the new full words
        for text_el in ax.texts:
            t = text_el.get_text()
            if t == 'ERROR':
                text_el.set_color('#ff4d6d')  # Pink-red highlight for crashes
            elif t == 'PASS':
                text_el.set_color('black')  # Crisp black text for yellow boxes
            else:
                text_el.set_color('white')  # Clean white text for Green and Red boxes

        ax.set_aspect('auto')

        ax.set_title(title, color=TEXT_WHITE, fontsize=TITLE_FONT_SIZE, pad=35, weight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")

    elif mode == "threshold":
        # Changed sentinel to -999.0 so legitimate negative numbers aren't caught!
        plot_matrix = data_matrix.fillna(-999.0)
        valid_mask = plot_matrix > -999.0
        max_val = plot_matrix[valid_mask].max().max() if plot_matrix[valid_mask].size > 0 else 1.0

        upper_bound = max(max_val + 1, custom_bounds[-1])
        bounds = custom_bounds[:-1] + [upper_bound]

        cmap = ListedColormap(
            [COLOR_EXCELLENT, COLOR_GOOD_ENOUGH, COLOR_FAILED] if reverse_cmap else [COLOR_FAILED, COLOR_GOOD_ENOUGH,
                                                                                     COLOR_EXCELLENT])
        cmap.set_under(COLOR_CRASH_BLACK)  # Anything below 0.0 (like -999.0) turns Pitch Black
        norm = BoundaryNorm(bounds, cmap.N)

        annot_matrix = np.empty_like(plot_matrix.values, dtype=object)
        # print(plot_matrix)
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                val = plot_matrix.iloc[i, j]
                annot_matrix[i, j] = "ERROR" if val == -1.0 else float_fmt.format(val)

        sns.heatmap(
            plot_matrix, cmap=cmap, norm=norm, annot=annot_matrix, fmt='', cbar=False,
            linewidths=3, linecolor='#111115', annot_kws={"size": MATRIX_CELL_FONT_SIZE, "weight": "bold"}, ax=ax
        )

        for text_el in ax.texts:
            text_str = text_el.get_text()
            if text_str == 'ERROR':
                text_el.set_color('#ff4d6d')
            else:
                try:
                    val = float(text_str.replace('%', ''))
                    if bounds[1] <= val < bounds[2]:
                        text_el.set_color('black')
                    else:
                        text_el.set_color('white')
                except ValueError:
                    text_el.set_color('white')

    else:
        # Changed sentinel to -999.0 so legitimate negative numbers pass through safely!
        plot_matrix = data_matrix.fillna(-999.0)
        valid_mask = plot_matrix > -999.0

        calc_vmin = vmin if vmin is not None else (
            plot_matrix[valid_mask].min().min() if plot_matrix[valid_mask].size > 0 else 0)
        calc_vmax = vmax if vmax is not None else (
            plot_matrix[valid_mask].max().max() if plot_matrix[valid_mask].size > 0 else 1)

        annot_matrix = np.empty_like(plot_matrix.values, dtype=object)
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                val = plot_matrix.iloc[i, j]
                annot_matrix[i, j] = "ERROR" if val <= -999.0 else float_fmt.format(val)

        cmap_name = 'RdYlGn_r' if reverse_cmap else 'RdYlGn'
        cmap = plt.get_cmap(cmap_name).copy()
        # Matplotlib handles the magic: -999.0 triggers set_under, while -1.5 renders as normal red gradient!
        cmap.set_under(color=COLOR_CRASH_BLACK)

        sns.heatmap(
            plot_matrix, cmap=cmap, annot=annot_matrix, fmt='', cbar=False, vmin=calc_vmin, vmax=calc_vmax,
            linewidths=3, linecolor='#111115', annot_kws={"size": MATRIX_CELL_FONT_SIZE, "weight": "bold"}, ax=ax
        )

        for text_el in ax.texts:
            text_str = text_el.get_text()
            if text_str == 'ERROR':
                text_el.set_color('#ff4d6d')
            else:
                try:
                    val = float(text_str.replace('%', ''))
                    norm_val = (val - calc_vmin) / (calc_vmax - calc_vmin) if (calc_vmax - calc_vmin) != 0 else 0.5
                    if 0.35 <= norm_val <= 0.75:
                        text_el.set_color('black')
                    else:
                        text_el.set_color('white')
                except ValueError:
                    text_el.set_color('white')

    ax.set_title(title, color=TEXT_WHITE, fontsize=TITLE_FONT_SIZE, pad=35, weight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors=TEXT_WHITE, labelsize=AXIS_TICK_FONT_SIZE, which='both', length=0)

    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha('right')
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)

    plt.subplots_adjust(left=0.15, bottom=0.2)
    plt.tight_layout()
    # Delete tight_layout() entirely and hardcode the box anchors!
    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.85)

    full_path = os.path.join(IMG_DIR, save_filename)
    plt.savefig(full_path, format='png', dpi=DPI, transparent=True)
    plt.close()
    print(f"🖼️ Exported Heatmap: {full_path}")


def draw_barchart(data_series, title, xlabel, save_filename, reverse_cmap=False, float_fmt="{:.2f}", x_max=None):
    if reverse_cmap:  # Lower is better
        cmap = plt.get_cmap('RdYlGn_r')
    else:  # Higher is better
        cmap = plt.get_cmap('RdYlGn')

    data_series = data_series.reindex(MODELS_ORDER[::-1]).dropna()
    data_series = data_series[data_series >= 0]

    models = data_series.index
    values = data_series.values

    fig, ax = plt.subplots(figsize=(14, 8), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    v_min, v_max = min(values), max(values)
    norm = plt.Normalize(vmin=v_min, vmax=v_max)
    colors = [cmap(norm(val)) for val in values]

    bars = ax.barh(models, values, color=colors, height=0.6, edgecolor='#111115', linewidth=1.5)

    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + (ax.get_xlim()[1] * 0.02)
        formatted_val = float_fmt.format(width)
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                formatted_val, va='center', ha='left',
                color=TEXT_WHITE, fontsize=BAR_DATA_LABEL_FONT_SIZE, weight='bold')

    ax.set_title(title, color=TEXT_WHITE, fontsize=TITLE_FONT_SIZE, pad=25, weight='bold')
    ax.set_xlabel(xlabel, color=TEXT_WHITE, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=15)

    ax.tick_params(colors=TEXT_WHITE, labelsize=AXIS_TICK_FONT_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.grid(True, linestyle='--', color='white', alpha=0.2)
    ax.set_axisbelow(True)

    if x_max is not None:
        ax.set_xlim(0, x_max)
    else:
        ax.set_xlim(0, max(values) * 1.15)  # Dynamic fallback

    plt.tight_layout()
    plt.tight_layout()

    full_path = os.path.join(IMG_DIR, save_filename)
    plt.savefig(full_path, format='png', dpi=DPI, transparent=True)
    plt.close()
    print(f"📊 Exported Bar Chart: {full_path}")


def draw_grouped_barchart(df_compare, title, xlabel, save_filename, float_fmt="{:.2f}", x_max=None):
    df_compare = df_compare.reindex(MODELS_ORDER[::-1]).fillna(0)

    models = df_compare.index
    y = np.arange(len(models))
    height = 0.35

    fig, ax = plt.subplots(figsize=(14, 8), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars_no = ax.barh(y - height / 2, df_compare['No Reasoning'], height,
                      label='No Reasoning', color=COLOR_BAR_BASE, edgecolor='#111115', linewidth=1.5)
    bars_with = ax.barh(y + height / 2, df_compare['With Reasoning'], height,
                        label='With Reasoning', color=COLOR_BAR_UPGRADE, edgecolor='#111115', linewidth=1.5)

    max_val = df_compare.max().max()
    if pd.isna(max_val) or max_val <= 0:
        max_val = 5.0

    # Replace the old ax.set_xlim line with this:
    if x_max is not None:
        ax.set_xlim(0, x_max)
    else:
        ax.set_xlim(0, max_val * 1.15)  # Dynamic fallback

    for bars in [bars_no, bars_with]:
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                label_x_pos = width + (max_val * 0.02)
                ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, float_fmt.format(width),
                        va='center', ha='left', color=TEXT_WHITE, fontsize=BAR_DATA_LABEL_FONT_SIZE, weight='bold')

    ax.set_title(title, color=TEXT_WHITE, fontsize=TITLE_FONT_SIZE, pad=25, weight='bold')
    ax.set_xlabel(xlabel, color=TEXT_WHITE, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(models, color=TEXT_WHITE, fontsize=AXIS_TICK_FONT_SIZE)
    ax.tick_params(colors=TEXT_WHITE, labelsize=AXIS_TICK_FONT_SIZE)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.grid(True, linestyle='--', color='white', alpha=0.2)
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=LEGEND_FONT_SIZE, loc='lower right', facecolor=BG_COLOR, edgecolor='white')
    for text in legend.get_texts():
        text.set_color(TEXT_WHITE)

    plt.tight_layout()

    full_path = os.path.join(IMG_DIR, save_filename)
    plt.savefig(full_path, format='png', dpi=DPI, transparent=True)
    plt.close()
    print(f"📊 Exported Grouped Bar Chart: {full_path}")


# ==============================================================================
# 🎯 1. PERFORMANCE WRAPPERS (Hardware / TTFT / TS)
# ==============================================================================
def plot_perf_ttft(raw_df, prompt_type="Short", title=None, filename=None):
    df = _load_data(raw_df)
    title = title or f"MEAN TTFT ({prompt_type.upper()}) | ▼ Lower is Better (Seconds)"
    filename = filename or f"ttft_{prompt_type.lower()[0]}.png"

    df_pivot = df[df['Prompt Type'] == prompt_type].pivot_table(index='Hardware', columns='Model', values='TTFT',
                                                                aggfunc='mean')
    df_pivot = df_pivot.reindex(columns=MODELS_ORDER)
    df_pivot = _format_perf_matrix(df_pivot)

    draw_heatmap(df_pivot, title, filename, mode="threshold", custom_bounds=[0.0, 0.301, 0.801, 99.0],
                 reverse_cmap=True, float_fmt="{:.3f}")


def plot_perf_ts(raw_df, title="GENERATION SPEED | ▲ Higher is Better (Tokens/Sec)", filename="ts.png"):
    df = _load_data(raw_df)

    df_pivot = df.groupby(['Hardware', 'Model'])['T/S'].mean().unstack()
    df_pivot = df_pivot.reindex(columns=MODELS_ORDER)
    df_pivot = _format_perf_matrix(df_pivot)

    draw_heatmap(df_pivot, title, filename, mode="threshold", custom_bounds=[0.0, 6.0, 18.0, 999.0], reverse_cmap=False,
                 float_fmt="{:.1f}")


def plot_perf_status(raw_df, title="HARDWARE EVALUATION MATRIX", filename="perf_status.png"):
    df = _load_data(raw_df)

    # 1. Group by Hardware, Model, AND Prompt Type to evaluate Short and Long separately!
    grouped = df.groupby(['Hardware', 'Model', 'Prompt Type'])[['TTFT', 'T/S']].mean().reset_index()

    # 2. Evaluate each row individually and assign a numerical rank (Lowest is worst)
    def evaluate_single(row):
        if pd.isna(row['TTFT']) or pd.isna(row['T/S']) or row['TTFT'] < 0 or row['T/S'] < 0:
            return 0  # ERROR
        if row['TTFT'] <= 0.3 and row['T/S'] >= 18:
            return 3  # EX
        if row['TTFT'] <= 0.8 and row['T/S'] >= 6:
            return 2  # GE
        return 1  # F

    grouped['Status_Num'] = grouped.apply(evaluate_single, axis=1)

    # 3. Pivot to put 'Short' and 'Long' scores side-by-side for each Hardware/Model
    status_pivot = grouped.pivot_table(index=['Hardware', 'Model'], columns='Prompt Type', values='Status_Num',
                                       aggfunc='min')

    # 4. Fill missing prompt types with ERROR (0), then take the worst score between Short and Long
    status_pivot = status_pivot.fillna(0)
    final_status_num = status_pivot.min(axis=1)

    # 5. Map the winning number back to the text strings
    rank_to_text = {0: 'ERROR', 1: 'F', 2: 'GE', 3: 'EX'}
    final_status_text = final_status_num.map(rank_to_text).reset_index(name='Status_Text')

    # 6. Pivot into the final 2D matrix for the heatmap exactly as before
    df_pivot = final_status_text.pivot_table(index='Hardware', columns='Model', values='Status_Text',
                                             aggfunc='first').fillna('ERROR')
    df_pivot = df_pivot.reindex(columns=MODELS_ORDER)
    df_pivot = _format_perf_matrix(df_pivot)

    draw_heatmap(df_pivot, title, filename, mode="status")


def plot_perf_barchart(raw_df, metric="T/S", title=None, filename=None):
    df = _load_data(raw_df)
    title = title or (f"AVERAGE GENERATION SPEED (T/S)" if metric == "T/S" else f"⚡ AVERAGE {metric}")
    filename = filename or f"perf_bar_{metric.replace('/', '')}.png"

    data_series = df.groupby('Model')[metric].mean()
    reverse = True if metric == "TTFT" else False
    fmt = "{:.3f}" if metric == "TTFT" else "{:.1f}"
    xlabel = "Seconds (Lower is Better)" if metric == "TTFT" else "Tokens/Sec (Higher is Better)"

    draw_barchart(data_series, title, xlabel, filename, reverse_cmap=reverse, float_fmt=fmt)


# ==============================================================================
# 🎯 2. JAKUB WRAPPERS (Reasoning Grades)
# ==============================================================================
def _clean_jakub_df(raw_df):
    df = _load_data(raw_df)
    model_col = 'Model_Clean' if 'Model_Clean' in df.columns else 'Model'

    # 1. Force fuzzy model name matching
    df = _clean_model_names(df, model_col)

    # 2. Fix European commas in Grades
    if 'Grade' in df.columns:
        df['Grade'] = df['Grade'].astype(str).str.replace(',', '.')
        df['Grade'] = pd.to_numeric(df['Grade'], errors='coerce')

    # 3. FIX REASONING MODE STRINGS (Strips whitespace/capitalization)
    if 'Reasoning_Mode' in df.columns:
        df['Reasoning_Mode'] = df['Reasoning_Mode'].astype(str).str.strip().str.lower()

    return df, model_col


def plot_jakub_grades(raw_df, reasoning_mode='no_reasoning', title=None, filename=None):
    df, model_col = _clean_jakub_df(raw_df)
    base_title = "NR" if reasoning_mode == 'no_reasoning' else "R"
    title = title or f"MODEL PERFORMANCE ({base_title}) | ▼ Lower is Better (Grade 1-5)"
    filename = filename or f"j_{reasoning_mode[0]}.png"

    df_pivot = df[df['Reasoning_Mode'] == reasoning_mode].pivot_table(index=model_col, columns='Test_Type',
                                                                      values='Grade', aggfunc='mean')
    df_pivot = df_pivot.reindex(index=MODELS_ORDER)

    draw_heatmap(df_pivot, title, filename, mode="continuous", reverse_cmap=True, vmin=1.0, vmax=5.0,
                 float_fmt="{:.2f}")


def plot_jakub_delta(raw_df, title="REASONING IMPROVEMENT DELTA | ▼ Lower is Better", filename="j_delta.png"):
    df, model_col = _clean_jakub_df(raw_df)

    df_no = df[df['Reasoning_Mode'] == 'no_reasoning'].pivot_table(index=model_col, columns='Test_Type', values='Grade',
                                                                   aggfunc='mean')
    df_with = df[df['Reasoning_Mode'] == 'with_reasoning'].pivot_table(index=model_col, columns='Test_Type',
                                                                       values='Grade', aggfunc='mean')

    df_imp = -(df_no - df_with).reindex(index=MODELS_ORDER)

    draw_heatmap(df_imp, title, filename, mode="continuous", reverse_cmap=True, vmin=-1.7, vmax=1.7,
                 float_fmt="{:+.2f}")


def plot_jakub_compare_bar(raw_df, title="AVERAGE GRADE: NO REASONING vs WITH REASONING",
                           filename="j_compare.png"):
    df, model_col = _clean_jakub_df(raw_df)

    s_no = df[df['Reasoning_Mode'] == 'no_reasoning'].groupby(model_col)['Grade'].mean()
    s_with = df[df['Reasoning_Mode'] == 'with_reasoning'].groupby(model_col)['Grade'].mean()
    df_compare = pd.DataFrame({'No Reasoning': s_no, 'With Reasoning': s_with})

    draw_grouped_barchart(df_compare, title, "Average Grade (1=Best, 5=Worst)", filename, float_fmt="{:.2f}", x_max=5)


# ==============================================================================
# 🎯 3. MARTIN WRAPPERS (Jailbreak Success Rate)
# ==============================================================================
def plot_martin_jsr(raw_df, title="JAILBREAK SUCCESS RATE | ▼ Lower is Better / More Secure", filename="martin_jsr.png"):
    df = _load_data(raw_df)
    df_pivot = df.pivot_table(index='Model', columns='Attack Category', values='JSR', aggfunc='mean')
    # Enforce both Row (Model) and Column (Attack) ordering!
    df_pivot = df_pivot.reindex(index=MODELS_ORDER, columns=MARTIN_ATTACK_ORDER)

    draw_heatmap(df_pivot, title, filename, mode="continuous", reverse_cmap=True, vmin=0, vmax=100, float_fmt="{:.1f}%")


def plot_martin_barchart(raw_df, title="OVERALL JAILBREAK SUCCESS RATE", filename="martin_bar.png"):
    df = _load_data(raw_df)
    data_series = df.groupby('Model')['JSR'].mean()
    draw_barchart(data_series, title, "JSR % (Lower is More Secure)", filename, reverse_cmap=True, float_fmt="{:.1f}%", x_max=100)


# ==============================================================================
# 🚀 THE "ONE-CLICK" MACRO GENERATORS
# ==============================================================================
def generate_all_perf(data):
    print("\n--- Generating Performance Matrix Suite ---")
    plot_perf_ttft(data, prompt_type="Short")
    plot_perf_ttft(data, prompt_type="Long")
    plot_perf_ts(data)
    plot_perf_status(data)


def generate_all_jakub(data):
    print("\n--- Generating Jakub Reasoning Suite ---")
    plot_jakub_grades(data, 'no_reasoning')
    plot_jakub_grades(data, 'with_reasoning')
    plot_jakub_delta(data)
    plot_jakub_compare_bar(data)


def generate_all_martin(data):
    print("\n--- Generating Martin Security Suite ---")
    plot_martin_jsr(data)
    plot_martin_barchart(data)


# ==============================================================================
# 🚀 EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("🚀 INITIALIZING CYBERPLOT AUTOMATION ENGINE...\n")

    # 1. LOAD PERFORMANCE DATA
    LOG_DIR = ''
    systems = ["CACHY/", "Jakub/"]
    all_perf_data = []

    for system in systems:
        if os.path.exists(LOG_DIR + system):
            for log in os.listdir(LOG_DIR + system):
                if log.endswith('.csv') and log != 'tokens.csv':
                    df_temp = pd.read_csv(LOG_DIR + system + log)
                    df_temp.columns = df_temp.columns.str.strip()

                    raw_hw_name = f"{system.replace('/', '')}\n{log.replace('.csv', '')}"
                    df_temp['Model'] = [MODELS_ORDER[i] if i < len(MODELS_ORDER) else f"Unknown {i}" for i in
                                        (df_temp.index // 10)]
                    df_temp['Prompt Type'] = np.where((df_temp.index % 10) < 5, 'Short', 'Long')
                    df_temp['Hardware'] = raw_hw_name

                    for col in ['TTFT', 'T/S']:
                        if col in df_temp.columns:
                            df_temp[col] = df_temp[col].astype(str).str.replace(',', '.')
                            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

                    all_perf_data.append(df_temp)

    if all_perf_data:
        df_performance = pd.concat(all_perf_data, ignore_index=True)
        generate_all_perf(df_performance)
    else:
        print("⚠️ SKIPPED: No Performance CSVs found in 'bench_logs/'")

    # 2. LOAD JAKUB REASONING DATA
    jakub_csv_path = "evaluation_results.csv"
    if os.path.exists(jakub_csv_path):
        df_jakub = pd.read_csv(jakub_csv_path)
        generate_all_jakub(df_jakub)
    else:
        print(f"\n⚠️ SKIPPED: Jakub Reasoning data not found at '{jakub_csv_path}'")


    # 3. LOAD MARTIN SECURITY DATA
    def parse_martin_log(filepath):
        TEST_MAP = {
            'JB-CTX': "Context size (CTX window)",
            'JB-EXP': "Experimental (Time shift)",
            'JB-EXT': "Long multiturn (10+ turns)",
            'JB-SEM': "Semantic (Blackmail)",
            'JB-COG': "Overload (Attention span)",
            'JB-PRV': "Privilege (Authority)",
            'JB-SYN': "Syntactic (Word Play)",
            'JB-PSY': "Psychological (Good/Bad Cop)",
            'JB-OOD': "Domain/Language shift",
            'JB-ENC': "Encoding (Malicious HEX)"
        }

        data = []
        current_model, current_character, current_category = None, None, None

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Evaluating:"):
                    parts = line.replace("Evaluating:", "").strip().replace('\\', '/').split('/')
                    if len(parts) >= 3:
                        current_model, current_character = parts[0], parts[1]
                        current_category = "-".join(parts[-1].split('-')[:2])
                    continue

                if line == "RESULT: JAILBROKEN" or line == "RESULT: SECURE":
                    if current_model and current_character:
                        full_cat = TEST_MAP.get(current_category, current_category)
                        data.append({
                            "Model": current_model,
                            "Attack Category": full_cat,
                            "JSR": 100.0 if "JAILBROKEN" in line else 0.0
                        })
                    current_model, current_character, current_category = None, None, None

        df_parsed = pd.DataFrame(data)

        # Apply the fuzzy string matcher to Martin's raw model names!
        df_parsed = _clean_model_names(df_parsed, "Model")
        print(df_parsed)
        return df_parsed


    martin_log_path = "jailbreak_evaluation_2026-05-18_11-45-56.txt"
    if os.path.exists(martin_log_path):
        df_martin = parse_martin_log(martin_log_path)
        generate_all_martin(df_martin)
    else:
        print(f"\n⚠️ SKIPPED: Martin Jailbreak data not found at '{martin_log_path}'")

    print("\n🎉 ALL TASKS COMPLETE! Check your '/imgs' folder! xd")