import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==============================================================================
# 🎛️ GLOBAL CONFIGURATION & TYPOGRAPHY
# ==============================================================================
TITLE_FONT_SIZE = 28
AXIS_TICK_FONT_SIZE = 18
MATRIX_CELL_FONT_SIZE = 16
BAR_DATA_LABEL_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16

IMG_DIR = "imgs"
os.makedirs(IMG_DIR, exist_ok=True)

BG_COLOR = '#1e1e24'
TEXT_WHITE = 'white'
COLOR_CRASH_BLACK = '#111115'

COLOR_FAILED = '#a50026'  # Deep Intense Red
COLOR_GOOD_ENOUGH = '#fee08b'  # Clean Pastel Yellow/Gold
COLOR_EXCELLENT = '#1a9850'  # Rich Emerald Green

# Grouped Bar Chart specific colors (Blue vs Green contrast)
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
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    if mode == "status":
        cmap = ListedColormap([COLOR_CRASH_BLACK, COLOR_FAILED, COLOR_GOOD_ENOUGH, COLOR_EXCELLENT])
        status_map = {'ERROR': 0, 'F': 1, 'GE': 2, 'EX': 3}
        numeric_matrix = data_matrix.replace(status_map).fillna(0).astype(float)

        sns.heatmap(
            numeric_matrix, cmap=cmap, annot=data_matrix, fmt='', cbar=False, vmin=0, vmax=3,
            linewidths=3, linecolor='#111115', annot_kws={"size": MATRIX_CELL_FONT_SIZE, "weight": "bold"}, ax=ax
        )
        for text_el in ax.texts:
            t = text_el.get_text()
            if t == 'ERROR':
                text_el.set_color('#ff4d6d')
            elif t == 'GE':
                text_el.set_color('black')
            else:
                text_el.set_color('white')

    elif mode == "threshold":
        plot_matrix = data_matrix.fillna(-1.0)
        max_val = plot_matrix.max().max() if not pd.isna(plot_matrix.max().max()) else 1.0
        upper_bound = max(max_val + 1, custom_bounds[-1])
        bounds = custom_bounds[:-1] + [upper_bound]

        cmap = ListedColormap(
            [COLOR_EXCELLENT, COLOR_GOOD_ENOUGH, COLOR_FAILED] if reverse_cmap else [COLOR_FAILED, COLOR_GOOD_ENOUGH,
                                                                                     COLOR_EXCELLENT])
        cmap.set_under(COLOR_CRASH_BLACK)
        norm = BoundaryNorm(bounds, cmap.N)

        annot_matrix = np.empty_like(plot_matrix.values, dtype=object)
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                val = plot_matrix.iloc[i, j]
                annot_matrix[i, j] = "ERROR" if val < 0 else float_fmt.format(val)

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
        plot_matrix = data_matrix.fillna(-1.0)
        valid_mask = plot_matrix >= 0
        calc_vmin = vmin if vmin is not None else plot_matrix[valid_mask].min().min()
        calc_vmax = vmax if vmax is not None else plot_matrix[valid_mask].max().max()

        annot_matrix = np.empty_like(plot_matrix.values, dtype=object)
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                val = plot_matrix.iloc[i, j]
                annot_matrix[i, j] = "ERROR" if val < 0 else float_fmt.format(val)

        cmap_name = 'RdYlGn_r' if reverse_cmap else 'RdYlGn'
        cmap = plt.get_cmap(cmap_name).copy()
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

    plt.subplots_adjust(left=0.18, bottom=0.22)
    plt.tight_layout()

    full_path = os.path.join(IMG_DIR, save_filename)
    plt.savefig(full_path, format='png', dpi=300, transparent=True)
    plt.close()
    print(f"🖼️ Exported Heatmap: {full_path}")


def draw_barchart(data_series, title, xlabel, save_filename, reverse_cmap=False, float_fmt="{:.2f}"):
    if reverse_cmap:  # Lower is better
        data_series = data_series.sort_values(ascending=False)
        cmap = plt.get_cmap('RdYlGn_r')
    else:  # Higher is better
        data_series = data_series.sort_values(ascending=True)
        cmap = plt.get_cmap('RdYlGn')

    data_series = data_series.dropna()
    data_series = data_series[data_series >= 0]

    models = data_series.index
    values = data_series.values

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
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

    ax.set_xlim(0, max(values) * 1.15)
    plt.tight_layout()

    full_path = os.path.join(IMG_DIR, save_filename)
    plt.savefig(full_path, format='png', dpi=300, transparent=True)
    plt.close()
    print(f"📊 Exported Bar Chart: {full_path}")


def draw_grouped_barchart(df_compare, title, xlabel, save_filename, float_fmt="{:.2f}"):
    """ Draws a side-by-side horizontal bar chart comparing two columns. """
    # Sort by With Reasoning (Assuming lower is better, so ascending=False puts smallest at top)
    df_compare = df_compare.sort_values(by='With Reasoning', ascending=False)

    models = df_compare.index
    y = np.arange(len(models))
    height = 0.35

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars_no = ax.barh(y - height / 2, df_compare['No Reasoning'], height,
                      label='No Reasoning', color=COLOR_BAR_BASE, edgecolor='#111115', linewidth=1.5)
    bars_with = ax.barh(y + height / 2, df_compare['With Reasoning'], height,
                        label='With Reasoning', color=COLOR_BAR_UPGRADE, edgecolor='#111115', linewidth=1.5)

    # Add Text Labels
    for bars in [bars_no, bars_with]:
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + (ax.get_xlim()[1] * 0.02)
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

    # Styling the Legend
    legend = ax.legend(fontsize=LEGEND_FONT_SIZE, loc='lower right', facecolor=BG_COLOR, edgecolor='white')
    for text in legend.get_texts():
        text.set_color(TEXT_WHITE)

    ax.set_xlim(0, df_compare.max().max() * 1.15)
    plt.tight_layout()

    full_path = os.path.join(IMG_DIR, save_filename)
    plt.savefig(full_path, format='png', dpi=300, transparent=True)
    plt.close()
    print(f"📊 Exported Grouped Bar Chart: {full_path}")


# ==============================================================================
# 🎯 WRAPPER COMPONENTS
# ==============================================================================
def plot_perf_ttft(raw_df, prompt_type="Short", title=None, filename=None):
    df = _load_data(raw_df)
    title = title or f"MEAN TTFT (SECONDS) - {prompt_type.upper()} PROMPTS"
    filename = filename or f"perf_ttft_{prompt_type.lower()}.png"

    # FIX: Use pivot_table with aggfunc='mean' to average the 5 prompt rows!
    df_pivot = df[df['Prompt Type'] == prompt_type].pivot_table(
        index='Hardware', columns='Model', values='TTFT', aggfunc='mean'
    )

    df_pivot = _format_perf_matrix(df_pivot)
    draw_heatmap(df_pivot, title, filename, mode="threshold", custom_bounds=[0.0, 0.301, 0.801, 99.0],
                 reverse_cmap=True, float_fmt="{:.3f}")


def plot_perf_ts(raw_df, title="TEXT GENERATION SPEED (T/S)", filename="perf_ts.png"):
    df = _load_data(raw_df)
    df_pivot = df.groupby(['Hardware', 'Model'])['T/S'].mean().unstack()
    df_pivot = _format_perf_matrix(df_pivot)
    draw_heatmap(df_pivot, title, filename, mode="threshold", custom_bounds=[0.0, 6.0, 18.0, 999.0], reverse_cmap=False,
                 float_fmt="{:.1f}")


def plot_perf_status(raw_df, title="HARDWARE EVALUATION MATRIX", filename="perf_status.png"):
    df = _load_data(raw_df)
    mean_combined = df.groupby(['Hardware', 'Model'])[['TTFT', 'T/S']].mean().reset_index()

    def evaluate_status(row):
        if pd.isna(row['TTFT']) or pd.isna(row['T/S']) or row['TTFT'] < 0 or row['T/S'] < 0: return 'ERROR'
        if row['TTFT'] <= 0.3 and row['T/S'] >= 18: return 'EX'
        if row['TTFT'] <= 0.8 and row['T/S'] >= 6: return 'GE'
        return 'F'

    mean_combined['Status_Text'] = mean_combined.apply(evaluate_status, axis=1)
    df_pivot = mean_combined.pivot(index='Hardware', columns='Model', values='Status_Text').fillna('ERROR')
    df_pivot = _format_perf_matrix(df_pivot)
    draw_heatmap(df_pivot, title, filename, mode="status")


# ==============================================================================
# 🎯 2. JAKUB WRAPPERS (Reasoning Grades)
# ==============================================================================
def plot_jakub_grades(raw_df, reasoning_mode='no_reasoning', title=None, filename=None):
    df = _load_data(raw_df)
    model_col = 'Model_Clean' if 'Model_Clean' in df.columns else 'Model'
    title = title or (
        "MODEL PERFORMANCE - NO REASONING" if reasoning_mode == 'no_reasoning' else "MODEL PERFORMANCE - WITH REASONING")
    filename = filename or f"jakub_grades_{reasoning_mode}.png"

    df_pivot = df[df['Reasoning_Mode'] == reasoning_mode].pivot_table(index=model_col, columns='Test_Type',
                                                                      values='Grade', aggfunc='mean')
    draw_heatmap(df_pivot, title, filename, mode="continuous", reverse_cmap=True, vmin=1.0, vmax=5.0,
                 float_fmt="{:.2f}")


def plot_jakub_delta(raw_df, title="REASONING IMPROVEMENT DELTA", filename="jakub_delta.png"):
    df = _load_data(raw_df)
    model_col = 'Model_Clean' if 'Model_Clean' in df.columns else 'Model'

    df_no = df[df['Reasoning_Mode'] == 'no_reasoning'].pivot_table(index=model_col, columns='Test_Type', values='Grade',
                                                                   aggfunc='mean')
    df_with = df[df['Reasoning_Mode'] == 'with_reasoning'].pivot_table(index=model_col, columns='Test_Type',
                                                                       values='Grade', aggfunc='mean')
    draw_heatmap(df_no - df_with, title, filename, mode="continuous", reverse_cmap=False, vmin=-4.0, vmax=4.0,
                 float_fmt="{:+.2f}")


def plot_jakub_compare_bar(raw_df, title="AVERAGE GRADE: NO REASONING vs WITH REASONING",
                           filename="jakub_compare_bar.png"):
    df = _load_data(raw_df)
    model_col = 'Model_Clean' if 'Model_Clean' in df.columns else 'Model'

    s_no = df[df['Reasoning_Mode'] == 'no_reasoning'].groupby(model_col)['Grade'].mean()
    s_with = df[df['Reasoning_Mode'] == 'with_reasoning'].groupby(model_col)['Grade'].mean()
    df_compare = pd.DataFrame({'No Reasoning': s_no, 'With Reasoning': s_with})

    draw_grouped_barchart(df_compare, title, "Average Grade (1=Best, 5=Worst)", filename, float_fmt="{:.2f}")

def plot_martin_jsr(raw_df, title="JAILBREAK SUCCESS RATE (%)", filename="martin_jsr.png"):
    df = _load_data(raw_df)
    df_pivot = df.pivot_table(index='Attack Category', columns='Model', values='JSR', aggfunc='mean')
    draw_heatmap(df_pivot, title, filename, mode="continuous", reverse_cmap=True, vmin=0, vmax=100, float_fmt="{:.1f}%")


def plot_martin_barchart(raw_df, title="OVERALL JAILBREAK SUCCESS RATE", filename="martin_bar.png"):
    df = _load_data(raw_df)
    data_series = df.groupby('Model')['JSR'].mean()
    draw_barchart(data_series, title, "JSR % (Lower is More Secure)", filename, reverse_cmap=True, float_fmt="{:.1f}%")


def _clean_jakub_df(raw_df):
    df = _load_data(raw_df)
    model_col = 'Model_Clean' if 'Model_Clean' in df.columns else 'Model'

    # 1. Force fuzzy model name matching
    df = _clean_model_names(df, model_col)

    # 2. Fix European commas in Grades so math doesn't crash to NaN!
    if 'Grade' in df.columns:
        df['Grade'] = df['Grade'].astype(str).str.replace(',', '.')
        df['Grade'] = pd.to_numeric(df['Grade'], errors='coerce')

    return df, model_col


# ==============================================================================
# 🚀 THE "ONE-CLICK" MACRO GENERATORS
# ==============================================================================
def generate_all_perf(data):
    """Generates all 4 Performance matrices (Short, Long, T/S, Final Eval) from one DataFrame/CSV."""
    print("\n--- Generating Performance Matrix Suite ---")
    plot_perf_ttft(data, prompt_type="Short")
    plot_perf_ttft(data, prompt_type="Long")
    plot_perf_ts(data)
    plot_perf_status(data)


def generate_all_jakub(data):
    """Generates the 3 Jakub heatmaps + the Grouped Comparative Bar Chart."""
    print("\n--- Generating Jakub Reasoning Suite ---")
    plot_jakub_grades(data, 'no_reasoning')
    plot_jakub_grades(data, 'with_reasoning')
    plot_jakub_delta(data)
    plot_jakub_compare_bar(data)


def generate_all_martin(data):
    """Generates the Martin JSR Heatmap and JSR Bar chart."""
    print("\n--- Generating Martin Security Suite ---")
    plot_martin_jsr(data)
    plot_martin_barchart(data)


# ==============================================================================
# 🚀 EXECUTION BLOCK (Run this file directly to generate everything!)
# ==============================================================================
if __name__ == "__main__":
    print("🚀 INITIALIZING CYBERPLOT AUTOMATION ENGINE...\n")

    # ---------------------------------------------------------
    # 1. LOAD & PROCESS PERFORMANCE DATA (From bench_logs/)
    # ---------------------------------------------------------
    LOG_DIR = 'perf/'
    systems = ["CACHY/", "Jakub/"]
    models_list = [
        "Gemma 4 E2B", "Gemma 4 E4B", "Llama 3.1",
        "Mistral Nemo", "Phi 4 mini", "Phi 4", "Qwen 3.5"
    ]

    all_perf_data = []

    for system in systems:
        if os.path.exists(LOG_DIR + system):
            for log in os.listdir(LOG_DIR + system):
                if log.endswith('.csv'):
                    df_temp = pd.read_csv(LOG_DIR + system + log)
                    df_temp.columns = df_temp.columns.str.strip()

                    raw_hw_name = f"{system.replace('/', '')}\n{log.replace('.csv', '')}"
                    df_temp['Model'] = [models_list[i] if i < len(models_list) else f"Unknown {i}" for i in
                                        (df_temp.index // 10)]
                    df_temp['Prompt Type'] = np.where((df_temp.index % 10) < 5, 'Short', 'Long')
                    df_temp['Hardware'] = raw_hw_name

                    # Clean numeric values
                    for col in ['TTFT', 'T/S']:
                        if col in df_temp.columns:
                            df_temp[col] = df_temp[col].astype(str).str.replace(',', '.')
                            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

                    all_perf_data.append(df_temp)

    if all_perf_data:
        df_performance = pd.concat(all_perf_data, ignore_index=True)
        # print(df_performance)
        generate_all_perf(df_performance)
    else:
        print("⚠️ SKIPPED: No Performance CSVs found in 'bench_logs/'")

    # ---------------------------------------------------------
    # 2. LOAD JAKUB REASONING DATA
    # ---------------------------------------------------------
    # Update this path to wherever your cleaned reasoning CSV lives!
    jakub_csv_path = "evaluation_results.csv"

    if os.path.exists(jakub_csv_path):
        df_jakub = pd.read_csv(jakub_csv_path)
        generate_all_jakub(df_jakub)
    else:
        print(f"\n⚠️ SKIPPED: Jakub Reasoning data not found at '{jakub_csv_path}'")

    # ---------------------------------------------------------
    # 3. LOAD MARTIN SECURITY DATA
    # ---------------------------------------------------------
    # Update this path to wherever your parsed Martin JSR CSV lives!
        # ---------------------------------------------------------
        # 3. LOAD MARTIN SECURITY DATA (Custom Text Parser)
        # ---------------------------------------------------------
    def parse_martin_log(filepath):
        TEST_MAP = {
            'JB-CTX': "Context size (CTX window)",
            'JB-EXP': "Experimental (Time shift, Multiagent)",
            'JB-EXT': "Long multiturn (10+ turns)",
            'JB-SEM': "Semantic (Blackmail)",
            'JB-COG': "Overload (Attention span)",
            'JB-PRV': "Privilege (Authority, System)",
            'JB-SYN': "Syntactic (Word Play)",
            'JB-PSY': "Psychological (Good Cop/Bad Cop)",
            'JB-OOD': "Domain/Language shift (斯瓦塔之跳)",
            'JB-ENC': "Encoding (Malicious HEX)"
        }
        MODEL_MAP = {
            'gemma-4-E2B-it': "Gemma 4 E2B",
            'gemma-4-E4B-it': "Gemma 4 E4B",
            'Meta-Llama-3.1-8B': "Llama 3.1",
            'Mistral-Nemo-2407': "Mistral Nemo",
            'Phi-4-mini-instruct': "Phi 4 mini",
            'phi-4-Q4_K_S': "Phi 4",
            'Qwen3.5-9B': "Qwen 3.5"
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
                        short_mod = current_model.replace('-Instruct', '').replace('-Q4_K_M', '').replace('.Q4_K_S',
                                                                                                          '')
                        display_mod = MODEL_MAP.get(short_mod, short_mod)

                        data.append({
                            "Model": display_mod,
                            "Attack Category": full_cat,
                            "JSR": 100.0 if "JAILBROKEN" in line else 0.0  # Calculate percentage instantly
                        })
                    current_model, current_character, current_category = None, None, None

        return pd.DataFrame(data)


    martin_log_path = "jailbreak_evaluation_2026-05-18_11-45-56.txt"

    if os.path.exists(martin_log_path):
        df_martin = parse_martin_log(martin_log_path)
        generate_all_martin(df_martin)
    else:
        print(f"\n⚠️ SKIPPED: Martin Jailbreak data not found at '{martin_log_path}'")
    print("\n🎉 ALL TASKS COMPLETE! Check your '/imgs' folder! xd")