import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
models = [
    "Gemma 4 E2B (2B)", "Gemma 4 E4B (4B)", "Llama 3.1 (8B)",
    "Mistral Nemo (12B)", "Phi 4 mini (3.8B)", "Phi 4 (14B)", "Qwen 3.5 (9B)"
]
LOG_DIR = 'bench_logs/'
systems = ["CACHY/", "Jakub/"]

# --- DATA AGGREGATION & CLEANING ---
all_results = []

for system in systems:
    if not os.path.exists(LOG_DIR + system):
        continue
    for log in os.listdir(LOG_DIR + system):
        if not log.endswith('.csv'):
            continue

        df = pd.read_csv(LOG_DIR + system + log)
        df.columns = df.columns.str.strip()
        hw_name = f"{system.replace('/', '')} | {log.replace('.csv', '')}"

        df['Model'] = [models[i] if i < len(models) else f"Unknown {i}" for i in (df.index // 10)]
        df['Prompt Type'] = np.where((df.index % 10) < 5, 'Short', 'Long')
        df['Hardware'] = hw_name

        for col in ['TTFT', 'T/S']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.replace(-1.0, np.nan).replace(-1, np.nan)

        if 'TTFT' in df.columns and 'T/S' in df.columns:
            means = df.groupby(['Hardware', 'Model', 'Prompt Type'])[['TTFT', 'T/S']].mean().reset_index()
            all_results.append(means)

if not all_results:
    print("No valid CSV benchmarks found!")
    exit()

combined_df = pd.concat(all_results, ignore_index=True)

# Build individual base matrices
matrix_short_ttft = combined_df[combined_df['Prompt Type'] == 'Short'].pivot(index='Hardware', columns='Model',
                                                                             values='TTFT').reindex(columns=models)
matrix_long_ttft = combined_df[combined_df['Prompt Type'] == 'Long'].pivot(index='Hardware', columns='Model',
                                                                           values='TTFT').reindex(columns=models)
matrix_overall_ts = combined_df.groupby(['Hardware', 'Model'])['T/S'].mean().unstack().reindex(columns=models)

# Process Evaluation Matrix
mean_combined = combined_df.groupby(['Hardware', 'Model'])[['TTFT', 'T/S']].mean().reset_index()


def evaluate_status(row):
    if pd.isna(row['TTFT']) or pd.isna(row['T/S']): return 'F'
    return 'EX' if (row['TTFT'] <= 0.3 and row['T/S'] >= 18) else (
        'GE' if (row['TTFT'] <= 0.8 and row['T/S'] >= 6) else 'F')


mean_combined['Status'] = mean_combined.apply(evaluate_status, axis=1)
matrix_status = mean_combined.pivot(index='Hardware', columns='Model', values='Status').reindex(columns=models)


# --- CYBERPUNK DARK-MODE TABLE RENDERER ---
def plot_dark_table(df, title, filename, mode):
    """
    Renders an elegant dark-mode table matching the jailbreak evaluation style sheets.
    """
    # Dynamic styling setup to map the text layout spacing perfectly
    fig, ax = plt.subplots(figsize=(15, 2.5 + len(df) * 0.55), dpi=300)
    fig.patch.set_facecolor('#1e1e24')  # Set background to dark tone
    ax.set_facecolor('#1e1e24')
    ax.axis('off')

    cell_text = df.round(2).fillna('-').astype(str).values
    col_labels = [c.replace(' (', '\n(') for c in df.columns]
    row_labels = df.index.tolist()

    # Dark Theme Palette Mapping (Bright text, clean semi-dark cell blocks)
    theme = {
        'bg_main': '#1e1e24',
        'bg_header': '#111115',
        'green_cell': '#1b4332', 'text_green': '#52b788',
        'yellow_cell': '#5c4d0a', 'text_yellow': '#ffd166',
        'red_cell': '#4c1c24', 'text_red': '#ff4d6d',
        'neutral_cell': '#2a2a35', 'text_neutral': '#e0e0e0'
    }

    cell_colors = []
    cell_text_colors = []

    for row in cell_text:
        row_bg, row_txt = [], []
        for val in row:
            if val == '-':
                row_bg.append(theme['neutral_cell']);
                row_txt.append(theme['text_neutral'])
                continue

            num = float(val) if mode != 'status' else val

            if mode == 'ttft':
                if num <= 0.3:
                    bg, txt = theme['green_cell'], theme['text_green']
                elif num <= 0.8:
                    bg, txt = theme['yellow_cell'], theme['text_yellow']
                else:
                    bg, txt = theme['red_cell'], theme['text_red']
            elif mode == 'ts':
                if num >= 20:
                    bg, txt = theme['green_cell'], theme['text_green']
                elif num >= 10:
                    bg, txt = theme['yellow_cell'], theme['text_yellow']
                else:
                    bg, txt = theme['red_cell'], theme['text_red']
            elif mode == 'status':
                if num == 'EX':
                    bg, txt = theme['green_cell'], theme['text_green']
                elif num == 'GE':
                    bg, txt = theme['yellow_cell'], theme['text_yellow']
                else:
                    bg, txt = theme['red_cell'], theme['text_red']

            row_bg.append(bg)
            row_txt.append(txt)
        cell_colors.append(row_bg)
        cell_text_colors.append(row_txt)

    # Render base layout
    mpl_table = ax.table(
        cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
        cellColours=cell_colors, loc='center', cellLoc='center'
    )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(11)
    mpl_table.scale(1, 2.2)  # Generous height scaling for code visibility

    # Style cells individually to inject foreground text rules and border alignments
    for (r, c), cell in mpl_table.get_celld().items():
        cell.set_edgecolor('#111115')  # Very faint border matching backgrounds
        cell.set_linewidth(1.5)

        if r > 0 and c >= 0:
            # Data cells
            cell.get_text().set_color(cell_text_colors[r - 1][c])
            cell.get_text().set_weight('bold')
        elif r == 0:
            # Column headers
            cell.set_facecolor(theme['bg_header'])
            cell.get_text().set_color('#ffffff')
            cell.get_text().set_weight('bold')
        elif c == -1:
            # Row headers (Hardware lines)
            cell.set_facecolor(theme['bg_header'])
            cell.get_text().set_color('#00b4d8')  # Cyan accent color for hardware text
            cell.get_text().set_weight('bold')

    plt.title(title, fontsize=15, pad=30, weight='bold', color='#ffffff')
    plt.tight_layout()

    # Save as high-quality transparent PNG just like your heatmaps!
    plt.savefig(filename, bbox_inches='tight', facecolor=theme['bg_main'], transparent=True)
    plt.close()
    print(f"🌆 Saved Cyberpunk Table to: {filename}")


# --- EXECUTE THE GENERATION ---
plot_dark_table(matrix_short_ttft, "⚡ MEAN TTFT (SECONDS) - SHORT PROMPTS", "dark_short_ttft.png", "ttft")
plot_dark_table(matrix_long_ttft, "⏳ MEAN TTFT (SECONDS) - LONG PROMPTS", "dark_long_ttft.png", "ttft")
plot_dark_table(matrix_overall_ts, "🚀 MEAN TEXT GENERATION SPEED (TOKENS/SEC)", "dark_tokens_sec.png", "ts")
plot_dark_table(matrix_status, "🏆 HARDWARE EVALUATION BREAKDOWN (EX / GE / F)", "dark_model_status.png", "status")

print("\n🎉 Your dark-mode benchmark dashboard is fully compiled! Go check the PNG outputs! xd")