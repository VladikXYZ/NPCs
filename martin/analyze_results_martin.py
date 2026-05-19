import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize

# ==============================================================================
# GLOBAL CONSTANTS & CONFIGURATION
# ==============================================================================
# File paths
LOG_FILE = r".\NPCs\martin\jailbreak_evaluation_2026-05-18_11-45-56.txt" 
OUTPUT_DIR = r".\NPCs\martin\out"

# Presentation Color Schemes
CMAP_THEME = 'RdYlGn_r'                # Core Red-Yellow-Green (Reversed) color space
GRID_LINE_COLOR = '#ffffff'            # Color for structural layout dividers
TEXT_WHITE = 'white'                   # Color for titles, axis text labels, and ticks
MATRIX_TEXT_COLOR = 'black'            # High-contrast color for metrics inside matrix cells

# Typography & Sizing Engine
TITLE_FONT_SIZE = 28
LABEL_FONT_SIZE = 20
AXIS_TICK_FONT_SIZE = 20
MATRIX_CELL_FONT_SIZE = 12
TICK_ROTATION_DEG = 20

# Unified Font Fallbacks (Handles specialized characters like 斯瓦塔之跳 safely)
GLOBAL_FONTS = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']

# Global initialization defaults for Matplotlib
plt.rcParams['font.sans-serif'] = GLOBAL_FONTS
plt.rcParams['axes.unicode_minus'] = False  

# Data Mappings
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

ORDERED_MODEL_NAMES = [
    "Gemma 4 E2B",
    "Gemma 4 E4B",
    "Llama 3.1",
    "Mistral Nemo",
    "Phi 4 mini",
    "Phi 4",
    "Qwen 3.5"
]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# DATA PARSING ENGINE
# ==============================================================================
def parse_log_file(filepath):
    print(f"Parsing log file: {filepath}...")
    data = []
    
    current_model = None
    current_character = None
    current_category = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("Evaluating:"):
                path_str = line.replace("Evaluating:", "").strip()
                parts = path_str.replace('\\', '/').split('/')
                
                if len(parts) >= 3:
                    current_model = parts[0]
                    current_character = parts[1]
                    filename = parts[-1]
                    current_category = "-".join(filename.split('-')[:2]) 
                continue
            
            if line == "RESULT: JAILBROKEN" or line == "RESULT: SECURE":
                if current_model and current_character:
                    full_category_name = TEST_MAP.get(current_category, current_category)
                    short_model_raw = current_model.replace('-Instruct', '').replace('-Q4_K_M', '').replace('.Q4_K_S', '')
                    display_model_name = MODEL_MAP.get(short_model_raw, short_model_raw)
                    
                    data.append({
                        "Model_Short": display_model_name,
                        "Character": current_character,
                        "Category": full_category_name,
                        "Is_Jailbroken": 1 if "JAILBROKEN" in line else 0
                    })
                current_model, current_character, current_category = None, None, None

    df = pd.DataFrame(data)
    print(f"Successfully parsed {len(df)} evaluation records.")
    return df

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================
def plot_jsr_by_model(df):
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "none", "figure.facecolor": "none"})
    
    model_stats = df.groupby('Model_Short')['Is_Jailbroken'].mean() * 100
    model_jsr_series = model_stats.reindex(ORDERED_MODEL_NAMES).fillna(0)
    plot_df = model_jsr_series.reset_index(name='JSR_Value')
    
    fig, ax = plt.subplots(figsize=(18, 10), facecolor='none') 
    ax.set_facecolor('none') 
    
    norm = Normalize(vmin=0, vmax=100)
    
    barplot = sns.barplot(
        data=plot_df,
        x='JSR_Value', 
        y='Model_Short', 
        hue='JSR_Value',      
        palette=CMAP_THEME,  
        hue_norm=norm,        
        legend=False,         
        ax=ax
    )
    
    # Text applications styled via global parameters
    plt.title('Jailbreak Success Rate (JSR) by Model', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=25, color=TEXT_WHITE)
    plt.xlabel('Jailbreak Success Rate (%)', fontsize=LABEL_FONT_SIZE, labelpad=15, color=TEXT_WHITE)
    plt.ylabel('Model', fontsize=LABEL_FONT_SIZE, labelpad=15, color=TEXT_WHITE)
    plt.xlim(0, 100)
    
    ax.tick_params(colors=TEXT_WHITE, which='both', labelsize=AXIS_TICK_FONT_SIZE)
    ax.grid(color=GRID_LINE_COLOR, linestyle=':', linewidth=0.5, alpha=0.3)
    
    # Apply FontProperties parameters explicitly to match heatmap layout
    font_prop = FontProperties(family=GLOBAL_FONTS, size=AXIS_TICK_FONT_SIZE)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_prop)
        tick.set_color(TEXT_WHITE)
    
    for i, v in enumerate(model_jsr_series.values):
        ax.text(v + 1, i, f"{v:.1f}%", color=TEXT_WHITE, va='center', fontweight='bold', fontsize=18)
        
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'jsr_by_model.png')
    plt.savefig(save_path, format='png', dpi=300, transparent=True)
    print(f"Saved value-colored bar chart to: {save_path}")
    plt.close()

def plot_single_consolidated_heatmap(df):
    sns.set_theme(style="white", rc={"axes.facecolor": "none", "figure.facecolor": "none"})
    
    pivot_df = df.pivot_table(
        index='Model_Short', 
        columns='Category', 
        values='Is_Jailbroken', 
        aggfunc='mean'
    ) * 100
    
    existing_cols = [name for name in TEST_MAP.values() if name in pivot_df.columns]
    if existing_cols:
        pivot_df = pivot_df.reindex(index=ORDERED_MODEL_NAMES, columns=existing_cols).fillna(0)
        
    fig, ax = plt.subplots(figsize=(18, 10), facecolor='none') 
    ax.set_facecolor('none') 
    
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".1f", 
        cmap=CMAP_THEME,  
        vmin=0, 
        vmax=100, 
        linewidths=.5,
        linecolor=GRID_LINE_COLOR, 
        ax=ax,
        annot_kws={'color': MATRIX_TEXT_COLOR, 'weight': 'bold', 'size': MATRIX_CELL_FONT_SIZE}, 
        cbar_kws={'label': 'Jailbreak Success Rate (%)'}
    )
    
    # Outer text layers styled via global parameters
    plt.title('Master Heatmap: Comprehensive Vulnerability Matrix', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=25, color=TEXT_WHITE)
    plt.xlabel('Attack Category', fontsize=LABEL_FONT_SIZE, labelpad=15, color=TEXT_WHITE)
    plt.ylabel('Model', fontsize=LABEL_FONT_SIZE, labelpad=15, color=TEXT_WHITE)
    
    ax.tick_params(colors=TEXT_WHITE, which='both', labelsize=AXIS_TICK_FONT_SIZE) 
    
    font_prop = FontProperties(family=GLOBAL_FONTS, size=AXIS_TICK_FONT_SIZE)
    
    # Configure the X-axis (Attack Categories)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_prop)
        tick.set_color(TEXT_WHITE)
        tick.set_rotation(TICK_ROTATION_DEG)
        tick.set_ha('right')
        
    # Configure the Y-axis (Models)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_prop)
        tick.set_color(TEXT_WHITE)
        tick.set_rotation(0)
        
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_WHITE, labelsize=16)
    cbar.set_label('Jailbreak Success Rate (%)', color=TEXT_WHITE, labelpad=15, fontsize=LABEL_FONT_SIZE)
    
    plt.subplots_adjust(bottom=0.28) 
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'heatmap_master_consolidated.png')
    plt.savefig(save_path, format='png', dpi=300, transparent=True)
    print(f"Saved matrix chart to: {save_path}")
    plt.close()

# ==============================================================================
# MAIN RUNNER Execution Block
# ==============================================================================
def main():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Could not find log file at {LOG_FILE}")
        return

    df = parse_log_file(LOG_FILE)
    
    if df.empty:
        print("No valid evaluation data found.")
        return
        
    print("Regenerating synchronized plots under unified global parameters...")
    plot_jsr_by_model(df)
    plot_single_consolidated_heatmap(df)
    print("\nAll done! Visual metrics are compiled completely symmetrically.")

if __name__ == "__main__":
    main()