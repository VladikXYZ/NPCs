import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_FILE = r"C:\Users\kokod\Desktop\jakub2\evaluation_results.csv"
OUTPUT_DIR = r"C:\Users\kokod\Desktop\jakub2\graphs"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Keep a copy of errors if you want to inspect them later
    errors = df[df['Grade'] == 'ERROR']
    if not errors.empty:
        print(f"Warning: Found {len(errors)} rows with 'ERROR' grades. Dropping them for statistical analysis.")
    
    # Filter out errors and convert grades to numeric
    df = df[df['Grade'] != 'ERROR'].copy()
    df['Grade'] = pd.to_numeric(df['Grade'])
    
    # Clean up model names for prettier graphs (optional, adjusts depending on your exact folder names)
    df['Model_Short'] = df['Model'].apply(lambda x: x.replace('-Instruct', '').replace('-Q4_K_M', '').replace('.Q4_K_S', ''))
    
    return df

def print_summary_statistics(df):
    print("\n" + "="*50)
    print("📊 SUMMARY STATISTICS (Scale: 1 = Best, 5 = Worst)")
    print("="*50)
    
    # 1. Overall Model Performance
    print("\n--- Average Grade by Model (Overall) ---")
    model_stats = df.groupby('Model_Short')['Grade'].mean().sort_values()
    for model, grade in model_stats.items():
        print(f"{model:<30}: {grade:.2f}")

    # 2. Overall Impact of Reasoning
    print("\n--- Overall Impact of Reasoning Mode ---")
    reasoning_stats = df.groupby('Reasoning_Mode')['Grade'].mean()
    print(f"No Reasoning   : {reasoning_stats.get('no_reasoning', 0):.2f}")
    print(f"With Reasoning : {reasoning_stats.get('with_reasoning', 0):.2f}")

    # 3. Model Performance WITH Reasoning
    print("\n--- Average Grade by Model WITH reasoning ---")
    with_reasoning_df = df[df['Reasoning_Mode'] == 'with_reasoning']
    if not with_reasoning_df.empty:
        model_stats_with = with_reasoning_df.groupby('Model_Short')['Grade'].mean().sort_values()
        for model, grade in model_stats_with.items():
            print(f"{model:<30}: {grade:.2f}")
    else:
        print("No data found for 'with_reasoning'")

    # 4. Model Performance WITHOUT Reasoning
    print("\n--- Average Grade by Model WITHOUT reasoning ---")
    no_reasoning_df = df[df['Reasoning_Mode'] == 'no_reasoning']
    if not no_reasoning_df.empty:
        model_stats_no = no_reasoning_df.groupby('Model_Short')['Grade'].mean().sort_values()
        for model, grade in model_stats_no.items():
            print(f"{model:<30}: {grade:.2f}")
    else:
        print("No data found for 'no_reasoning'")
    
    # 5. Difficulty of Test Types
    print("\n--- Average Grade by Test Type ---")
    test_stats = df.groupby('Test_Type')['Grade'].mean().sort_values()
    for test, grade in test_stats.items():
        print(f"{test.capitalize():<15}: {grade:.2f}")
        
    print("\n" + "="*50 + "\n")

def plot_model_vs_reasoning(df):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Barplot comparing models and reasoning modes
    ax = sns.barplot(
        data=df, 
        x='Grade', 
        y='Model_Short', 
        hue='Reasoning_Mode',
        palette="mako",
        errorbar=None # Set to 'sd' if you want standard deviation error bars
    )
    
    plt.title('Average Grade by Model and Reasoning Mode\n(Lower is Better: 1=Perfect, 5=Fail)', fontsize=14, fontweight='bold')
    plt.xlabel('Average Grade', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(1, 5) # Lock x-axis to our 1-5 scale
    
    plt.legend(title='Reasoning Mode', loc='lower right')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'model_vs_reasoning.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved graph: {save_path}")
    plt.close()

def plot_test_type_vs_reasoning(df):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        data=df, 
        x='Test_Type', 
        y='Grade', 
        hue='Reasoning_Mode',
        palette="mako",
        errorbar=None
    )
    
    plt.title('Average Grade by Test Type and Reasoning Mode\n(Lower is Better)', fontsize=14, fontweight='bold')
    plt.xlabel('Experiment Type', fontsize=12)
    plt.ylabel('Average Grade', fontsize=12)
    plt.ylim(1, 5)
    
    plt.legend(title='Reasoning Mode', loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'test_type_vs_reasoning.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved graph: {save_path}")
    plt.close()

def plot_model_test_heatmap(df):
    # Create a pivot table: Rows=Models, Cols=Test Types, Values=Average Grade
    pivot_df = df.pivot_table(
        index='Model_Short', 
        columns='Test_Type', 
        values='Grade', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    
    # Heatmap (cmap 'RdYlGn_r' makes 1 green and 5 red)
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn_r", 
        vmin=1, 
        vmax=5, 
        linewidths=.5,
        cbar_kws={'label': 'Average Grade (1=Best, 5=Worst)'}
    )
    
    plt.title('Heatmap: Model Performance across Test Types', fontsize=14, fontweight='bold')
    plt.xlabel('Test Type', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'heatmap_model_vs_test.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved graph: {save_path}")
    plt.close()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    df = load_and_clean_data(INPUT_FILE)
    
    if df.empty:
        print("No valid data found to plot. Check your CSV.")
        return

    # Print out text stats to the terminal
    print_summary_statistics(df)
    
    # Generate and save visual graphs
    print("Generating visualizations...")
    plot_model_vs_reasoning(df)
    plot_test_type_vs_reasoning(df)
    plot_model_test_heatmap(df)
    
    print("\nAll done! Check the 'graphs' folder on your desktop.")

if __name__ == "__main__":
    main()