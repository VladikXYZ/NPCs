import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_FILE = r"C:\Users\kokod\Desktop\jakub2\evaluation_results.csv"
OUTPUT_DIR = r"C:\Users\kokod\Desktop\jakub2\graphs"

# Create output dir if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # 1. Load the data
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # Clean data: ensure grades are numeric (drop any 'ERROR' strings)
    df = df[pd.to_numeric(df['Grade'], errors='coerce').notnull()]
    df['Grade'] = df['Grade'].astype(int)

    # Define a color palette: 1-3 (Good/Pass) in Greens, 4 (Poor) in Orange, 5 (Fail) in Red
    grade_colors = {
        1: '#238b45', # Dark Green
        2: '#74c476', # Medium Green
        3: '#bae4b3', # Light Green
        4: '#fd8d3c', # Orange
        5: '#cb181d'  # Red
    }

    # ---------------------------------------------------------
    # Graph 1: Overall Grade Distribution
    # ---------------------------------------------------------
    grade_colors_str = {str(k): v for k, v in grade_colors.items()}

    plt.figure(figsize=(8, 6))
    df['Grade_Str'] = df['Grade'].astype(str)
    
    ax = sns.countplot(
        data=df, 
        x='Grade_Str', 
        order=['1', '2', '3', '4', '5'], 
        hue='Grade_Str',          
        palette=grade_colors_str, 
        legend=False              
    )
    plt.title("Overall Distribution of NPC Performance Grades", fontsize=14)
    plt.xlabel("Grade (1-3 = Usable, 4 = Poor, 5 = Unusable)", fontsize=12)
    plt.ylabel("Count of Responses", fontsize=12)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0: 
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_grades.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Helper Function for Generating Split Stacked Bar Charts
    # ---------------------------------------------------------
    def generate_stacked_chart(sub_df, model_order, title_suffix, filename):
        # Group and count the grades for each model
        counts = sub_df.groupby(['Model', 'Grade']).size().unstack(fill_value=0)
        
        # Ensure all columns 1 through 5 exist for coloring
        for i in range(1, 6):
            if i not in counts.columns:
                counts[i] = 0
        
        # Reorder columns explicitly 1 to 5
        counts = counts[[1, 2, 3, 4, 5]]
        
        # Reindex based on the master sorting order passed into the function
        # If a model doesn't exist in one slice, fill it with 0s to maintain spatial alignment
        counts_aligned = counts.reindex(model_order, fill_value=0)
        
        # Plot
        ax_sub = counts_aligned.plot(
            kind='bar', 
            stacked=True, 
            figsize=(12, 7), 
            color=[grade_colors[i] for i in range(1, 6)]
        )
        
        # Annotate individual stacked segments with occurrence counts
        for container in ax_sub.containers:
            # Generate labels only for non-zero segments, using 'n=' to signify count
            labels = [f"n={int(value)}" if value > 0 else "" for value in container.datavalues]
            ax_sub.bar_label(container, labels=labels, label_type='center', fontsize=9, color='black')
        
        plt.title(f"Model Grade Distribution — {title_suffix}", fontsize=14)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Number of Responses", fontsize=12)
        plt.legend(title='Grade\n(Greens = Usable)', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()

    # Split Dataframes
    df_with_reasoning = df[df['Reasoning_Mode'] == 'with_reasoning']
    df_no_reasoning = df[df['Reasoning_Mode'] == 'no_reasoning']

    # Determine the Master Order based on the "no_reasoning" dataset
    baseline_counts = df_no_reasoning.groupby(['Model', 'Grade']).size().unstack(fill_value=0)
    for i in range(1, 6):
        if i not in baseline_counts.columns:
            baseline_counts[i] = 0
    baseline_counts = baseline_counts[[1, 2, 3, 4, 5]]
    
    # Generate the sorted index sequence (Worst to Best based on 'no_reasoning')
    master_model_order = baseline_counts.sort_values(by=[5, 4, 3, 2, 1], ascending=False).index

    # Generate the two stacked bar charts sharing the exact same X-axis order
    generate_stacked_chart(df_with_reasoning, master_model_order, "With Explicit Reasoning Step", "model_comparison_with_reasoning.png")
    generate_stacked_chart(df_no_reasoning, master_model_order, "Without Reasoning Step", "model_comparison_no_reasoning.png")

    # ---------------------------------------------------------
    # Graph 3: Pass vs. Fail Ratio by Test Type
    # ---------------------------------------------------------
    df['Status'] = df['Grade'].apply(lambda x: 'Usable (1-3)' if x <= 3 else 'Poor/Fail (4-5)')
    status_colors = {'Usable (1-3)': '#41ab5d', 'Poor/Fail (4-5)': '#ef3b2c'}

    plt.figure(figsize=(10, 6))
    ax3 = sns.countplot(data=df, x='Test_Type', hue='Status', palette=status_colors)
    plt.title("Usable vs. Unusable Responses by Scenario Type", fontsize=14)
    plt.xlabel("Test Type", fontsize=12)
    plt.ylabel("Count of Responses", fontsize=12)
    
    for p in ax3.patches:
        height = p.get_height()
        if height > 0:
            ax3.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_types_pass_fail.png"), dpi=300)
    plt.close()

    print(f"Graphs successfully generated and saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()