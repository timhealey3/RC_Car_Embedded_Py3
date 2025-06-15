import pandas as pd
import os
from datetime import datetime

def get_file_path(csv_file, script_dir):
    """Helper function to get absolute file path."""
    if csv_file is None:
        csv_file = 'training_dataset.csv'
    return os.path.join(script_dir, csv_file) if not os.path.isabs(csv_file) else csv_file

def analyze_forward_values(csv_file=None):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = get_file_path(csv_file, script_dir)
    
    print(f"\nAnalyzing data file: {os.path.basename(csv_path)}")
    print("-" * 50)
    
    try:
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        forward_counts = df['forward'].value_counts().to_dict()
        
            
        return True
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return False

def filter_forward_zero(csv_file=None, output_suffix='_filtered'):
    """
    Create a new CSV file with rows where forward is 0 removed.
    
    Args:
        csv_file (str): Path to the input CSV file
        output_suffix (str): Suffix to add to the original filename for the output file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = get_file_path(csv_file, script_dir)
    
    # Create output filename with timestamp
    base_name, ext = os.path.splitext(os.path.basename(input_path))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{base_name}{output_suffix}_{timestamp}{ext}"
    output_path = os.path.join(script_dir, output_filename)
    
    try:
        print(f"\nReading from: {input_path}")
        df = pd.read_csv(input_path)
        
        # Filter out rows where forward is 0
        filtered_df = df[df['forward'] != 0]
        rows_removed = len(df) - len(filtered_df)
        
        # Save to new file
        filtered_df.to_csv(output_path, index=False)
        
        print(f"Created filtered file: {output_filename}")
        print(f"- Original rows: {len(df)}")
        print(f"- Rows removed: {rows_removed}")
        print(f"- Rows kept: {len(filtered_df)}")
        
        return output_path
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    # First analyze the original file
    print("Training Data Analysis Tool")
    print("=" * 50)
    
    if analyze_forward_values():
        # Ask user if they want to filter the data
        response = input("\nDo you want to create a filtered version with forward=0 rows removed? (y/n): ").strip().lower()
        if response == 'y':
            output_file = filter_forward_zero()
            if output_file:
                print("\nAnalyzing filtered data:")
                analyze_forward_values(output_file)
    
    print("\nProcessing complete.")