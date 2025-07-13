import pandas as pd

def save_results_to_excel(results, output_path):
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"\n📁 Results saved to: {output_path}")
