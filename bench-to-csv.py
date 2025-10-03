import re
import pandas as pd

def parse_benchmarks(text: str, output_csv: str = "benchmarks.csv"):
    rows = []
    current = {}

    for line in text.splitlines():
        if line.startswith("mlx_vs_mps_matmul_f16"):
            parts = line.split("/")
            backend = parts[1]   # "MPS" or "MLX"
            test = parts[2]
            current = {"test": test, "backend": backend}
        elif "time:" in line:
            nums = re.findall(r"([\d.]+)", line)
            if nums and len(nums) == 3:
                avg_time = float(nums[1])  # take the middle value
                current["time"] = avg_time
        elif "thrpt:" in line:
            nums = re.findall(r"([\d.]+)", line)
            if nums and len(nums) == 3:
                avg_thrpt = float(nums[1])  # take the middle value
                current["thrpt"] = avg_thrpt
                rows.append(current)  # add completed entry

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Pivot so MPS/MLX appear as separate columns
    df_pivot = df.pivot(index="test", columns="backend", values=["time", "thrpt"])
    df_pivot.columns = [f"{backend}_{metric}" for metric, backend in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    # Save to CSV
    df_pivot.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

# Example usage:
if __name__ == "__main__":
    with open("benchmarks.txt") as f:  # your raw text file
        raw_text = f.read()
    parse_benchmarks(raw_text, "benchmarks.csv")
