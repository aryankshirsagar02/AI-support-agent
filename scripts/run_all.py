"""
End-to-End Master Pipeline Runner.
Executes data preparation, retrieval indexing, model training, baselines, and evaluation in one command.
"""
import sys
import os
import subprocess

# Prefer direct python binary over WindowsApp aliases
PYTHON_EXEC = r"C:\Users\aryan\AppData\Local\Python\bin\python.exe"
if not os.path.exists(PYTHON_EXEC):
    PYTHON_EXEC = sys.executable


def run_cmd(cmd_list, desc):
    print("\n" + "=" * 70)
    print(f"STEP: {desc}")
    print("=" * 70)
    result = subprocess.run(cmd_list, check=True)
    return result


def main():
    print("=" * 70)
    print("SUPPORTIQ AI - COMPLETE END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Data Preparation
    run_cmd([PYTHON_EXEC, "scripts/prepare_data.py"], "1/4 Data Preparation & Thread Reconstruction")

    # 2. Build Retrieval Index
    run_cmd([PYTHON_EXEC, "scripts/build_index.py"], "2/4 Vector Retrieval Indexing")

    # 3. Train Baseline & Proposed Models
    run_cmd([PYTHON_EXEC, "scripts/run_baselines.py"], "3/4 Intent Model Training & Serialization")

    # 4. Comprehensive Evaluation & Benchmarking
    run_cmd([PYTHON_EXEC, "scripts/evaluate.py"], "4/4 Golden Set Evaluation & 5-System Benchmark")

    print("\n" + "*" * 70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("You can now launch the dashboard: python app/main.py")
    print("Or run the interactive demo:      python scripts/demo.py")
    print("*" * 70)


if __name__ == "__main__":
    main()
