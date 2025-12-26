# Overview of Trust Game

import pandas as pd
import numpy as np

# --- Configuration ---
CONSOLIDATED_FILE = 'trust_game_consolidated.csv'
SUMMARY_STATS_FILE = 'trust_game_summary_statistics.csv'
PER_RUN_ACCURACY_FILE = 'trust_game_per_run_accuracy.csv'

def analyze_consolidated_results():
    try:
        df = pd.read_csv(CONSOLIDATED_FILE)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{CONSOLIDATED_FILE}'. Run the batch processing script first.")
        return
    
    df['run_number'] = pd.to_numeric(df['run_number'], errors='coerce').astype('Int64')
    
    total_predictions = len(df)
    correct_predictions = len(df[df['prediction_correctness'] == 'Correct'])
    overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    # --- Accuracy by Run ---
    run_accuracy = []
    for run_num in sorted(df['run_number'].unique()):
        run_data = df[df['run_number'] == run_num]
        run_total = len(run_data)
        run_correct = len(run_data[run_data['prediction_correctness'] == 'Correct'])
        run_acc = (run_correct / run_total) * 100 if run_total > 0 else 0
        run_accuracy.append({
            'run_number': run_num,
            'total_predictions': run_total,
            'correct_predictions': run_correct,
            'accuracy_percentage': run_acc
        })

    # Export accuracy by run
    run_accuracy_df = pd.DataFrame(run_accuracy)
    run_accuracy_df.to_csv(PER_RUN_ACCURACY_FILE, index=False)

    # --- Consistency Analysis ---
    run_accuracies = [stat['accuracy_percentage'] for stat in run_accuracy]

    # --- Summary Statistics ---
    summary_stats = {
        'metric': [
            'total_runs',
            'total_predictions', 
            'overall_accuracy_percent',
            'mean_run_accuracy_percent',
            'std_run_accuracy_percent',
            'min_run_accuracy_percent',
            'max_run_accuracy_percent',
            'median_run_accuracy_percent',
            'unique_sessions',
            'predictions_per_session_avg'
        ],
        'value': [
            df['run_number'].nunique(),
            total_predictions,
            overall_accuracy,
            np.mean(run_accuracies) if run_accuracies else 0,
            np.std(run_accuracies) if len(run_accuracies) > 1 else 0,
            min(run_accuracies) if run_accuracies else 0,
            max(run_accuracies) if run_accuracies else 0,
            np.median(run_accuracies) if run_accuracies else 0,
            df['session_id'].nunique(),
            total_predictions / df['session_id'].nunique() if df['session_id'].nunique() > 0 else 0
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(SUMMARY_STATS_FILE, index=False)

    # --- SPLIT ANALYSIS: Runs 1-50 ---
    print("=== SPLIT ANALYSIS: RUNS 1-50 ===")
    df_1_50 = df[(df['run_number'] >= 1) & (df['run_number'] <= 50)]
    total_1_50 = len(df_1_50)
    correct_1_50 = len(df_1_50[df_1_50['prediction_correctness'] == 'Correct'])
    acc_1_50 = (correct_1_50 / total_1_50 * 100) if total_1_50 > 0 else 0
    print(f"Total Predictions: {total_1_50}")
    print(f"Correct Predictions: {correct_1_50}")
    print(f"Overall Accuracy: {acc_1_50:.2f}%")
    
    run_accuracies_1_50 = []
    for run_num in sorted(df_1_50['run_number'].unique()):
        run_data = df_1_50[df_1_50['run_number'] == run_num]
        run_total = len(run_data)
        run_correct = len(run_data[run_data['prediction_correctness'] == 'Correct'])
        run_acc = (run_correct / run_total) * 100 if run_total > 0 else 0
        run_accuracies_1_50.append(run_acc)
    
    if run_accuracies_1_50:
        mean_1_50 = np.mean(run_accuracies_1_50)
        std_1_50 = np.std(run_accuracies_1_50)
        print(f"Mean accuracy: {mean_1_50:.2f}%")
        print(f"Standard deviation: {std_1_50:.2f}%")

    # --- SPLIT ANALYSIS: Runs 51-100 ---
    print("\n=== SPLIT ANALYSIS: RUNS 51-100 ===")
    df_51_100 = df[(df['run_number'] >= 51) & (df['run_number'] <= 100)]
    total_51_100 = len(df_51_100)
    correct_51_100 = len(df_51_100[df_51_100['prediction_correctness'] == 'Correct'])
    acc_51_100 = (correct_51_100 / total_51_100 * 100) if total_51_100 > 0 else 0
    print(f"Total Predictions: {total_51_100}")
    print(f"Correct Predictions: {correct_51_100}")
    print(f"Overall Accuracy: {acc_51_100:.2f}%")
    
    run_accuracies_51_100 = []
    for run_num in sorted(df_51_100['run_number'].unique()):
        run_data = df_51_100[df_51_100['run_number'] == run_num]
        run_total = len(run_data)
        run_correct = len(run_data[run_data['prediction_correctness'] == 'Correct'])
        run_acc = (run_correct / run_total) * 100 if run_total > 0 else 0
        run_accuracies_51_100.append(run_acc)
    
    if run_accuracies_51_100:
        mean_51_100 = np.mean(run_accuracies_51_100)
        std_51_100 = np.std(run_accuracies_51_100)
        print(f"Mean accuracy: {mean_51_100:.2f}%")
        print(f"Standard deviation: {std_51_100:.2f}%")

    # --- SPLIT ANALYSIS: Runs 101 Onwards ---
    print("\n=== SPLIT ANALYSIS: RUNS 101 ONWARDS ===")
    df_101_onwards = df[df['run_number'] >= 101]
    total_101_onwards = len(df_101_onwards)
    correct_101_onwards = len(df_101_onwards[df_101_onwards['prediction_correctness'] == 'Correct'])
    acc_101_onwards = (correct_101_onwards / total_101_onwards * 100) if total_101_onwards > 0 else 0
    print(f"Total Predictions: {total_101_onwards}")
    print(f"Correct Predictions: {correct_101_onwards}")
    print(f"Overall Accuracy: {acc_101_onwards:.2f}%")
    
    run_accuracies_101_onwards = []
    for run_num in sorted(df_101_onwards['run_number'].unique()):
        run_data = df_101_onwards[df_101_onwards['run_number'] == run_num]
        run_total = len(run_data)
        run_correct = len(run_data[run_data['prediction_correctness'] == 'Correct'])
        run_acc = (run_correct / run_total) * 100 if run_total > 0 else 0
        run_accuracies_101_onwards.append(run_acc)
    
    if run_accuracies_101_onwards:
        mean_101_onwards = np.mean(run_accuracies_101_onwards)
        std_101_onwards = np.std(run_accuracies_101_onwards)
        print(f"Mean accuracy: {mean_101_onwards:.2f}%")
        print(f"Standard deviation: {std_101_onwards:.2f}%")

    print(f"\n✓ Analysis complete! Files saved:")
    print(f"  - {PER_RUN_ACCURACY_FILE}")
    print(f"  - {SUMMARY_STATS_FILE}")


if __name__ == "__main__":
    analyze_consolidated_results()
