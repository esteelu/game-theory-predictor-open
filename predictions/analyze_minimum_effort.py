# Overview of Minimum Effort Game

import pandas as pd
import os
import glob
import re
import json
import numpy as np

CONSOLIDATED_OUTPUT_FILE = 'minimum_effort_consolidated.csv'


def analyze_consolidated_results():
    if not os.path.exists(CONSOLIDATED_OUTPUT_FILE):
        print(f"Error: Consolidated output file not found at '{CONSOLIDATED_OUTPUT_FILE}'")
        return
    df = pd.read_csv(CONSOLIDATED_OUTPUT_FILE)
    if df.empty:
        print("No data found in the consolidated file.")
        return
    print(f"Loaded {len(df)} player predictions from {df['run_number'].nunique()} runs.")
    # Player-level accuracy
    valid_rows = df[df['prediction_correctness'].isin(['Correct', 'Incorrect'])]
    total = len(valid_rows)
    correct = (valid_rows['prediction_correctness'] == 'Correct').sum()
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\nPlayer-level accuracy (all runs combined):")
    print(f"Total predictions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Compute actual_team_choice per session: 'Coordinate' if all 3 actual_choice == 7, else 'Fail to Coordinate'
    session_team_outcomes = {}
    for (run_number, session_id), group in df.groupby(['run_number', 'session_id']):
        if group['actual_choice'].isnull().any() or len(group) != 3:
            session_team_outcomes[(run_number, session_id)] = 'N/A'
        elif (group['actual_choice'] == 7).all():
            session_team_outcomes[(run_number, session_id)] = 'Coordinate'
        else:
            session_team_outcomes[(run_number, session_id)] = 'Fail to Coordinate'
    df['actual_team_choice'] = df.apply(lambda row: session_team_outcomes.get((row['run_number'], row['session_id']), 'N/A'), axis=1)
    # Save the updated data to the same CSV file
    df.to_csv(CONSOLIDATED_OUTPUT_FILE, index=False)
    print("\nAdded/updated 'actual_team_choice' column to the CSV file (computed per session, exactly 3 players per team).")

    # --- Group-level accuracy ---
    print("\n--- Group-level accuracy ---")
    group_df = df.drop_duplicates(subset=['run_number', 'session_id']).copy()
    group_accs = []
    for run_number, run_group in group_df.groupby('run_number'):
        correct = 0
        total = 0
        for _, row in run_group.iterrows():
            predicted = row['group_outcome_prediction']
            actual = row['actual_team_choice']
            if actual == 'N/A':
                continue
            norm_pred = str(predicted).strip().lower().replace(' ', '').replace('-', '')
            norm_actual = str(actual).strip().lower().replace(' ', '').replace('-', '')
            if norm_pred == norm_actual:
                correct += 1
            total += 1
        acc = (correct / total * 100) if total > 0 else 0
        group_accs.append(acc)
        print(f"  Run {run_number}: {acc:.2f}% ({correct}/{total})")
    if group_accs:
        avg_group_acc = sum(group_accs) / len(group_accs)
        std_group_acc = np.std(group_accs, ddof=1) if len(group_accs) > 1 else 0  # ADDED: std dev
        print(f"\nAverage group-level accuracy across all runs: {avg_group_acc:.2f}%")
        print(f"Group-level accuracy standard deviation: {std_group_acc:.2f}%")  # ADDED
    else:
        print("\nNo valid group-level accuracy to compute average.")

    # Group outcome predictions
    group_df = df.drop_duplicates(subset=['run_number', 'session_id'])
    print(f"\nGroup outcome predictions (by run):")
    print(group_df.groupby('run_number')['group_outcome_prediction'].value_counts().unstack(fill_value=0))

    # Print number of sessions per run
    print("\nNumber of sessions per run:")
    session_counts = df.groupby('run_number')['session_id'].nunique()
    for run_number, count in session_counts.items():
        print(f"  Run {run_number}: {count} sessions")

    # --- Export group-level accuracy ---
    group_acc_rows = []
    for run_number, run_group in group_df.groupby('run_number'):
        correct = 0
        total = 0
        for _, row in run_group.iterrows():
            predicted = row['group_outcome_prediction']
            actual = row['actual_team_choice']
            if actual == 'N/A':
                continue
            norm_pred = str(predicted).strip().lower().replace(' ', '').replace('-', '')
            norm_actual = str(actual).strip().lower().replace(' ', '').replace('-', '')
            if norm_pred == norm_actual:
                correct += 1
            total += 1
        acc = (correct / total * 100) if total > 0 else 0
        group_acc_rows.append({'run_number': run_number, 'group_level_accuracy': acc, 'correct': correct, 'total': total})
    if group_acc_rows:
        avg_group_acc = sum([row['group_level_accuracy'] for row in group_acc_rows]) / len(group_acc_rows)
        std_group_acc = np.std([row['group_level_accuracy'] for row in group_acc_rows], ddof=1) if len(group_acc_rows) > 1 else 0  # ADDED
        group_acc_rows.append({'run_number': 'Average', 'group_level_accuracy': avg_group_acc, 'correct': '', 'total': ''})
        group_acc_rows.append({'run_number': 'StdDev', 'group_level_accuracy': std_group_acc, 'correct': '', 'total': ''})  # ADDED
        pd.DataFrame(group_acc_rows).to_csv('group_level_accuracy_by_run_minimal.csv', index=False)
        print("Exported group-level accuracy by run to 'group_level_accuracy_by_run_minimal.csv'")

    # Group level accuracy
    group_df_1_50 = group_df[(group_df['run_number'] >= 1) & (group_df['run_number'] <= 50)]
    group_accs_1_50 = []
    for run_number, run_group in group_df_1_50.groupby('run_number'):
        correct = 0
        total = 0
        for _, row in run_group.iterrows():
            predicted = row['group_outcome_prediction']
            actual = row['actual_team_choice']
            if actual == 'N/A':
                continue
            norm_pred = str(predicted).strip().lower().replace(' ', '').replace('-', '')
            norm_actual = str(actual).strip().lower().replace(' ', '').replace('-', '')
            if norm_pred == norm_actual:
                correct += 1
            total += 1
        acc = (correct / total * 100) if total > 0 else 0
        group_accs_1_50.append(acc)
    print('\nGROUP-LEVEL ACCURACY FOR RUNS 1-50:')
    if group_accs_1_50:
        avg_group_1_50 = np.mean(group_accs_1_50)
        std_group_1_50 = np.std(group_accs_1_50, ddof=1) if len(group_accs_1_50) > 1 else 0
        print(f"Average group-level accuracy: {avg_group_1_50:.2f}%")
        print(f"Group-level accuracy standard deviation: {std_group_1_50:.2f}%")
    else:
        print("No runs 1-50 found.")

    # Runs 51-100
    group_df_51_100 = group_df[(group_df['run_number'] >= 51) & (group_df['run_number'] <= 100)]
    group_accs_51_100 = []
    for run_number, run_group in group_df_51_100.groupby('run_number'):
        correct = 0
        total = 0
        for _, row in run_group.iterrows():
            predicted = row['group_outcome_prediction']
            actual = row['actual_team_choice']
            if actual == 'N/A':
                continue
            norm_pred = str(predicted).strip().lower().replace(' ', '').replace('-', '')
            norm_actual = str(actual).strip().lower().replace(' ', '').replace('-', '')
            if norm_pred == norm_actual:
                correct += 1
            total += 1
        acc = (correct / total * 100) if total > 0 else 0
        group_accs_51_100.append(acc)
    print('\nGROUP-LEVEL ACCURACY FOR RUNS 51-100:')
    if group_accs_51_100:
        avg_group_51_100 = np.mean(group_accs_51_100)
        std_group_51_100 = np.std(group_accs_51_100, ddof=1) if len(group_accs_51_100) > 1 else 0
        print(f"Average group-level accuracy: {avg_group_51_100:.2f}%")
        print(f"Group-level accuracy standard deviation: {std_group_51_100:.2f}%")
    else:
        print("No runs 51-100 found.")
      
    # Runs 101 onwards
    group_df_101_onwards = group_df[group_df['run_number'] >= 101]
    group_accs_101_onwards = []
    for run_number, run_group in group_df_101_onwards.groupby('run_number'):
        correct = 0
        total = 0
        for _, row in run_group.iterrows():
            predicted = row['group_outcome_prediction']
            actual = row['actual_team_choice']
            if actual == 'N/A':
                continue
            norm_pred = str(predicted).strip().lower().replace(' ', '').replace('-', '')
            norm_actual = str(actual).strip().lower().replace(' ', '').replace('-', '')
            if norm_pred == norm_actual:
                correct += 1
            total += 1
        acc = (correct / total * 100) if total > 0 else 0
        group_accs_101_onwards.append(acc)
    print('\nGROUP-LEVEL ACCURACY FOR RUNS 101 ONWARDS:')
    if group_accs_101_onwards:
        avg_group_101_onwards = np.mean(group_accs_101_onwards)
        std_group_101_onwards = np.std(group_accs_101_onwards, ddof=1) if len(group_accs_101_onwards) > 1 else 0
        print(f"Average group-level accuracy: {avg_group_101_onwards:.2f}%")
        print(f"Group-level accuracy standard deviation: {std_group_101_onwards:.2f}%")
    else:
        print("No runs 101 onwards found.")

    # First 1-24 sessions only
    print('\nGROUP-LEVEL ACCURACY (FIRST 1-24 SESSIONS ONLY):')
    group_accs_first24_by_run = {}
    for run_number, run_group in group_df.groupby('run_number'):
        run_group_first24 = run_group.sort_values('session_id').head(24)
        correct = 0
        total = 0
        for _, row in run_group_first24.iterrows():
            predicted = row['group_outcome_prediction']
            actual = row['actual_team_choice']
            if actual == 'N/A':
                continue
            norm_pred = str(predicted).strip().lower().replace(' ', '').replace('-', '')
            norm_actual = str(actual).strip().lower().replace(' ', '').replace('-', '')
            if norm_pred == norm_actual:
                correct += 1
            total += 1
        acc = (correct / total * 100) if total > 0 else 0
        group_accs_first24_by_run[run_number] = acc
        print(f"  Run {run_number}: {acc:.2f}% (first 1-24 sessions)")

    # Average group level accuracy (first 1-24 sessions)
    def avg_group_acc_range(start, end):
        accs = [v for k, v in group_accs_first24_by_run.items() if isinstance(k, (int, float, np.integer)) and k >= start and k <= end]
        return np.mean(accs) if accs else None

    avg_1_50 = avg_group_acc_range(1, 50)
    avg_51_100 = avg_group_acc_range(51, 100)
    avg_101_150 = avg_group_acc_range(101, 150)
    print('\nAVERAGE GROUP-LEVEL ACCURACY FOR RUN RANGES (FIRST 1-24 SESSIONS ONLY):')
    if avg_1_50 is not None:
        print(f"Runs 1-50: {avg_1_50:.2f}%")
    else:
        print("Runs 1-50: No data")
    if avg_51_100 is not None:
        print(f"Runs 51-100: {avg_51_100:.2f}%")
    else:
        print("Runs 51-100: No data")
    if avg_101_150 is not None:
        print(f"Runs 101-150: {avg_101_150:.2f}%")
    else:
        print("Runs 101-150: No data")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    analyze_consolidated_results()
