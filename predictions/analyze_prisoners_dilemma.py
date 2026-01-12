import pandas as pd
import glob
import re

def find_report_files():
    return sorted(glob.glob('final_full_analytical_report_task2_minimal_run*.csv'))

def extract_run_number(filename):
    match = re.search(r'_run(\d+)\.csv$', filename)
    return int(match.group(1)) if match else None

def summarize_accuracy(report_file):
    df = pd.read_csv(report_file)
    
    # Team-level
    team_level_df = df.drop_duplicates(subset=['session_id'])
    total_team_predictions = len(team_level_df)
    correct_team_predictions = team_level_df['team_prediction_correct'].sum()
    team_accuracy_majority = (correct_team_predictions / total_team_predictions * 100) if total_team_predictions > 0 else 0
    
    correct_team_predictions_ai = (team_level_df['actual_team_outcome'] == team_level_df['predicted_team_outcome']).sum()
    team_accuracy_ai = (correct_team_predictions_ai / total_team_predictions * 100) if total_team_predictions > 0 else 0
    
    # Individual-level
    total_individual_choices = len(df)
    correct_individual_choices = df['individual_prediction_correct'].sum()
    individual_accuracy = (correct_individual_choices / total_individual_choices * 100) if total_individual_choices > 0 else 0
    
    return {
        'file': report_file,
        'run_number': extract_run_number(report_file),
        'team_accuracy_majority': team_accuracy_majority,
        'correct_team_predictions_majority': correct_team_predictions,
        'team_accuracy_ai': team_accuracy_ai,
        'correct_team_predictions_ai': correct_team_predictions_ai,
        'total_team_predictions': total_team_predictions,
        'individual_accuracy': individual_accuracy,
        'correct_individual_choices': correct_individual_choices,
        'total_individual_choices': total_individual_choices
    }

def main():
    report_files = find_report_files()
    if not report_files:
        print('No report files found.')
        return
    
    results = [summarize_accuracy(f) for f in report_files]
    results.sort(key=lambda x: x['run_number'])
    
    summary_df = pd.DataFrame(results)
    
    print('\nACCURACY RESULTS')
    print('='*40)
    print(summary_df[['run_number','team_accuracy_majority','team_accuracy_ai','individual_accuracy',
                      'correct_team_predictions_majority','correct_team_predictions_ai','total_team_predictions',
                      'correct_individual_choices','total_individual_choices']].to_string(index=False, float_format='%.2f'))
    
    # Averages and std dev
    avg_team_acc = summary_df['team_accuracy_majority'].mean()
    std_team_acc = summary_df['team_accuracy_majority'].std(ddof=1) if len(summary_df) > 1 else 0
    avg_team_acc_ai = summary_df['team_accuracy_ai'].mean()
    std_team_acc_ai = summary_df['team_accuracy_ai'].std(ddof=1) if len(summary_df) > 1 else 0
    avg_indiv_acc = summary_df['individual_accuracy'].mean()
    std_indiv_acc = summary_df['individual_accuracy'].std(ddof=1) if len(summary_df) > 1 else 0
    
    print('\nAVERAGES ACROSS ALL RUNS')
    print(f"Average Team Accuracy (Majority):     {avg_team_acc:.2f}%")
    print(f"Standard Deviation (Majority):        {std_team_acc:.2f}%")
    print(f"Average Team Accuracy (AI):           {avg_team_acc_ai:.2f}%")
    print(f"Standard Deviation (AI):              {std_team_acc_ai:.2f}%")
    print(f"Average Individual Vote Accuracy:     {avg_indiv_acc:.2f}%")
    print(f"Standard Deviation (Individual):      {std_indiv_acc:.2f}%")
    
    # Save
    summary_df.to_csv('aggregated_accuracy_summary_minimal.csv', index=False)
    print('\nSaved summary to aggregated_accuracy_summary_minimal.csv')

if __name__ == '__main__':
    main()
