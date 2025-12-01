import pandas as pd
import os
import sys
import json
import re

from structured_prompt_loader_task1 import get_structured_prediction_from_system_user_task1

# --- Configuration ---
RUN_NUMBER = int(sys.argv[1]) if len(sys.argv) > 1 else 1

EXCEL_FILE = 'merged_table_cason_2019.xlsx'
SYSTEM_PROMPT_FILE = 'instructions/minimum_effort_game_structured_system_minimal.txt'
USER_PROMPT_TEMPLATE_FILE = 'instructions/user_message_template_minimum_effort_minimal.txt'

# --- Output ---
GROUND_TRUTH_FILE = f'ground_truth_answers_task1{RUN_NUMBER}.csv'
PREDICTIONS_FILE = f'predictions_minimum_effort{RUN_NUMBER}.csv'
CONSOLIDATED_OUTPUT_FILE = 'minimum_effort_consolidated.csv'

# --- Column Names from Excel ---
SESSION_COLS = ['session', 'Cluster.x', 'Subgroup.x', 'task']
SENDER_COL = 'Sender'
MESSAGE_COL = 'texttype'
PROPOSAL_COL = 'T1_XProposal'
ANSWER_COL = 'T1_XChoice'
# ---------------------------------------------------------

def process_and_predict():
    """
    Reads data, sends it to the API using separate system/user prompts, and saves the results.
    """
    # --- Load the prompt files ---
    print("--- 1. Loading data and prompt files ---")
    try:
        with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
            system_message = f.read()
        with open(USER_PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            user_template = f.read()
        df = pd.read_excel(EXCEL_FILE)
        df = df[df['task'] == 1]
        print(f"Files loaded. Filtered data to task 1, found {len(df)} rows.")
    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e}")
        return

    # Group by unique game sessions
    grouped_games = df.groupby(SESSION_COLS)
    print(f"Found and processing {len(grouped_games)} unique games.")

    ground_truth_data = []
    predictions_data = []

    # Loop through each unique game
    for name, group in grouped_games:
        session_id = str(name)
        print(f"--- Processing Game: {session_id} ---")
        group = group.sort_index()

        # Format Proposals and Chat Logs
        proposals_for_prompt = "Proposals:\n"
        for _, row in group[[SENDER_COL, PROPOSAL_COL]].drop_duplicates().iterrows():
            proposals_for_prompt += f"Player {row[SENDER_COL]} proposes: {row[PROPOSAL_COL]}\n"

        chat_log_for_prompt = "\nChat Log:\n"
        for _, row in group.iterrows():
            chat_log_for_prompt += f"Player {row[SENDER_COL]}: {row[MESSAGE_COL]}\n"
        
        try:
            # 1. Create the user message from the template
            user_message = user_template.format(
                PROPOSALS_DATA=proposals_for_prompt,
                CHAT_LOGS=chat_log_for_prompt
            )
            
            # 2. Call the API
            ai_prediction = get_structured_prediction_from_system_user_task1(
                system_message,
                user_message
            )
            print(f"AI Prediction Received for {session_id}")
        except Exception as e:
            print(f"Error getting prediction for {session_id}: {e}")
            ai_prediction = f'{{"error": "API call failed: {e}"}}'
        
        # Store the raw prediction text
        predictions_data.append({
            'session_id': session_id,
            'prediction_text': ai_prediction
        })

        # Extract the true answers
        for _, row in group[[SENDER_COL, ANSWER_COL]].drop_duplicates().iterrows():
            ground_truth_data.append({
                'session_id': session_id,
                'player': row[SENDER_COL],
                'true_choice': row[ANSWER_COL]
            })

    # Save the results
    if not predictions_data:
        print(f"\nWarning: No game sessions were found or processed from the Excel file.")
    else:
        truth_df = pd.DataFrame(ground_truth_data)
        truth_df.to_csv(GROUND_TRUTH_FILE, index=False)
        print(f"\nSuccessfully saved ground truth answers for all sessions to '{GROUND_TRUTH_FILE}'")

        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(PREDICTIONS_FILE, index=False)
        print(f"Successfully saved all AI predictions to '{PREDICTIONS_FILE}'")
        
        create_consolidated_comparison(predictions_df, truth_df, RUN_NUMBER)

def create_consolidated_comparison(predictions_df, truth_df, run_number):
    print(f"Creating consolidated comparison for run {run_number}...")
    all_player_rows = []
    for _, pred_row in predictions_df.iterrows():
        session_id = pred_row['session_id']
        prediction_text = pred_row['prediction_text']
        try:
            if prediction_text.strip().startswith('{'):
                parsed_json = json.loads(prediction_text)
            else:
                json_match = re.search(r'\{.*\}', prediction_text, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group(0))
                else:
                    parsed_json = {}
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not parse JSON for session {session_id}: {e}")
            parsed_json = {}
        session_truth = truth_df[truth_df['session_id'] == session_id]
        if session_truth.empty:
            print(f"Warning: No ground truth found for session {session_id}")
            continue
        player_predictions = parsed_json.get('player_predictions', [])
        group_outcome = parsed_json.get('conclusion', {}).get('outcome', 'N/A')
        for _, truth_row in session_truth.iterrows():
            player_id = str(truth_row['player'])
            actual_choice = truth_row['true_choice']
            predicted_choice = 'N/A'
            for pred in player_predictions:
                pred_player_id = str(pred.get('player_id', ''))
                id_match = re.search(r'\d+', pred_player_id)
                if id_match and id_match.group(0) == player_id:
                    predicted_choice = pred.get('predicted_choice', 'N/A')
                    break
            prediction_correctness = "N/A"
            if predicted_choice != 'N/A' and pd.notna(actual_choice):
                try:
                    prediction_correctness = "Correct" if int(predicted_choice) == int(actual_choice) else "Incorrect"
                except (ValueError, TypeError):
                    prediction_correctness = "Type Mismatch"
            player_row = {
                'run_number': run_number,
                'session_id': session_id,
                'player_id': player_id,
                'predicted_choice': predicted_choice,
                'actual_choice': actual_choice,
                'prediction_correctness': prediction_correctness,
                'group_outcome_prediction': group_outcome
            }
            all_player_rows.append(player_row)
    if not all_player_rows:
        print(f"No data created for run {run_number}")
        return
    run_df = pd.DataFrame(all_player_rows)
    if os.path.exists(CONSOLIDATED_OUTPUT_FILE):
        existing_df = pd.read_csv(CONSOLIDATED_OUTPUT_FILE)
        existing_df = existing_df[existing_df['run_number'] != run_number]
        consolidated_df = pd.concat([existing_df, run_df], ignore_index=True)
    else:
        consolidated_df = run_df
    consolidated_df.to_csv(CONSOLIDATED_OUTPUT_FILE, index=False)
    print(f"Updated consolidated comparison file '{CONSOLIDATED_OUTPUT_FILE}' with run {run_number} data")
    print(f"   - Added {len(run_df)} player predictions")
    print(f"   - Total runs in consolidated file: {consolidated_df['run_number'].nunique()}")

if __name__ == "__main__":
    process_and_predict()
