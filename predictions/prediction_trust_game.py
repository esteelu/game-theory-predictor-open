import pandas as pd
import os
import sys
import json

from structured_prompt_loader_task3 import get_structured_game_prediction_system_user

# --- Configuration ---
RUN_NUMBER = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# --- File Names ---
EXCEL_FILE = 'CD_trust_game_outcomes.csv'
SYSTEM_PROMPT_FILE = 'instructions/trust_game_system_minimal.txt'
USER_PROMPT_TEMPLATE_FILE = 'instructions/trust_game_user_minimal.txt'
GROUND_TRUTH_FILE = f'ground_truth_trust_game_run{RUN_NUMBER}.csv'
PREDICTIONS_FILE = f'predictions_trust_game_run{RUN_NUMBER}.csv'
CONSOLIDATED_OUTPUT_FILE = 'trust_game_consolidated.csv'

# --- Column Names from CSV ---
SESSION_COL = 'Session'
MESSAGE_COL = 'Message'
ACTION_COL = 'Action' # 0 = Defect, 1 = Cooperate
# ---------------------------------------------------------

def process_and_predict_trust_game():
    if not os.path.exists(SYSTEM_PROMPT_FILE) or not os.path.exists(USER_PROMPT_TEMPLATE_FILE):
        print(f"ERROR: System or user prompt file not found.")
        return
    try:
        df = pd.read_csv(EXCEL_FILE)
        df.dropna(subset=[MESSAGE_COL], inplace=True)
        print(f"Loaded '{EXCEL_FILE}'. Found {len(df)} sessions with messages to process.")
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{EXCEL_FILE}'")
        return

    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
        system_message = f.read()
    with open(USER_PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        user_template = f.read()

    ground_truth_data = []
    predictions_data = []

    for index, row in df.iterrows():
        session_id = row[SESSION_COL]
        message_text = row[MESSAGE_COL]
        actual_action = row[ACTION_COL]
        print(f"\n--- Processing Session: {session_id} ---")
        try:
            user_message = user_template.format(PLAYER_B_MESSAGE=message_text)
            ai_prediction_json = get_structured_game_prediction_system_user(
                system_message,
                user_message
            )
            print(f"AI Prediction Received for Session {session_id}")
        except Exception as e:
            print(f"  Error getting prediction for Session {session_id}: {e}")
            ai_prediction_json = f"Error: {e}"
        predictions_data.append({
            'session_id': session_id,
            'prediction_text': ai_prediction_json
        })
        action_label = 'Cooperate' if actual_action == 1 else 'Defect'
        ground_truth_data.append({
            'session_id': session_id,
            'actual_action': action_label
        })
    if not predictions_data:
        print(f"WARNING: No game sessions were processed from the CSV file.")
    else:
        truth_df = pd.DataFrame(ground_truth_data)
        truth_df.to_csv(GROUND_TRUTH_FILE, index=False)
        print(f"Successfully saved ground truth answers to '{GROUND_TRUTH_FILE}'")
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(PREDICTIONS_FILE, index=False)
        print(f"Successfully saved all AI predictions to '{PREDICTIONS_FILE}'")
        create_consolidated_comparison(predictions_df, truth_df, RUN_NUMBER)

def create_consolidated_comparison(predictions_df, truth_df, run_number):
    print(f"Creating consolidated comparison for run {run_number}...")
    all_session_rows = []
    for _, pred_row in predictions_df.iterrows():
        session_id = pred_row['session_id']
        prediction_text = pred_row['prediction_text']
        try:
            if prediction_text.strip().startswith('{'):
                parsed_json = json.loads(prediction_text)
            else:
                import re
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
        predicted_action = parsed_json.get('final_prediction', 'N/A')
        actual_action = session_truth.iloc[0]['actual_action'] if len(session_truth) > 0 else 'N/A'
        reasoning = parsed_json.get('prediction_summary', 'N/A')
        prediction_correctness = "N/A"
        if predicted_action != 'N/A' and actual_action != 'N/A':
            prediction_correctness = "Correct" if predicted_action == actual_action else "Incorrect"
        session_row = {
            'run_number': run_number,
            'session_id': session_id,
            'predicted_action': predicted_action,
            'actual_action': actual_action,
            'prediction_correctness': prediction_correctness,
            'reasoning': reasoning
        }
        all_session_rows.append(session_row)
    if not all_session_rows:
        print(f"No comparison data created for run {run_number}")
        return
    run_df = pd.DataFrame(all_session_rows)
    if os.path.exists(CONSOLIDATED_OUTPUT_FILE):
        existing_df = pd.read_csv(CONSOLIDATED_OUTPUT_FILE)
        existing_df = existing_df[existing_df['run_number'] != run_number]
        consolidated_df = pd.concat([existing_df, run_df], ignore_index=True)
    else:
        consolidated_df = run_df
    consolidated_df.to_csv(CONSOLIDATED_OUTPUT_FILE, index=False)
    print(f"Updated consolidated comparison file '{CONSOLIDATED_OUTPUT_FILE}' with run {run_number} data")
    print(f"   - Added {len(run_df)} session predictions")
    print(f"   - Total runs in consolidated file: {consolidated_df['run_number'].nunique()}")

if __name__ == "__main__":
    process_and_predict_trust_game()
