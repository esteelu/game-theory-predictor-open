# Prediction of Prisoners' Dilemma Games

import pandas as pd
import json
import re
import os
import sys

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- API Call Function ---
from host_pool import get_gpt4mini_client

def get_structured_prediction(system_message, user_message):
    """
    Gets a prediction from the AI using system and user messages with JSON output.
    """
    print("      -> Calling AI model for prediction...")
    client = get_gpt4mini_client()
    
    try:
        completion = client.chat.completions.create(
            model="gpt-5",
            temperature=1,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"      ❌ API call failed: {e}")
        return f'{{"error": "API call failed: {str(e)}"}}'
# ------------------------------------------------------------------------------

# --- Import Communication Texts and Prompts ---
EXCEL_FILE = 'merged_table_cason_2019.xlsx'
SYSTEM_PROMPT_FILE = 'instructions/ipd_system_message_prompt_minimal.txt'
USER_PROMPT_TEMPLATE_FILE = 'instructions/ipd_user_message_template_minimal.txt'
RAW_PREDICTIONS_FILE = None
FINAL_ANALYTICAL_REPORT_FILE = None

# --- Column Names from Raw Excel ---
GAME_ID_COLS = ['session', 'Cluster.x']
SUBGROUP_COL = 'Subgroup.x'
TREATMENT_COL = 'Treatment'
TASK_COL = 'task'
SENDER_COL = 'Sender'
MESSAGE_COL = 'texttype'
VOTE_COL = 'T3_Vote'

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def intelligent_parse(raw_json_text):
    """Robustly finds and decodes a JSON object from the AI's raw output."""
    if not isinstance(raw_json_text, str):
        return None
    match = re.search(r'\{.*\}', raw_json_text, re.DOTALL)
    if not match:
        print("      ERROR: No JSON object found in the AI response.")
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        print("      ERROR: Failed to decode the JSON object from the AI response.")
        return None

def normalize_vote(vote):
    """Converts various vote formats to 'Cooperate' or 'Defect'."""
    if not isinstance(vote, str):
        return 'N/A'
    vote_lower = vote.strip().lower()
    if vote_lower in ['m', 'cooperate', 'coop']:
        return 'Cooperate'
    if vote_lower in ['j', 'defect']:
        return 'Defect'
    return 'N/A'

# ==============================================================================
# --- 3. MAIN ANALYSIS PIPELINE ---
# ==============================================================================

def run_analysis(target_session=None, run_number=None):
    """
    Main pipeline to load data, run predictions, and generate the final report.
    """
    global RAW_PREDICTIONS_FILE, FINAL_ANALYTICAL_REPORT_FILE
    if run_number is not None:
        RAW_PREDICTIONS_FILE = f'minimal_raw_ai_predictions_run{run_number}.csv'
        FINAL_ANALYTICAL_REPORT_FILE = f'final_full_analytical_report_task2_minimal_run{run_number}.csv'
    else:
        RAW_PREDICTIONS_FILE = 'minimal_raw_ai_predictions.csv'
        FINAL_ANALYTICAL_REPORT_FILE = 'final_full_analytical_report_task2_minimal.csv'
    
    # 1. Load Data and Prompts
    print("--- 1. Loading data and prompt files ---")
    try:
        df = pd.read_excel(EXCEL_FILE)
        with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
            system_message = f.read()
        with open(USER_PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            user_template = f.read()
        print("Files loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Make sure '{e.filename}' exists. You must create the prompt files.")
        return

    # 2. Prepare and Filter Data
    print("--- 2. Preparing and filtering master data ---")
    relevant_df = df[df[TREATMENT_COL] == 2].copy()
    relevant_df['session'] = relevant_df['session'].astype(str)
    relevant_df['Cluster.x'] = pd.to_numeric(relevant_df['Cluster.x'], errors='coerce').astype('Int64')
    relevant_df['game_id'] = relevant_df[GAME_ID_COLS].astype(str).agg('_'.join, axis=1)
    relevant_df[SUBGROUP_COL] = pd.to_numeric(relevant_df[SUBGROUP_COL], errors='coerce').astype('Int64')
    print(f"Data prepared. Found {relevant_df['game_id'].nunique()} unique games (sessions).")

    # 3. Process Each Game Symmetrically
    print("--- 3. Generating predictions for each game ---")
    all_results = []
    all_raw_predictions = []

    grouped_games = relevant_df.groupby('game_id')
    
    # Filter to target session if specified, for checking purpose
    if target_session:
        print(f"🎯 TARGET MODE: Only processing session {target_session}")
        grouped_games = [(gid, gdata) for gid, gdata in grouped_games if gid == target_session]
        if not grouped_games:
            print(f"Target session {target_session} not found!")
            return
    
    for game_id, game_data in grouped_games:
        print(f"\n🎮 Processing Game: {game_id}")
        subgroups = sorted(game_data[SUBGROUP_COL].unique())
        if len(subgroups) != 2:
            print(f"Skipping {game_id} - found {len(subgroups)} subgroups instead of 2")
            continue
        
        team_a_subgroup, team_b_subgroup = subgroups[0], subgroups[1]

        for focal_subgroup, opponent_subgroup in [(team_a_subgroup, team_b_subgroup), (team_b_subgroup, team_a_subgroup)]:
            session_id = f"{game_id}_{focal_subgroup}"
            print(f"Processing Perspective: {session_id}")

            # A. Format Inputs for the AI
            team1_players = sorted(game_data[game_data[SUBGROUP_COL] == focal_subgroup][SENDER_COL].unique())
            team2_players = sorted(game_data[game_data[SUBGROUP_COL] == opponent_subgroup][SENDER_COL].unique())
            focal_chat_data = game_data[(game_data[SUBGROUP_COL] == focal_subgroup) & (game_data[TASK_COL] == 3)]
            intergroup_chat_data = game_data[game_data[TASK_COL] == 4]
            team1_chat_logs = "\n".join([f"Player {row[SENDER_COL]}: {row[MESSAGE_COL]}" for _, row in focal_chat_data.iterrows()])
            intergroup_chat_logs = "\n".join([f"Player {row[SENDER_COL]}: {row[MESSAGE_COL]}" for _, row in intergroup_chat_data.iterrows()])
            user_message = user_template.format(
                TEAM1_PLAYER_IDS=", ".join(map(str, team1_players)),
                TEAM2_PLAYER_IDS=", ".join(map(str, team2_players)),
                TEAM1_CHAT_LOGS=team1_chat_logs,
                INTERGROUP_CHAT_LOGS=intergroup_chat_logs
            )
            
            # B. Get AI Prediction
            ai_response_text = get_structured_prediction(system_message, user_message)

            # Store the raw prediction before parsing
            all_raw_predictions.append({'session_id': session_id, 'raw_prediction_text': ai_response_text})
            
            parsed_info = intelligent_parse(ai_response_text)
            
            if not parsed_info:
                print(f"SKIPPING session {session_id} due to parsing failure.")
                continue
            
            # C. Get Ground Truth for the Opponent Team
            opponent_actual_votes = {}
            
            for p_id in team2_players:
                # Search for this player's vote across all tasks in this game
                vote_series = game_data[game_data[SENDER_COL] == p_id][VOTE_COL].dropna()
                if not vote_series.empty:
                    vote_value = vote_series.iloc[0]
                    opponent_actual_votes[str(p_id)] = vote_value
                    print(f"Debug: Player {p_id} vote: {vote_value} -> normalized: {normalize_vote(vote_value)}")
                else:
                    print(f"WARNING: No vote found for player {p_id} in any task")
                    opponent_actual_votes[str(p_id)] = 'N/A'

            # Calculate team outcome using RAW votes before normalization
            raw_coop_votes = sum(1 for vote in opponent_actual_votes.values() 
                               if isinstance(vote, str) and vote.strip().lower() in ['m', 'cooperate', 'coop'])
            actual_team_outcome = 'Cooperate' if raw_coop_votes >= 2 else 'Defect'
            print(f"      Debug: Raw votes for team outcome: {raw_coop_votes} cooperate votes -> {actual_team_outcome}")

            # D. Parse AI Predictions and Match to Ground Truth
            ai_preds = parsed_info.get('team2_player_predictions', [])
            ai_final_pred = parsed_info.get('team2_final_prediction', {})

            # Compute predicted team outcome from individual predictions
            pred_coop_votes = sum(1 for ai_p in ai_preds if normalize_vote(ai_p.get('predicted_vote')) == 'Cooperate')
            pred_defect_votes = sum(1 for ai_p in ai_preds if normalize_vote(ai_p.get('predicted_vote')) == 'Defect')
            if pred_coop_votes + pred_defect_votes > 0:
                predicted_team_outcome = 'Cooperate' if pred_coop_votes >= 2 else 'Defect'
            else:
                predicted_team_outcome = 'N/A'

            for p_id_actual in team2_players:
                p_id_str = str(p_id_actual)
                actual_vote = opponent_actual_votes.get(p_id_str, 'N/A')
                predicted_vote, reasoning = 'N/A', 'Player not found in prediction'
                for ai_p in ai_preds:
                    if str(ai_p.get('player_id')) == p_id_str:
                        predicted_vote = ai_p.get('predicted_vote', 'N/A')
                        reasoning = ai_p.get('prediction_reasoning', 'N/A')
                        break

                # E. Store all data for the final report
                result_row = {
                    'session_id': session_id,
                    'game_id': game_id,
                    'focal_team_id': focal_subgroup,
                    'opponent_player_id': p_id_actual,
                    'actual_individual_vote': normalize_vote(actual_vote),
                    'predicted_individual_vote': normalize_vote(predicted_vote),
                    'individual_prediction_correct': 1 if normalize_vote(actual_vote) == normalize_vote(predicted_vote) else 0,
                    'actual_team_outcome': actual_team_outcome,
                    'predicted_team_outcome': predicted_team_outcome,
                    'team_prediction_correct': 1 if actual_team_outcome == predicted_team_outcome else 0,
                    'ai_reasoning_for_player': reasoning,
                    'ai_team_prediction_explanation': ai_final_pred.get('explanation', 'N/A')
                }
                all_results.append(result_row)
    
    # 4. Create and Save ALL Report Files
    print(f"--- 4. Creating and saving minimal report files ---")
    
    # 4a. Save the Raw Predictions File
    if all_raw_predictions:
        raw_df = pd.DataFrame(all_raw_predictions)
        raw_df.to_csv(RAW_PREDICTIONS_FILE, index=False)
        print(f"Raw AI predictions saved to '{RAW_PREDICTIONS_FILE}'")
    
    # 4b. Save the Final Analytical Report
    if not all_results:
        print("No results were generated.")
        return
        
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(FINAL_ANALYTICAL_REPORT_FILE, index=False)
    print(f"Final analytical report saved to '{FINAL_ANALYTICAL_REPORT_FILE}'")

    # 5. Calculate and Display Accuracy
    print("--- 5. Final Accuracy Analysis ---")
    total_individual_choices = len(final_df)
    correct_individual_choices = final_df['individual_prediction_correct'].sum()
    individual_accuracy = (correct_individual_choices / total_individual_choices * 100) if total_individual_choices > 0 else 0

    team_level_df = final_df.drop_duplicates(subset=['session_id'])
    total_team_predictions = len(team_level_df)
    correct_team_predictions = team_level_df['team_prediction_correct'].sum()
    team_accuracy = (correct_team_predictions / total_team_predictions * 100) if total_team_predictions > 0 else 0
    
    print("\n" + "="*40)
    print("MINIMAL ACCURACY REPORT")
    print("="*40)
    print(f"Total Team-Level Predictions (Perspectives): {total_team_predictions}")
    print(f"Correct Team-Level Predictions:              {correct_team_predictions}")
    print(f"TEAM ACCURACY:                               {team_accuracy:.2f}%")
    print("-"*40)
    print(f"Total Individual Player Predictions:         {total_individual_choices}")
    print(f"Correct Individual Player Predictions:       {correct_individual_choices}")
    print(f"INDIVIDUAL VOTE ACCURACY:                    {individual_accuracy:.2f}%")
    print("="*40)

    # 6. DATA VALIDATION CHECK
    print("--- 6. Data Validation: Checking for Missing Predictions ---")
    
    missing_individual_preds_df = final_df[final_df['predicted_individual_vote'] == 'N/A']
    if not missing_individual_preds_df.empty:
        print(f"WARNING: Found {len(missing_individual_preds_df)} missing individual player predictions.")
        print("Sessions and players with missing predictions:")
        print(missing_individual_preds_df[['session_id', 'opponent_player_id']].to_string(index=False))
    else:
        print("No missing individual player predictions found.")

    missing_team_preds_df = team_level_df[team_level_df['predicted_team_outcome'] == 'N/A']
    if not missing_team_preds_df.empty:
        print(f"WARNING: Found {len(missing_team_preds_df)} missing overall team predictions.")
        print("Sessions with missing team predictions:")
        print(missing_team_preds_df[['session_id']].to_string(index=False))
    else:
        print("No missing overall team predictions found.")
    
    print("--- Minimal Analysis Complete ---")


if __name__ == "__main__":
    # Allow specifying a target session and/or run number from command line
    target_session = None
    run_number = None
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if '_' in arg1:
            target_session = arg1
            print(f"Running in minimal target mode for session: {target_session}")
            if len(sys.argv) > 2:
                try:
                    run_number = int(sys.argv[2])
                    print(f"Run number: {run_number}")
                except ValueError:
                    print("Warning: Second argument should be a run number (integer)")
        else:
            try:
                run_number = int(arg1)
                print(f"Run number: {run_number}")
            except ValueError:
                print("Warning: Argument should be a run number (integer)")
    run_analysis(target_session, run_number)

