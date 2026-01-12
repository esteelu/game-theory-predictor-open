## General Pipeline

![Flow](Flow.png)

## Project Structure

- **instructions/** - This folder contains system and user instructions for the LLM predictor.
  System instructions: Configuration and setup instructions for the system
  User instructions: Guidelines for users on what to do in each task
- **agent_pool/** - This folder contains the agent configuration.
- **predictions/** - This folder hosts the prediction and prelimiary analysis scripts for the LLM predictor.
  `structured_prompt_loader*.py` - Making API calls
  `predict_*.py` - Making predictions
  `analyze_*.py` - Making preliminary analysis

## Execution
Execute `predict_*.py` and `analyze_*.py` for generating predictions and preliminary summary statistics.
**Example (MEG)**: Execute prediction_minimum_effort.py, follow by analyze_minimum_effort.py

## Outputs
Note: The output filenames in this package have been standardized for ease of future use. These names may differ from those used during development, but the structure is unchanged.

This code is licensed under the MIT License.

