# Making API calls for the Trust Game (Task 3).
import time
import json

def get_structured_game_prediction_system_user(system_message: str, user_message: str):

    from host_pool import get_gpt4mini_client, MODEL_NAME, TEMPERATURE
    max_retries = 3
    initial_wait_time = 2
    print("      -> Calling AI model for Trust Game prediction...")
    client = get_gpt4mini_client()
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                timeout=60.0,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                print(f"All {max_retries} retries failed. Giving up.")
                return f'{{"error": "API call failed after {max_retries} attempts: {str(e)}"}}'
            wait_time = initial_wait_time * (2 ** attempt)
            print(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
    return f'{{"error": "Exited retry loop unexpectedly."}}'
