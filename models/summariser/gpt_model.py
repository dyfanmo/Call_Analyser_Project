from models.gpt_prompts import call_summary_instructions


def get_gpt_call_summary(client, prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": call_summary_instructions}, {"role": "user", "content": prompt}],
    )

    return completion
