# wan/utils/prompt_parser.py

def normalize_prompt(prompt):
    if isinstance(prompt, str):
        return prompt.strip()

    if isinstance(prompt, dict):
        if "prompt_for_wan_one_t2v" in prompt:
            return prompt["prompt_for_wan_one_t2v"].strip()

        scenes = prompt.get("scenes", [])
        texts = []
        for s in scenes:
            actions = ", ".join(s.get("actions", []))
            mood = s.get("mood", "")
            texts.append(f"{actions}. Mood: {mood}")

        return " ".join(texts)

    raise TypeError("Prompt must be string or dict")
