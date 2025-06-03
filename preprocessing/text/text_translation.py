#!/usr/bin/env python3
import os
import sys
import pandas as pd
from openai import OpenAI
from openai import OpenAIError
import time

OPENAI_API_KEY_FILE = "openai_api_key.txt"
OPENAI_MODEL = "gpt-4.1-nano"

client = OpenAI(api_key="")

def translate_text(text, target_language="English", model="gpt-4.1-nano"):
    prompt = f"Translate the following text to {target_language} and extract only the title:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        print("OpenAI Error: {}".format(e))
        return ""

def translate_description(X: pd.Series):
    output = []
    rows = X.shape[0]
    for i, des in enumerate(X):
        translated_text = translate_text(des, model=OPENAI_MODEL)
        output.append(translated_text)
        print("Translation row {}/{}: {}".format(i, rows, translated_text))
        time.sleep(1)
    return pd.Series(data=output, index=X.index)

def main():
    if len(sys.argv) < 2:
        print("Please provide file of data to be translated.")
        return 0
    file = sys.argv[1]

    file_to_save = os.path.splitext(file)[0]+"_translated.csv"
    df = pd.read_csv(file, index_col=0)

    # Set OPENAI API key
    if not os.path.exists(OPENAI_API_KEY_FILE):
        print(f"Please create a file named {OPENAI_API_KEY_FILE} that contains your OPENAI API key.")
        return
    api_key = ""
    with open(OPENAI_API_KEY_FILE, 'r') as file:
        api_key = file.read().strip()
    client.api_key = api_key

    # Example usage
    df['translated_text'] = translate_description(df['description'])
    print("Translated text:", df['translated_text'])

    # save to csv
    df.to_csv(file_to_save)

if __name__=="__main__":
    main()