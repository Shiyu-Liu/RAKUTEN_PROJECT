#!/usr/bin/env python3
import os
import sys
import re
import math
from openai import OpenAI
from openai import OpenAIError
from typing import List
import pandas as pd
from text_translation import OPENAI_API_KEY_FILE, OPENAI_MODEL
from text_preprocessing import Filtering_Params

client = OpenAI(api_key="")
MINIMUM_WORDS=5  # minimum number of words that the data must contain to be used for text augmentation

def augment_text(text, samples_per_text: int = 3, model="gpt-4.1-nano") -> List[str]:
    augmented = []
    prompt = (f"Paraphrase the following sentence in {samples_per_text} different ways. "
        f"Ensure that the meaning is preserved and the sentence remains natural and fluent.\n\n"
        f"Original: {text}"
    )
    paraphrases = []
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=150,
        )
        content = response.choices[0].message.content
        paraphrases = [line.strip("- ").strip() for line in content.strip().split("\n") if line.strip()]
        augmented.extend(paraphrases)
    except OpenAIError as e:
        print(f"OpenAI Error: {e}")
    if paraphrases:
        for para in paraphrases:
            print(f"- {para}")
    return augmented

def augment_minority_class(data: pd.DataFrame, samples_per_data: int):
    new_data = pd.DataFrame(columns=['raw_text', 'raw_translation', 'text', 'prdtypecode'])
    for idx, text in zip(data.index, data['text']):
        augmented = augment_text(text, samples_per_data, OPENAI_MODEL)
        if len(augmented)==0:
            print(f"Warning: the following text has no new data generated\n{text}")
            continue
        for aug in augmented:
            d = data.loc[idx]
            aug = re.sub(r'^\s*\d+\.*\s*', '', aug)
            aug = re.sub(r'[^a-zA-Z\s\']', '', aug)
            aug = aug.lstrip().rstrip()
            new_data.loc[len(new_data)] = [d['raw_text'], d['raw_translation'], aug, d['prdtypecode']]
    return new_data

def main():
    if len(sys.argv) < 2:
        print("Please provide file of data to be augmented.")
        return 0
    file = sys.argv[1]

    file_to_save = os.path.splitext(file)[0]+"_augmented.csv"
    df = pd.read_csv(file, index_col=0, delimiter=';')

    # set OPENAI API key
    if not os.path.exists(OPENAI_API_KEY_FILE):
        print(f"Please create a file named {OPENAI_API_KEY_FILE} that contains your OpenAI API key.")
        return 0
    api_key = ""
    with open(OPENAI_API_KEY_FILE, 'r') as file:
        api_key = file.read().strip()
    client.api_key = api_key

    # find the minority classes
    target_size = Filtering_Params['target_sample_size']
    class_dist = df['prdtypecode'].value_counts()
    minority_class = class_dist[class_dist.values < target_size]
    print("Minority class:\n", minority_class)

    prg_idx = 1
    total_class = minority_class.shape[0]
    last_index = df.index[-1]
    augmented_data = pd.DataFrame()
    for index, value in zip(minority_class.index, minority_class.values):
        number_to_augment = target_size-value
        class_to_augment = df[df['prdtypecode']==index]
        data_to_augment = class_to_augment[class_to_augment['text'].apply(lambda x: len(x.split()))>MINIMUM_WORDS]
        number_of_data = data_to_augment.shape[0]
        print(f"\n\nAugmenting minority class {index}... progress: {prg_idx}/{total_class}")
        prg_idx +=1
        if number_of_data >= number_to_augment:
            data = data_to_augment.sample(n=number_to_augment, random_state=27)
            samples_per_data = 1
        else:
            data = data_to_augment
            samples_per_data = math.ceil(number_to_augment/number_of_data)
        new_samples = augment_minority_class(data, samples_per_data)
        augmented_data = pd.concat([augmented_data, new_samples], axis=0)

    augmented_data.index = range(last_index+1, last_index+augmented_data.shape[0]+1)
    augmented_data.to_csv(file_to_save, sep=';')

if __name__ == "__main__":
    main()