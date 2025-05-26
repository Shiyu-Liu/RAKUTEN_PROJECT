#!/usr/bin/env python3
import os
import sys
from openai import OpenAI
from openai import OpenAIError
from text_translation import OPENAI_API_KEY_FILE

from typing import List
from transformers import pipeline
import torch
from parrot.parrot import Parrot
import warnings
warnings.filterwarnings("ignore")

client = OpenAI(api_key="")

# Generate augmented texts using the paraphraser
def generate_paraphrases(paraphraser, text: str, num_return_sequences: int = 3, num_beams: int = 10) -> List[str]:
    input_text = f"paraphrase: {text} </s>"
    outputs = paraphraser(input_text, max_length=128, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    return list(set([output['generated_text'] for output in outputs]))

def augment_dataset_vamsi(texts: List[str], samples_per_text: int = 3) -> List[str]:
    print("Loading model...")
    augmented = []
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
    for text in texts:
        print(f"\nOriginal: {text}")
        paraphrases = []
        try:
            paraphrases = generate_paraphrases(paraphraser, text, num_return_sequences=samples_per_text)
            augmented.extend(paraphrases)
        except Exception as e:
            print(f"Error generating for text: {text}\n{e}")
        if paraphrases:
            for para in paraphrases:
                print(f"- {para}")
    return augmented

def augment_dataset_parrot(texts: List[str], samples_per_text: int = 5) -> List[str]:
    augmented = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=(device=='cuda'))
    for text in texts:
        print(f"\nOriginal: {text}")
        paraphrases = []
        try:
            paraphrases = parrot.augment(
                input_phrase=text,
                use_gpu=(device == 'cuda'),
                diversity_ranker="levenshtein",
                do_diverse=True,
                max_return_phrases=samples_per_text
            )
        except Exception as e:
            print(f"Error generating for text: {text}\n{e}")
        if paraphrases:
            for para, score in paraphrases:
                print(f"- {para} (score: {round(score, 2)})")
                augmented.append(para)
    return augmented

def augment_dataset_openai(texts: List[str], samples_per_text: int = 3, model="gpt-4.1-nano") -> List[str]:
    augmented = []
    for text in texts:
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
            print("OpenAI Error: {}".format(e))
        if paraphrases:
            for para in paraphrases:
                print(f"- {para}")
    return augmented

def main():
    if len(sys.argv) < 2:
        print("Please provide the text augmentation model to be used: \n"
        "(0: Parrot_Paraphraser_on_T5)\n" \
        "(1: Vamsi/T5_Paraphrase_Paws)\n" \
        "(2: OpenAI/chatgpt-4.1-nano)"
        )
        return 0
    aug_type = int(sys.argv[1])
    if aug_type not in range(3):
        print("Please provide the integer corresponding to text augmentation model to be used: \n"
        "(0: Parrot_Paraphraser_on_T5)\n" \
        "(1: Vamsi/T5_Paraphrase_Paws)\n" \
        "(2: OpenAI/chatgpt-4.1-nano)"
        )
        return 0
    model_type=""
    match aug_type:
        case 0: 
            model_type = "parrot"
        case 1:
            model_type = "vamsi"
        case 2:
            model_type = "openai"

    # Example input texts from a mimic minority class
    minority_texts = [
        "The app keeps crashing when I try to open it.",
        "I can't log into my account with the correct password.",
        "The delivery was delayed without notice.",
    ]

    if model_type=="parrot":
        print("Generating augmented texts using Parrot_Paraphraser_on_T5 model...")
        augmented_texts = augment_dataset_parrot(minority_texts, samples_per_text=3)
    if model_type=="vamsi":
        print("Generating augmented texts using Vamsi/T5_Paraphrase_Paws model...")
        augmented_texts = augment_dataset_vamsi(minority_texts, samples_per_text=3)
    if model_type=="openai":
        # Set OPENAI API key
        if not os.path.exists(OPENAI_API_KEY_FILE):
            print(f"Please create a file named {OPENAI_API_KEY_FILE} that contains your deepl API key.")
            return 0
        api_key = ""
        with open(OPENAI_API_KEY_FILE, 'r') as file:
            api_key = file.read().strip()
        client.api_key = api_key

        print("Generating augmented texts using OpenAI/chatgpt-4.1-nano model...")
        augmented_texts = augment_dataset_openai(minority_texts, samples_per_text=3)

    for i, text in enumerate(augmented_texts, 1):
        print(f"{i}. {text}")

if __name__ == "__main__":
    main()