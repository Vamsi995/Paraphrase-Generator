from flask import Flask, request
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import random

from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

app = Flask(__name__)

output_cache = []

# Selecting the tokenizer
def select_tokenizer(tokenizer_name):
    if tokenizer_name == "t5-small":
        tokenizer = T5Tokenizer.from_pretrained('T5-small')
    else:
        tokenizer = T5Tokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws')
    return tokenizer


def run_model(sentence, decoding_params, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    text = "paraphrase: " + sentence + " </s>"

    max_len = decoding_params["max_len"]

    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    if decoding_params["strategy"] == "Greedy Decoding":
        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=max_len,
        )
    elif decoding_params["strategy"] == "Beam Search":
        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=max_len,
            num_beams=decoding_params["beams"],
            no_repeat_ngram_size=decoding_params["ngram"],
            early_stopping=True,
            temperature=decoding_params["temperature"],
            num_return_sequences=decoding_params["return_sen_num"]  # Number of sentences to return
        )
    else:
        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=max_len,
            top_k=decoding_params["top_k"],
            top_p=decoding_params["top_p"],
            early_stopping=True,
            # temperature=decoding_params["temperature"],
            num_return_sequences=decoding_params["return_sen_num"]  # Number of sentences to return
        )

    return beam_outputs


def preprocess_output(model_output, tokenizer, temp, sentence, decoding_params, model):
    for line in model_output:
        paraphrase = tokenizer.decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if paraphrase.lower() != sentence.lower() and paraphrase not in temp:
            temp.append(paraphrase)

    if len(temp) < decoding_params["return_sen_num"]:
        sentence = temp[random.randint(0, len(temp) - 1)]
        model_output = run_model(sentence, decoding_params, tokenizer, model)
        temp = preprocess_output(model_output, tokenizer, temp, sentence, decoding_params, model)

    return temp


@app.route("/run_forward", methods=["POST"])
def forward():
    params = request.get_json()
    sentence = params["sentence"]
    decoding_params = params["decoding_params"]

    tokenizer_name = decoding_params["tokenizer"]
    model = T5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws')
    tokenizer = select_tokenizer(tokenizer_name)

    model_output = run_model(sentence, decoding_params, tokenizer, model)

    paraphrases = []
    temp = []

    temp = preprocess_output(model_output, tokenizer, temp, sentence, decoding_params, model)

    global output_cache
    output_cache = temp

    for i, line in enumerate(temp):
        paraphrases.append(f"{i + 1}. {line}")

    return {"data": paraphrases}


@app.route("/embedding", methods=["POST"])
def embedding():
    params = request.get_json()

    sentence = params["sentence"]
    paraphrased_sentences = output_cache

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model_USE = hub.load(module_url)

    embedding_vectors = model_USE(paraphrased_sentences)
    print(embedding_vectors.numpy().tolist())

    return {"data": embedding_vectors.numpy().tolist(), "paraphrased": paraphrased_sentences}

if __name__ == "__main__":
    app.run()
