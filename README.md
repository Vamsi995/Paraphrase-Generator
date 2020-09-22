# Paraphrase Generator with T5

A Paraphrase-Generator built using transformers which takes an English sentence as an input and produces a set of paraphrased sentences.
This is an NLP task of conditional text-generation. The model used here is the [T5ForConditionalGeneration](https://huggingface.co/transformers/model_doc/t5.html#t5forconditionalgeneration) from the [huggingface transformers](https://huggingface.co/transformers)  library. 
This model is trained on the [Google's PAWS Dataset](https://github.com/google-research-datasets/paws) and the model is saved in the transformer model hub of hugging face library under the name [Vamsi/T5_Paraphrase_Paws](https://huggingface.co/Vamsi/T5_Paraphrase_Paws).


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Streamlit library
- Huggingface transformers library
- Pytorch
- Tensorflow 

### Installing

- Streamlit

```
$ pip install streamlit
```

- Huggingface transformers library
```
$ pip install transformers
```

- Tensorflow
```
$ pip install --upgrade tensorflow
```

- Pytorch 
```
Head to the docs and install a compatible version
https://pytorch.org/
```

## Running the web app 

- Clone the repository
```
$ git clone [repolink] 
```
- Running streamlit app
```
$ cd Streamlit

$ streamlit run paraphrase.py
```
- Running the flask app
```
$ cd Server

$ python server.py
```

The initial server call will take some time as it downloads the model parameters. The later calls will be relatively faster as it will store the model params in the cahce.


![](/Images/Paraphrase.png)


![](/Images/TextualSimilarity.png)


## General Usage
PyTorch and TF models are available
​
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
​
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
​
sentence = "This is something which i cannot understand at all"

text =  "paraphrase: " + sentence + " </s>"

encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")


outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    do_sample=True,
    top_k=200,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=5
)

for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print(line)
```







## Built With

* [Streamlit](https://www.streamlit.io/) - Fastest way for building data apps
* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - Backend framework
* [Transformers-Huggingface](https://huggingface.co/) - On a mission to solve NLP, one commit at a time. Transformers Library.


## Authors
- [Sai Vamsi Alisetti](https://github.com/Vamsi995)

## Acknowledgments
- [Sampath Kethineedi](https://github.com/sampathkethineedi)
