# Paraphrase-Generator

This is a paraphrase generator built using Streamlit API for the front end and its running on flask backend. 
The model used is the T5ForConditionalGeneration from the hugging face library, which is trained on the PAWS dataset of paraphrased questions.The model is uploaded on the transformers model hub under the name [Vamsi/T5_Paraphrase_Paws](https://huggingface.co/Vamsi/T5_Paraphrase_Paws).
This API has controls over the decoding methods of the decoder in the T5 model and various parameters that can be fine tuned.


# How to use
​
PyTorch and TF models available
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
