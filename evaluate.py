from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

model.to("cuda")

sentence = "This is something which i cannot understand at all"

text =  "paraphrase: " + sentence + " </s>"


max_len = 256

encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")


outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True,
    max_length=256,
    top_k=220,
    top_p=1,
    early_stopping=True,
    num_return_sequences=5
)


print ("\nOriginal Question ::")
print (sentence)
print ("\n")
print ("Paraphrased Questions :: ")
final_outputs =[]
for output in outputs:
    sent = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    if sent.lower() != sentence.lower() and sent not in final_outputs:
        final_outputs.append(sent)

for i, final_output in enumerate(final_outputs):
    print("{}: {}".format(i, final_output))