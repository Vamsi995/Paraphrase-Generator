# Paraphrase-Generator

This is a paraphrase generator built using Streamlit API for the front end and its running on flask backend. 
The model used is the T5ForConditionalGeneration from the hugging face library, which is trained on the QQP dataset of paraphrased questions.
This API has controls over the decoding methods of the decoder in the T5 model and various parameters that can be fine tuned.