import streamlit as st
import requests
import json


def main():
    user_input = st.text_area("Enter your text here", None)
    decoding_strategy = st.sidebar.selectbox(("Decoding Methods"),("Greedy Decoding", "Beam Search", "Top-k, Top-p sampling"))
    max_len = st.sidebar.slider("Max-Length",0,512,256)
    decoding_params = get_Sliders(decoding_strategy,max_len)
    decoding_params["max_len"] = max_len
    decoding_params["strategy"] = decoding_strategy
    if(decoding_strategy == "Beam Search"):
        if(decoding_params["beams"] > 0):
            num_return_sequences = st.sidebar.slider("Number of return sentences",0,decoding_params["beams"])
        else:
            num_return_sequences = 5;
    else:
        num_return_sequences = st.sidebar.slider("Number of return sentences",0,10)

    decoding_params["return_sen_num"] = num_return_sequences

    if st.button("Phrase It"):
        output = forward(user_input, decoding_params)
        for i in range(len(output)):
            st.write(output[i])




def get_Sliders(decoding_strategy, max_len):
    params = {}
    if(decoding_strategy == "Beam Search"):
        beams_no = st.sidebar.slider("No Of Beams",0,10,2)
        no_repeat_ngram_size = st.sidebar.slider("N-Gram Size",0,10)
        params["beams"] = beams_no
        params["ngram"] = no_repeat_ngram_size

    elif(decoding_strategy == "Top-k, Top-p sampling"):
        top_p = st.sidebar.slider("Top-P",0.0,1.0)
        top_k = st.sidebar.slider("Top-K",0,max_len)
        params["top_p"]=top_p
        params["top_k"]=top_k

    return params

def forward(sentence,decoding_params):
    headers = {"content-type":"application/json"}
    r = requests.post("http://127.0.0.1:5000/run_forward",headers=headers,data=json.dumps({'sentence': sentence, 'decoding_params': decoding_params}))
    data = r.json()
    return data["data"]

if __name__ == "__main__":
    main()


