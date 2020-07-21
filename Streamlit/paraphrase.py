import streamlit as st
import requests
import json


def main():

    # Heading
    st.header("Paraphrase Generator")

    # Text area for user input
    user_input = st.text_area("Enter your text here", "Enter you text here and hit Phrase It")

    # Select Box for the tokenizer
    tokenizer_name = st.sidebar.selectbox("Tokenizer", ("T5-small", "T5-base"))

    # Select Box for decoding strategies
    decoding_strategy = st.sidebar.selectbox("Decoding Methods",
                                             ("Greedy Decoding", "Beam Search", "Top-k, Top-p sampling"))

    # Slider for max_len
    max_len = st.sidebar.slider("Max-Length", 0, 512, 256)

    # Calling get_Sliders() to get all the decoding parameters
    decoding_params = get_sliders(decoding_strategy, max_len)

    # Setting additional decoding params
    decoding_params["tokenizer"] = tokenizer_name
    decoding_params["max_len"] = max_len
    decoding_params["strategy"] = decoding_strategy

    # Number of beams should be always greater than or equal to the number of return sequences
    if decoding_strategy == "Beam Search":
        if decoding_params["beams"] > 0:
            num_return_sequences = st.sidebar.slider("Number of return sentences", 0, decoding_params["beams"])
        else:
            num_return_sequences = 5
    else:
        num_return_sequences = st.sidebar.slider("Number of return sentences", 0, 10)

    decoding_params["return_sen_num"] = num_return_sequences

    # Phrase it button
    if st.button("Phrase It"):

        # Checking for exceptions
        if not check_exceptions(decoding_params):

            # Calling the forward method on click of Phrase It
            with st.spinner('T5 is processing your text ... '):
                output = forward(user_input, decoding_params)

                # Writing the output
                for i in range(len(output)):
                    st.write(output[i])


def get_sliders(decoding_strategy, max_len):
    params = {}

    # Setting different parameters for Beam Search and top-p top-k sampling

    if decoding_strategy == "Beam Search":
        beams_no = st.sidebar.slider("No Of Beams", 0, 10, 2)
        no_repeat_ngram_size = st.sidebar.slider("N-Gram Size", 0, 10)
        params["beams"] = beams_no
        params["ngram"] = no_repeat_ngram_size

    elif decoding_strategy == "Top-k, Top-p sampling":
        top_p = st.sidebar.slider("Top-P", 0.0, 1.0)
        top_k = st.sidebar.slider("Top-K", 0, max_len)
        params["top_p"] = top_p
        params["top_k"] = top_k

    return params


def forward(sentence, decoding_params):
    # Making the request to the backend
    headers = {"content-type": "application/json"}
    r = requests.post("http://127.0.0.1:5000/run_forward", headers=headers,
                      data=json.dumps({'sentence': sentence, 'decoding_params': decoding_params}))
    data = r.json()
    return data["data"]


def check_exceptions(decoding_params):
    # Checking for zero on the num of return sequences
    if decoding_params["return_sen_num"] == 0:
        st.error("Please set the number of return sequences to more than one")
        return True
    return False


if __name__ == "__main__":
    main()
