import streamlit as st
import requests
import json
import numpy as np
import seaborn as sns


def main():
    # Applying styles to the buttons
    st.markdown("""<style>
                        .st-eb {
                            background-color:#F9786F
                        } </style>""", unsafe_allow_html=True)

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
    decoding_params = get_sliders(decoding_strategy, max_len, user_input)

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
        decoding_params["return_sen_num"] = num_return_sequences

    elif decoding_strategy == "Top-k, Top-p sampling":
        num_return_sequences = st.sidebar.slider("Number of return sentences", 0, 10)
        decoding_params["return_sen_num"] = num_return_sequences
        decoding_params["common"] = st.sidebar.slider("Common Words", 0, len(user_input.split(" ")))

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

    if st.button("Semantic Similarity Map"):

        with st.spinner('USE is making your map ...'):
            (embedding_vectors, paraphrased_sentences) = make_map(user_input)
            count = len(paraphrased_sentences)
            label_in = np.arange(count) + 1
            for i, line in enumerate(paraphrased_sentences):
                st.write(f"{i + 1}. {line}")
            run_and_plot(label_in.tolist(), embedding_vectors)
            # print(embedding_vectors)


def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels)
    g.set_title("Semantic Textual Similarity")
    st.pyplot()


def run_and_plot(messages_, message_embeddings_):
    plot_similarity(messages_, message_embeddings_, 90)


def make_map(sentence):
    headers = {"content-type": "application/json"}
    r = requests.post("http://127.0.0.1:5000/embedding", headers=headers,
                      data=json.dumps({'sentence': sentence}))
    data = r.json()

    return data["data"], data["paraphrased"]


def get_sliders(decoding_strategy, max_len, user_input):
    params = {}

    # Setting different parameters for Beam Search and top-p top-k sampling

    if decoding_strategy == "Beam Search":
        beams_no = st.sidebar.slider("No Of Beams", 0, 10, 2)
        no_repeat_ngram_size = st.sidebar.slider("N-Gram Size", 0, 10)
        params["beams"] = beams_no
        params["ngram"] = no_repeat_ngram_size
        params["temperature"] = st.sidebar.slider("Temperature", 0.0, 1.0)

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
    if decoding_params["strategy"] != "Greedy Decoding":

        if decoding_params["return_sen_num"] == 0:
            st.error("Please set the number of return sequences to more than one")
            return True
        return False
    else:
        return False

if __name__ == "__main__":
    main()
