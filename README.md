# NLP Chatbot using TF-IDF and Cosine Similarity

A simple retrieval-based chatbot built with Python and Jupyter Notebook. It matches a user's message against questions in a dialogue dataset and returns the answer associated with the most similar question.

## Features

- Loads question-and-answer pairs from a tab-separated dialogue dataset.
- Cleans text by converting it to lowercase, removing most punctuation, and normalizing whitespace.
- Converts questions into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.
- Uses **cosine similarity** to find the question most similar to the user's message.
- Applies a similarity threshold to avoid returning answers when no sufficiently close match is found.
- Includes sample tests and an interactive command-line chat loop.
- Saves the trained vectorizer and related data to `chatbot_model.pkl` with `joblib`.

## How it works

1. **Load data:** Read the dataset as two columns: `question` and `answer`.
2. **Preprocess text:** Normalize questions with a small text-cleaning function.
3. **Vectorize questions:** Fit a `TfidfVectorizer` using unigrams and bigrams (`ngram_range=(1, 2)`).
4. **Match a message:** Transform the user's message using the fitted vectorizer.
5. **Calculate similarity:** Compute cosine similarity between the message vector and all stored question vectors.
6. **Choose a response:** Return the answer for the highest-scoring question if its score meets the threshold (default `0.25`); otherwise, return a fallback message.

> This is a retrieval-based chatbot, not a generative AI model. It selects an existing answer from the dataset rather than composing a new answer.

## Project files

```text
Chatbot-Project/
├── chatbot_project.ipynb
├── README.md
├── requirements.txt
└── dialogs.txt                 # required dataset; include only if you can share it
```

The notebook may save `chatbot_model.pkl` after running the model-saving cell. This file can be generated locally and usually does not need to be committed to GitHub.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- pandas
- scikit-learn
- joblib

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset setup

The notebook expects a tab-separated text file named `dialogs.txt` with two fields per line:

```text
question<TAB>answer
```

Place `dialogs.txt` in the same folder as the notebook. If the dataset is not yours to redistribute, do not upload it publicly; instead, document where it can be obtained and how to place it in the project folder.

**Important:** Update the notebook's original machine-specific Windows path before sharing. For example, use:

```python
DATA_PATH = "dialogs.txt"
```

This relative path works when the notebook is run from the project directory.

## Run the project

1. Clone or download this repository.
2. Open a terminal in the project directory and install the requirements.
3. Ensure `dialogs.txt` is present in the project directory and its format matches the description above.
4. Open `chatbot_project.ipynb` in Jupyter Notebook or JupyterLab.
5. Run the notebook cells in order.
6. Use the interactive chat cell to enter messages. Type `exit`, `quit`, or `bye` to stop the chat.

The notebook also includes example questions for basic testing and cells to save and reload the trained model.

## Configuration

The chatbot function uses a default similarity threshold of `0.25`:

```python
chatbot_reply(message, threshold=0.25)
```

- A higher threshold makes the chatbot more selective and may increase fallback responses.
- A lower threshold allows weaker matches and may return less relevant answers.

The best threshold depends on the dataset and should be tested with representative questions.

## Limitations

- Responses are limited to answers present in the dataset.
- Similar wording does not always imply the same meaning; TF-IDF is primarily lexical and does not understand context like a modern language model.
- Results depend on the quality, coverage, and formatting of the dataset.
- The dataset must be available locally for the notebook's data-loading cell to run.

## Future improvements

- Add a web interface using Streamlit or Flask.
- Evaluate response quality on a separate test set.
- Tune the similarity threshold using real examples.
- Add logging and clearer error handling for missing or malformed datasets.
- Add semantic embeddings to improve matching of differently worded questions.

## Author

**Tarun Kataria**

---

If you use or redistribute a dataset, check its licence and provide attribution where required.
