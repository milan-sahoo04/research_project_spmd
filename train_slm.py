import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# ==============================
# LOAD QA DATA
# ==============================

df = pd.read_csv("data/labeled/qa_text.csv")

questions = df["question"].tolist()
answers = df["answer"].tolist()

# ==============================
# LOAD MODEL
# ==============================

model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode questions
question_embeddings = model.encode(questions)

# ==============================
# SAVE MODEL DATA
# ==============================

data = {
    "questions": questions,
    "answers": answers,
    "embeddings": question_embeddings
}

with open("models/text_retriever.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Text model trained and saved!")