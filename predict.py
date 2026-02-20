import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# ==============================
# LOAD TEXT KNOWLEDGE BASE
# ==============================

text_data = [
"Vehicular emissions have increased significantly in Bhubaneswar causing higher AQI levels.",
"Industrial smoke from thermal plants is affecting air quality in Angul.",
"Traffic congestion during peak hours raises pollution in Cuttack.",
"Coal mining activities contribute to airborne dust in Talcher region.",
"Steel plant emissions are under monitoring in Rourkela.",
"Sea breeze helps in reducing particulate concentration in Puri.",
"Crop residue burning impacts western Odisha districts.",
"AQI levels remained stable due to rainfall in Sambalpur.",
"Construction dust is increasing PM2.5 in Berhampur.",
"Government issued warnings due to rising AQI in Bhubaneswar."
]

# ==============================
# LOAD TABLE DATA
# ==============================

aqi_data = pd.read_csv("data/labeled/air_quality.csv")

# ==============================
# LOAD EMBEDDING MODEL
# ==============================

model = SentenceTransformer('all-MiniLM-L6-v2')

text_embeddings = model.encode(text_data, convert_to_tensor=True)

# ==============================
# QUESTION LOOP
# ==============================

while True:
    question = input("\nAsk question (or 'exit'): ")

    if question.lower() == "exit":
        break

    # -------- TEXT ANSWER --------

    q_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, text_embeddings)[0]
    best_idx = torch.argmax(scores).item()

    print("\n🧠 Text Answer:")
    print(text_data[best_idx])
    print(f"Confidence: {scores[best_idx]:.2f}")

    # -------- TABLE ANSWER --------

    print("\n📊 Table Insights:")

    # Highest pollution tolerant plant
    best_plant = aqi_data.loc[aqi_data["APTI"].idxmax()]

    print("\n🌿 Most Pollution-Tolerant Plant:")
    print(best_plant)

    # Show plants with GOOD grade
    good_plants = aqi_data[aqi_data["Grade"] == "Good"]

    print("\n🌱 Plants with GOOD tolerance:")
    print(good_plants[["Plant_Species", "APTI", "Grade"]])