
# import joblib
# import torch
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# # =========================
# # LOAD MODELS
# # =========================
# print("Loading models...")

# aqi_model = joblib.load("models/aqi_model.pkl")
# scaler = joblib.load("models/scaler.pkl")

# tokenizer = AutoTokenizer.from_pretrained("models/slm_model")
# slm_model = AutoModelForSeq2SeqLM.from_pretrained("models/slm_model")

# print("Models loaded successfully.")


# # =========================
# # AQI CATEGORY FUNCTION
# # =========================
# def get_category(aqi):

#     if aqi <= 50:
#         return "Good"
#     elif aqi <= 100:
#         return "Moderate"
#     elif aqi <= 200:
#         return "Poor"
#     elif aqi <= 300:
#         return "Very Poor"
#     else:
#         return "Hazardous"


# # =========================
# # INSIGHT ENGINE
# # =========================
# def generate_insights(data):

#     insights = []

#     if data["wind_speed"] < 2:
#         insights.append("Low wind traps pollutants")

#     if data["humidity"] > 80:
#         insights.append("High humidity keeps particles suspended")

#     if data["PM2.5"] > 150:
#         insights.append("Fine particles extremely high")

#     if data["PM10"] > 200:
#         insights.append("Coarse particles high")

#     if data["temperature"] > 35:
#         insights.append("High temperature accelerates chemical reactions")

#     return ", ".join(insights)


# # =========================
# # SLM GENERATOR
# # =========================
# def generate_explanation(prompt):

#     inputs = tokenizer(prompt, return_tensors="pt")

#     with torch.no_grad():
#         outputs = slm_model.generate(
#             **inputs,
#             max_length=120,
#             temperature=0.7,
#             top_p=0.9
#         )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# # =========================
# # MAIN PREDICTION FUNCTION
# # =========================
# # def predict_air_quality(input_data):

# #     df = pd.DataFrame([input_data])
# #     df.columns = ["Temperature","Humidity","WindSpeed","Ozone","pm_ratio"]

# #     scaled = scaler.transform(df)

# #     aqi = aqi_model.predict(scaled)[0]
# #     aqi = round(aqi, 2)

# #     category = get_category(aqi)

# #     insights = generate_insights(input_data)

# #     # build SLM prompt
# #     prompt = f"""
# #     Analyze air condition.

# #     PM2.5 = {input_data['PM2.5']}
# #     PM10 = {input_data['PM10']}
# #     Humidity = {input_data['humidity']}
# #     Wind Speed = {input_data['wind_speed']}
# #     Temperature = {input_data['temperature']}
# #     AQI = {aqi}
# #     Category = {category}

# #     Environmental observations:
# #     {insights}

# #     Explain air quality in simple terms.
# #     """

# #     explanation = generate_explanation(prompt)

# #     return {
# #         "AQI": aqi,
# #         "Category": category,
# #         "Insights": insights,
# #         "Explanation": explanation
# #     }
# # def predict_air_quality(input_data):

# #     print("Loading models...")

# #     model = joblib.load("models/aqi_model.pkl")
# #     scaler = joblib.load("models/scaler.pkl")

# #     print("Models loaded successfully.")

# #     df = pd.DataFrame([input_data])

# #     # IMPORTANT FIX — column names must match training
# #     df.columns = ["Temperature","Humidity","WindSpeed","Ozone","pm_ratio"]

# #     scaled = scaler.transform(df)

# #     prediction = model.predict(scaled)[0]

# #     return prediction

# def predict_air_quality(input_data):

#     print("Loading models...")

#     model = joblib.load("models/aqi_model.pkl")
#     scaler = joblib.load("models/scaler.pkl")

#     print("Models loaded successfully.")

#     df = pd.DataFrame([input_data])

#     # ensure column match with training
#     df = df[scaler.feature_names_in_]

#     scaled = scaler.transform(df)

#     prediction = model.predict(scaled)[0]

#     return prediction

# # =========================
# # SAMPLE TEST RUN
# # =========================
# if __name__ == "__main__":

#     sample_input = {
#         "PM2.5": 180,
#         "PM10": 240,
#         "NO2": 40,
#         "SO2": 12,
#         "CO": 1.2,
#         "O3": 30,
#         "temperature": 36,
#         "humidity": 85,
#         "wind_speed": 1.2
#     }

#     result = predict_air_quality(sample_input)

#     print("\n--- AIR REPORT ---")
#     for k, v in result.items():
#         print(f"{k}: {v}")





# import joblib
# import pandas as pd
# import torch

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# MODEL_PATH = "models/aqi_model.pkl"
# SCALER_PATH = "models/scaler.pkl"
# SLM_PATH = "models/slm_model"

# # -------------------------
# # LOAD MODELS
# # -------------------------
# def load_models():

#     print("Loading models...")

#     model = joblib.load(MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)

#     tokenizer = AutoTokenizer.from_pretrained(SLM_PATH)
#     slm_model = AutoModelForSeq2SeqLM.from_pretrained(SLM_PATH)

#     print("Models loaded successfully.")

#     return model, scaler, tokenizer, slm_model


# # -------------------------
# # AQI CATEGORY
# # -------------------------
# def aqi_category(aqi):

#     if aqi <= 50:
#         return "Good"
#     elif aqi <= 100:
#         return "Moderate"
#     elif aqi <= 150:
#         return "Unhealthy for Sensitive Groups"
#     elif aqi <= 200:
#         return "Unhealthy"
#     elif aqi <= 300:
#         return "Very Unhealthy"
#     else:
#         return "Hazardous"


# # -------------------------
# # GENERATE TEXT USING SLM
# # -------------------------
# def generate_report(tokenizer, model, values, aqi):

#     prompt = f"""
#     Air report:
#     Temperature {values['Temperature']}°C,
#     Humidity {values['Humidity']}%,
#     Wind speed {values['WindSpeed']} km/h,
#     Ozone {values['Ozone']},
#     PM ratio {values['pm_ratio']},
#     AQI {aqi}.
#     Explain health impact and precautions.
#     """

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

#     outputs = model.generate(
#         **inputs,
#         max_length=150
#     )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# # -------------------------
# # PREDICTION FUNCTION
# # -------------------------
# def predict_air_quality(input_data):

#     model, scaler, tokenizer, slm = load_models()

#     # Convert input to DataFrame
#     df = pd.DataFrame([input_data])

#     # Expected training columns
#     expected_cols = list(scaler.feature_names_in_)

#     # Add missing columns as 0
#     for col in expected_cols:
#         if col not in df.columns:
#             df[col] = 0

#     # Keep only required columns
#     df = df[expected_cols]

#     # Scale
#     scaled = scaler.transform(df)

#     # Predict
#     aqi = model.predict(scaled)[0]
#     category = aqi_category(aqi)

#     # Generate explanation
#     report = generate_report(tokenizer, slm, input_data, round(aqi,2))

#     return {
#         "AQI": round(aqi,2),
#         "Category": category,
#         "Report": report
#     }


# # -------------------------
# # SAMPLE RUN
# # -------------------------
# if __name__ == "__main__":

#     sample_input = {
#         "Temperature":30,
#         "Humidity":65,
#         "WindSpeed":4,
#         "Ozone":35,
#         "pm_ratio":0.7
#     }

#     result = predict_air_quality(sample_input)

#     print("\n====== AIR QUALITY RESULT ======")
#     print("AQI:", result["AQI"])
#     print("Category:", result["Category"])
#     print("\nAI Report:")
#     print(result["Report"])






# src/predict.py
# import pandas as pd
# import numpy as np
# import joblib

# # -------------------------
# # Load trained Linear Regression + Scaler
# # -------------------------
# ml_model = joblib.load("models/aqi_model.pkl")
# scaler = joblib.load("models/scaler.pkl")

# # -------------------------
# # Load dataset
# # -------------------------
# df = pd.read_csv("data/processed/clean_air_data.csv")

# # Compute stats for confidence
# X = df.drop("AQI", axis=1)
# y = df["AQI"]
# X_scaled = scaler.transform(X)
# y_pred_train = ml_model.predict(X_scaled)
# sigma = np.std(y - y_pred_train)

# # -------------------------
# # AQI categories and advice
# # -------------------------
# def aqi_category(aqi):
#     if aqi <= 50: return "Good", "Air quality is healthy. Safe for all activities."
#     elif aqi <= 100: return "Moderate", "Air quality is acceptable. Sensitive people should reduce prolonged outdoor exertion."
#     elif aqi <= 150: return "Unhealthy for Sensitive Groups", "Sensitive groups should avoid outdoor activities."
#     elif aqi <= 200: return "Unhealthy", "Everyone should reduce outdoor activity. Wear a mask if necessary."
#     elif aqi <= 300: return "Very Unhealthy", "Avoid going outside. Use air purifiers if possible."
#     else: return "Hazardous", "Stay indoors. Use masks and air purifiers."

# # -------------------------
# # Predict AQI + confidence for a row
# # -------------------------
# def predict_aqi(row):
#     X_row = row.values.reshape(1, -1)
#     X_row_scaled = scaler.transform(X_row)
#     pred_aqi = ml_model.predict(X_row_scaled)[0]
#     category, advice = aqi_category(pred_aqi)
#     confidence = max(0, 100 - (abs(pred_aqi - y.mean()) / (3 * sigma) * 100))
#     return pred_aqi, category, confidence, advice

# # -------------------------
# # Find the most relevant row from CSV for a question
# # -------------------------
# def select_relevant_row(question):
#     # Simple heuristic: pick row with highest pollutant mentioned in question
#     pollutants = {
#         "pm": "pm_ratio",
#         "ozone": "Ozone",
#         "o3": "Ozone",
#         "temperature": "Temperature",
#         "humidity": "Humidity",
#         "wind": "WindSpeed"
#     }

#     selected_cols = []
#     for key, col in pollutants.items():
#         if key in question:
#             selected_cols.append(col)

#     if selected_cols:
#         # Pick row with **highest value of relevant column**
#         col = selected_cols[0]
#         row = df.loc[df[col].idxmax()].drop("AQI")
#     else:
#         # If no pollutant mentioned, pick latest row
#         row = df.iloc[-1].drop("AQI")

#     return row

# # -------------------------
# # Interactive Q&A
# # -------------------------
# def ask_question():
#     print("Ask air-related questions (type 'exit' to quit):")
#     while True:
#         question = input("Q: ").lower()
#         if question == "exit":
#             break

#         relevant_row = select_relevant_row(question)
#         pred_aqi, category, confidence, advice = predict_aqi(relevant_row)

#         # Rule-based answers based on question keywords
#         if any(word in question for word in ["children", "outside", "safe"]):
#             print(f"A: Children should {'stay indoors' if pred_aqi > 100 else 'can play outside'} today.")
#         elif any(word in question for word in ["mask", "precaution", "protect"]):
#             print(f"A: {advice}")
#         elif any(word in question for word in ["aqi", "air quality", "pollution"]):
#             print(f"A: AQI is {pred_aqi:.2f} ({category}) with confidence {confidence:.2f}%")
#         else:
#             print("A: Sorry, I can only answer questions about AQI and air quality based on numeric data.")

#         print(f"(Predicted AQI: {pred_aqi:.2f}, Category: {category}, Confidence: {confidence:.2f}%)\n")

# # -------------------------
# # Main
# # -------------------------
# if __name__ == "__main__":
#     ask_question()



# src/predict.py
import pandas as pd
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Load trained Linear Regression + Scaler
# -------------------------
ml_model = joblib.load("models/aqi_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------
# Load SLM model
# -------------------------
slm_model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(slm_model_name)
slm_model = AutoModelForSeq2SeqLM.from_pretrained(slm_model_name)

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/processed/clean_air_data.csv")

# Compute stats for confidence
X = df.drop("AQI", axis=1)
y = df["AQI"]
X_scaled = scaler.transform(X)
y_pred_train = ml_model.predict(X_scaled)
sigma = np.std(y - y_pred_train)

# AQI categories
def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

# Predict AQI + confidence for a row
def predict_aqi(row):
    X_row = row.values.reshape(1, -1)
    X_row_scaled = scaler.transform(X_row)
    pred_aqi = ml_model.predict(X_row_scaled)[0]
    category = aqi_category(pred_aqi)
    confidence = max(0, 100 - (abs(pred_aqi - y.mean()) / (3 * sigma) * 100))
    return pred_aqi, category, confidence

# -------------------------
# Convert numeric row to textual prompt for SLM
# -------------------------
def row_to_prompt(row, question):
    row_info = ", ".join([f"{col}: {val}" for col, val in row.items()])
    prompt = f"Data: {row_info}. Question: {question}. Provide a clear answer based on this data."
    return prompt

# -------------------------
# Find the most relevant row for a question
# -------------------------
def select_row_for_question():
    # For now, pick the latest row in dataset
    return df.iloc[-1].drop("AQI")

# -------------------------
# Ask questions interactively
# -------------------------
def ask_question():
    print("Ask any air-related question (type 'exit' to quit):")
    while True:
        question = input("Q: ")
        if question.lower() == "exit":
            break

        # Pick a row from dataset
        row = select_row_for_question()
        pred_aqi, category, confidence = predict_aqi(row)

        # Prepare prompt for SLM
        prompt = row_to_prompt(row, question)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = slm_model.generate(**inputs, max_length=150)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"A: {answer}")
        print(f"(Predicted AQI: {pred_aqi:.2f}, Category: {category}, Confidence: {confidence:.2f}%)\n")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    ask_question()