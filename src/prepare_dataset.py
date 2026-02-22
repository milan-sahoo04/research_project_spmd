import pandas as pd

# Load numeric dataset
df = pd.read_csv("data/processed/clean_air_data.csv")

# Function to get AQI category
def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

prompts = []
for _, row in df.iterrows():
    aqi = row["AQI"]
    category = aqi_category(aqi)
    
    # Multiple questions per row
    prompts.append({
        "prompt": f"Temperature: {row['Temperature']}, Humidity: {row['Humidity']}, WindSpeed: {row['WindSpeed']}, Ozone: {row['Ozone']}, pm_ratio: {row['pm_ratio']}. Question: Should children go outside today?",
        "answer": f"AQI is {aqi} ({category}). Children should {'stay indoors' if aqi>100 else 'can play outside'}."
    })
    prompts.append({
        "prompt": f"Temperature: {row['Temperature']}, Humidity: {row['Humidity']}, WindSpeed: {row['WindSpeed']}, Ozone: {row['Ozone']}, pm_ratio: {row['pm_ratio']}. Question: What precautions should I take?",
        "answer": f"AQI is {aqi} ({category}). Wear a mask and avoid prolonged outdoor activity if AQI > 100."
    })
    prompts.append({
        "prompt": f"Temperature: {row['Temperature']}, Humidity: {row['Humidity']}, WindSpeed: {row['WindSpeed']}, Ozone: {row['Ozone']}, pm_ratio: {row['pm_ratio']}. Question: What is the air quality today?",
        "answer": f"AQI is {aqi} ({category})."
    })

# Save dataset for SLM training
pd.DataFrame(prompts).to_csv("data/processed/dataset_for_slm.csv", index=False)
print("SLM dataset created at data/processed/dataset_for_slm.csv")