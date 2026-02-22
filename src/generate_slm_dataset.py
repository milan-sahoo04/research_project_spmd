import random
import pandas as pd
import os

rows = []

for _ in range(3000):

    pm25 = random.randint(10,300)
    humidity = random.randint(20,95)
    wind = round(random.uniform(0.5,8),1)
    aqi = int(pm25*1.3 + humidity*0.5 - wind*5)

    if pm25 > 150:
        reason = "fine particles are extremely high"
    elif pm25 > 80:
        reason = "pollution levels are elevated"
    else:
        reason = "pollution is low"

    if wind < 2:
        wind_reason = "wind is low and pollutants are trapped"
    else:
        wind_reason = "wind helps disperse pollutants"

    text = f"Air quality is affected because {reason} and {wind_reason}."

    rows.append({
        "input": f"PM2.5={pm25} humidity={humidity} wind={wind} AQI={aqi}",
        "output": text
    })

df = pd.DataFrame(rows)

os.makedirs("slm", exist_ok=True)
df.to_csv("slm/dataset.csv", index=False)

print("Dataset generated successfully.")