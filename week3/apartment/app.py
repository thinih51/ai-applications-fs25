import gradio as gr
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Laden des trainierten Modells
model_filename = "random_forest_regression.pkl"
with open(model_filename, 'rb') as f:
    random_forest_model = pickle.load(f)

# Laden der BFS-Daten
df_bfs_data = pd.read_csv("bfs_municipality_and_tax_data.csv", sep=",", encoding="utf-8")
df_bfs_data["tax_income"] = df_bfs_data["tax_income"].str.replace("'", "").astype(float)

# Liste der verfügbaren Orte
locations = {
    "Zürich": 261,
    "Kloten": 62,
    "Uster": 198,
    "Winterthur": 230,
    "Dietlikon": 69,
    # Weitere Orte hinzufügen
}

def predict_price(rooms, area, town):
    if town not in locations:
        return "Unknown Location"
    
    town_code = locations[town]
    town_data = df_bfs_data[df_bfs_data["bfs_id"] == town_code]
    if town_data.empty:
        return "No data available"
    
    features = np.array([
        [rooms, area, town_data.iloc[0]["pop"], town_data.iloc[0]["pop_dens"],
         town_data.iloc[0]["frg_pct"], town_data.iloc[0]["emp"], town_data.iloc[0]["tax_income"]]
    ])
    return random_forest_model.predict(features)[0]

def compare_apartments(rooms1, area1, town1, rooms2, area2, town2):
    price1 = predict_price(rooms1, area1, town1)
    price2 = predict_price(rooms2, area2, town2)
    return price1, price2, abs(price1 - price2)

demo = gr.Blocks()

with demo:
    gr.Markdown("## 🏘️ Apartment Price Comparison")
    gr.Markdown("Enter details for two apartments to see individual prices and the difference.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Apartment 1")
            rooms1 = gr.Number(label="Rooms")
            area1 = gr.Number(label="Area (m²)")
            town1 = gr.Dropdown(choices=list(locations.keys()), label="Town")
        with gr.Column():
            gr.Markdown("### Apartment 2")
            rooms2 = gr.Number(label="Rooms")
            area2 = gr.Number(label="Area (m²)")
            town2 = gr.Dropdown(choices=list(locations.keys()), label="Town")

    compare_btn = gr.Button("Compare Prices")

    with gr.Row():
        price1 = gr.Number(label="Price Apartment 1")
        price2 = gr.Number(label="Price Apartment 2")
        difference = gr.Number(label="Price Difference")

    compare_btn.click(
        fn=compare_apartments,
        inputs=[rooms1, area1, town1, rooms2, area2, town2],
        outputs=[price1, price2, difference]
    )

demo.launch()