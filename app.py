# Gradio app for Laptop Price Prediction (Bangladesh Market)
import gradio as gr
import pandas as pd
import pickle
import numpy as np

# =====================
# 1. Load the Model
# =====================
with open("laptop_gb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")

# =====================
# 3. The Prediction Function
# =====================
def predict_price(brand, quality_score, ram, ram_type, rom, rom_type,
                  display_size, resolution_width, resolution_height, 
                  os, warranty):
    
    # Pack inputs into a DataFrame
    input_df = pd.DataFrame([[
        brand, quality_score, ram, ram_type, rom, rom_type,
        display_size, resolution_width, resolution_height, os, warranty
    ]], 
    columns=[
        'brand', 'spec_rating', 'Ram', 'Ram_type', 'ROM', 'ROM_type',
        'display_size', 'resolution_width', 'resolution_height', 'OS', 'warranty'
    ])
    
    # Predict (returns price in INR)
    prediction_inr = model.predict(input_df)[0]
    
    # Convert to BDT
    prediction_bdt = prediction_inr 
    
    # Determine price category
    if prediction_bdt < 42000:
        category = "Budget Range"
        emoji = "💚"
    elif prediction_bdt < 107000:
        category = "Mid Range"
        emoji = "💙"
    else:
        category = "Premium Range"
        emoji = "💎"
    
    # Format output
    result = f"""
{emoji} Predicted Laptop Price

Price: ৳{prediction_bdt:,.0f}
Category: {category}
    """
    
    return result

# =====================
# 4. The App Interface
# =====================
inputs = [
    gr.Dropdown(
        ["HP", "Lenovo", "Asus", "Dell", "Acer", "MSI", "Apple", "Samsung", 
         "Infinix", "LG", "Xiaomi", "Gigabyte", "Fujitsu", "Ultimus", 
         "Zebronics", "Realme", "Wings", "Chuwi", "Tecno", "Honor", "AXL", 
         "Huawei", "Microsoft", "iBall", "Walker", "Avita", "Vaio", "Razer"],
        label="Laptop Brand",
        value="HP"
    ),
    gr.Slider(
        60, 90, step=1, 
        label="Overall Performance Score",
        value=70,
        info="Budget: 60-65 | Standard: 66-70 | Good: 71-80 | Premium: 81-90"
    ),
    gr.Dropdown(
        [4, 8, 12, 16, 32, 64], 
        label="RAM Size (GB)", 
        value=8
    ),
    gr.Dropdown(
        ["DDR4", "DDR5", "LPDDR4", "LPDDR5", "LPDDR4X", "LPDDR4x", 
         "LPDDR5X", "LPDDR5x", "DDR3", "Unified", "DDR4-", "DDR"], 
        label="RAM Type", 
        value="DDR4"
    ),
    gr.Dropdown(
        [32, 64, 128, 256, 512, 1024, 2048], 
        label="Storage Capacity (GB)", 
        value=512
    ),
    gr.Radio(
        ["SSD", "HDD"], 
        label="Storage Type", 
        value="SSD"
    ),
    gr.Slider(
        11.6, 18, step=0.1, 
        label="Screen Size (inches)", 
        value=15.6
    ),
    gr.Number(
        label="Screen Resolution Width", 
        value=1920
    ),
    gr.Number(
        label="Screen Resolution Height", 
        value=1080
    ),
    gr.Dropdown(
        ["Windows 11 OS", "Windows 10 OS", "DOS OS", "Windows 11  OS", 
         "Mac OS", "Windows 10  OS", "Chrome OS", "Windows OS", 
         "Ubuntu OS", "DOS 3.0 OS", "Android 11 OS"],
        label="Operating System", 
        value="Windows 11 OS"
    ),
    gr.Slider(
        0, 3, step=1, 
        label="Warranty Period (years)", 
        value=1
    )
]

# Create the Gradio Interface
app = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=gr.Textbox(label="Price Prediction", lines=5),
    title="💻 Laptop Price Predictor - Bangladesh",
    description="Get laptop price predictions for the Bangladesh market based on specifications",
    theme="default",
    examples=[
        ["HP", 73, 8, "DDR4", 512, "SSD", 15.6, 1920, 1080, "Windows 11 OS", 1],
        ["Apple", 85, 16, "LPDDR5", 512, "SSD", 13.3, 2560, 1600, "Mac OS", 1],
        ["Lenovo", 66, 16, "LPDDR5", 512, "SSD", 14.0, 2240, 1400, "Windows 11 OS", 1],
        ["Acer", 69, 8, "DDR4", 512, "SSD", 14.0, 1920, 1080, "Windows 11 OS", 1],
        ["MSI", 80, 16, "DDR5", 1024, "SSD", 15.6, 1920, 1080, "Windows 11 OS", 2]
    ]
)

# =====================
# 5. Launch the App
# =====================
if __name__ == "__main__":
    app.launch()