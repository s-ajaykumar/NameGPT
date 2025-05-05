# app.py
import gradio as gr
from model import load_model_and_generate_name

# Path to the model checkpoint
MODEL_PATH = "NameGPT.pth"  # Update with the correct path

# Custom CSS for button styling
css = """
#male_button {
    background-color: #2196F3 !important;  /* Blue */
    color: white !important;
    border-radius: 8px !important;
}
#female_button {
    background-color: #F06292 !important;  /* Pink */
    color: white !important;
    border-radius: 8px !important;
}
.button_container {
    display: flex;
    gap: 10px;
    justify-content: center;
}
#title {
    font-size: 36px !important;
    font-weight: bold !important;
    text-align: center !important;
}
#instruction {
    text-align: center !important;
}
"""

def generate_male_name():
    try:
        name = load_model_and_generate_name(MODEL_PATH, "male")
        return name
    except Exception as e:
        return f"Error: {str(e)}"
    
def generate_female_name():
    try:
        name = load_model_and_generate_name(MODEL_PATH, "female")
        return name
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(css = css) as demo:
    gr.Markdown("# NameGPT", elem_id = "title")
    gr.Markdown("Click the button to generate a beautiful name for your baby!", elem_id = "instruction")
    with gr.Row(elem_classes="button_container"):
        male_button = gr.Button("♂", elem_id="male_button")
        female_button = gr.Button("♀", elem_id="female_button")
    
    output = gr.Textbox(label="Generated Name")
    
    male_button.click(
        fn=generate_male_name,
        outputs=output
    )
    female_button.click(
        fn=generate_female_name,
        outputs=output
    )

# Launch the app
demo.launch()