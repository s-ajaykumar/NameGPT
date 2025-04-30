import gradio as gr



with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Name-GPT
        This is a simple name generator using a transformer model.
        """
    )
    with gr.Row():
        with gr.Column():
            generate_button = gr.Button("Generate")
        with gr.Column():
            output_text = gr.Textbox(label="Generated Text", placeholder="Generated text will appear here...")

    generate_button.click(inputs = 2, outputs=output_text)
    
    demo.launch()