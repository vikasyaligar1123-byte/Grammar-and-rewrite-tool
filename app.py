import gradio as gr
from transformers import pipeline

# ----------------------------------
# Load models ONCE (CPU-safe)
# ----------------------------------
grammar_corrector = pipeline(
    "text2text-generation",
    model="prithivida/grammar_error_correcter_v1"
)

paraphraser = pipeline(
    "text2text-generation",
    model="Vamsi/T5_Paraphrase_Paws"
)

# ----------------------------------
# Core logic
# ----------------------------------
def grammar_and_rewrite(text, mode):
    if not text.strip():
        return "", ""

    # Step 1: Grammar correction
    corrected = grammar_corrector(
        text,
        max_length=256,
        clean_up_tokenization_spaces=True
    )[0]["generated_text"]

    # Step 2: Rewrite (style-based)
    if mode == "Only Grammar Correction":
        rewritten = corrected

    else:
        prompt_map = {
            "Formal": f"paraphrase: {corrected}",
            "Simple": f"paraphrase: {corrected}",
            "Professional": f"paraphrase: {corrected}"
        }

        rewritten = paraphraser(
            prompt_map[mode],
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
            temperature=1.2,
            clean_up_tokenization_spaces=True
        )[0]["generated_text"]

    return corrected, rewritten

# ----------------------------------
# Gradio UI
# ----------------------------------
interface = gr.Interface(
    fn=grammar_and_rewrite,
    inputs=[
        gr.Textbox(label="Enter Text", lines=8, placeholder="Paste your text here..."),
        gr.Radio(
            choices=["Only Grammar Correction", "Formal", "Simple", "Professional"],
            value="Only Grammar Correction",
            label="Rewrite Mode"
        )
    ],
    outputs=[
        gr.Textbox(label="Grammar Corrected Text", lines=6),
        gr.Textbox(label="Rewritten Text", lines=6)
    ],
    title="Grammar & Rewrite Tool",
    description="Corrects grammatical errors and rewrites text in different styles using Transformer models"
)

# ----------------------------------
# Entry point (REQUIRED)
# ----------------------------------
interface.launch()
