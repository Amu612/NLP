import gradio as gr
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = [
    "Winter", "Summer", "Autumn", "Spring",
    "Land", "Water", "Fire", "Air",
    "Desert", "Forest", "Ocean", "Mountain"
]

def classify_environment(text):
    result = classifier(text, candidate_labels=labels)
    top_label = result["labels"][0]
    score = result["scores"][0]
    return top_label, float(score)

interface = gr.Interface(
    fn=classify_environment,
    inputs=gr.Textbox(lines=2, placeholder="Describe an environment..."),
    outputs=[
        gr.Textbox(label="Environmental Component"),
        gr.Textbox(label="Confidence Score")
    ],
    title="Text Classifier",
    description="Classify text into environmental components (e.g. seasons, biomes, elements) using Hugging Face models.",
    theme="default"
)

interface.launch()
