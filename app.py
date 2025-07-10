import gradio as gr
from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline
import spacy
import matplotlib.pyplot as plt
from collections import Counter
import random

# Load Hugging Face Pipelines
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

# Environment categories
labels = [
    "Winter", "Summer", "Autumn", "Spring",
    "Land", "Water", "Fire", "Air",
    "Desert", "Forest", "Ocean", "Mountain"
]

# Environmental keywords for spaCy + NER graph
environment_keywords = [
    "forest", "desert", "mountain", "river", "ocean", "sea", "beach",
    "climate", "wildlife", "lake", "earthquake", "volcano", "storm",
    "snow", "rain", "sun", "summer", "winter", "spring", "autumn",
    "land", "air", "fire", "water", "hurricane", "drought", "flood"
]

# 1. Classification
def classify_environment(text):
    result = classifier(text, candidate_labels=labels)
    top_label = result["labels"][0]
    score = round(result["scores"][0], 4)
    return top_label, score

# 2. Image Generation
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# 3. Named Entity Recognition + Graph
def extract_and_plot_entities(text):
    doc = nlp(text)
    all_entities = [ent.text.lower() for ent in doc.ents]
    filtered_entities = [ent for ent in all_entities if any(word in ent for word in environment_keywords)]

    if not filtered_entities:
        return "No environment-related entities found.", None

    counts = Counter(filtered_entities)
    labels_, values = zip(*counts.items())
    colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in labels_]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels_, values, color=colors)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(bar.get_height()),
                ha='center', va='bottom', fontsize=9)

    ax.set_title("NER Entity Distribution", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_xlabel("Entity Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return "Entities found:", fig

# 4. Fill-in-the-blank
def fill_blank(text):
    if "[MASK]" not in text:
        return ["Please include [MASK] in your sentence to get predictions."]
    outputs = fill_mask(text)
    return [out["sequence"] for out in outputs]

# Combine all UI components
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue")) as demo:
    gr.Markdown("""
    # 🌍 Environment-Aware AI Processor
    **Powered by Hugging Face Models**  
    Type text related to environment and process it through classification, NER, image generation, and more.
    """)

    with gr.Row():
        text_input = gr.Textbox(label="Enter sentence with a blank (use [MASK] if needed)",
                                value="My name is Will Smith. I like the Himalaya mountains. I study at GLS University. Next holidays, I will visit the [MASK] for trekking")
        process_btn = gr.Button("🚀 Process")

    with gr.Accordion("Uses", open=False):
        gr.Markdown("""
        - `facebook/bart-large-mnli` for classification  
        - `runwayml/stable-diffusion-v1-5` for image generation  
        - `dslim/bert-base-NER` for NER  
        - `bert-base-uncased` for fill-mask
        """)

    output1 = gr.Textbox(label="1. Category Prediction")
    output2 = gr.Image(label="2. Generated Image")
    output3_text = gr.Textbox(label="3. NER Summary")
    output3_fig = gr.Plot(label="3. NER Graph")
    output4 = gr.Textbox(label="4. Fill-in-the-Blank Output", lines=5)

    def full_process(text):
        label, score = classify_environment(text)
        img = generate_image(label + " environment")
        ner_text, ner_graph = extract_and_plot_entities(text)
        fill_results = fill_blank(text)
        return f"{label} ({score})", img, ner_text, ner_graph, "\n".join(fill_results)

    process_btn.click(fn=full_process, inputs=text_input,
                      outputs=[output1, output2, output3_text, output3_fig, output4])

if __name__ == "__main__":
    demo.launch()
