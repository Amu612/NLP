import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import networkx as nx
import matplotlib.pyplot as plt

# ==================== Load Models ====================
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Stable Diffusion
image_gen = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    revision="fp16" if torch.cuda.is_available() else None
).to("cuda" if torch.cuda.is_available() else "cpu")

# NER and Fill-mask
ner_pipe = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# ==================== Custom Utilities ====================

# Environmental keywords for filter
env_keywords = [
    "pollution", "waste", "climate", "plastic", "ocean", "recycle",
    "recycling", "deforestation", "conservation", "emissions", "toxins",
    "greenhouse", "eco", "water", "air", "global", "carbon", "fuel"
]

# Color mapping for HighlightedText
highlight_colors = {
    "ORG": "lightblue",
    "LOC": "lightgreen",
    "POLLUTION": "lightcoral",
    "THRESHOLD": "orange"
}

# ==================== Functions ====================

def classify_sentence(text):
    labels = ["detect water leakage", "waste management", "climate change", "recycling", "air pollution"]
    result = classifier(text, labels)
    return dict(zip(result['labels'], map(lambda x: round(x, 3), result['scores'])))

def generate_image(prompt):
    result = image_gen(prompt)
    return result.images[0]

def custom_ner_highlight(text):
    ents = ner_pipe(text)
    spans = []

    for ent in ents:
        word = ent['word']
        label = ent['entity_group']

        # Custom environmental classification
        if 'pollution' in word.lower() or 'emission' in word.lower():
            label = 'POLLUTION'
        elif 'limit' in word.lower():
            label = 'THRESHOLD'
        elif word.lower() in ['who', 'unesco', 'greenpeace']:
            label = 'ORG'
        elif word.lower() in ['india', 'delhi', 'arctic', 'ganga', 'kanpur', 'varanasi']:
            label = 'LOC'

        spans.append((word, label))
    return spans

def ner_graph(text):
    entities = ner_pipe(text)
    G = nx.Graph()

    for entity in entities:
        G.add_node(entity['word'], label=entity['entity_group'])

    for i in range(len(entities) - 1):
        G.add_edge(entities[i]['word'], entities[i + 1]['word'])

    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray',
            node_size=1200, font_size=10, ax=ax)
    plt.tight_layout()
    return fig

def fill_sentence(masked):
    if '[MASK]' not in masked:
        return "❗ Please include [MASK] in your sentence."

    predictions = fill_mask(masked)

    filtered = []
    for pred in predictions:
        token = pred['token_str'].lower()
        if any(env_key in token for env_key in env_keywords):
            filtered.append(f"- {pred['sequence']}")

    if not filtered:
        return "❌ No environment-related suggestions found."
    return "\n".join(filtered)


# ==================== Gradio UI ====================

with gr.Blocks() as demo:
    gr.Markdown("## 🌿 Environmental AI Tool — All-in-One")
    gr.Markdown("This app uses HuggingFace models to analyze environment-related text.")

    with gr.Tab("1️⃣ Sentence Classification"):
        input_text = gr.Textbox(label="Enter Environmental Sentence")
        classify_btn = gr.Button("Classify")
        classify_out = gr.Label(label="Classification Results")
        classify_btn.click(fn=classify_sentence, inputs=input_text, outputs=classify_out)

    with gr.Tab("2️⃣ Image Generation (Stable Diffusion)"):
        img_prompt = gr.Textbox(label="Enter Prompt (e.g., 'waste near river')")
        img_btn = gr.Button("Generate Image")
        image_out = gr.Image(label="Generated Image")
        img_btn.click(fn=generate_image, inputs=img_prompt, outputs=image_out)

    with gr.Tab("3️⃣ NER & Entity Graph"):
        ner_input = gr.Textbox(label="Enter Sentence for NER")
        ner_btn1 = gr.Button("Highlight Entities")
        ner_btn2 = gr.Button("Generate Graph")
        ner_out_highlight = gr.HighlightedText(label="Highlighted Entities", color_map=highlight_colors)
        ner_out_graph = gr.Plot(label="Entity Relationship Graph")

        ner_btn1.click(fn=custom_ner_highlight, inputs=ner_input, outputs=ner_out_highlight)
        ner_btn2.click(fn=ner_graph, inputs=ner_input, outputs=ner_out_graph)

    with gr.Tab("4️⃣ Fill in the Blank"):
        mask_input = gr.Textbox(label="Sentence (use [MASK])", placeholder="e.g., Water [MASK] is harmful.")
        mask_btn = gr.Button("Suggest Environmental Terms")
        mask_output = gr.Textbox(label="Filtered Predictions")
        mask_btn.click(fn=fill_sentence, inputs=mask_input, outputs=mask_output)

demo.launch()
