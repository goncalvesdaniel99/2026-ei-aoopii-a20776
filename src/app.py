"""
Virtual Try-On — interface Gradio.

A UI é deliberadamente fina: toda a lógica vive em tryon.py. Aqui só ligamos o dataset
ao motor e mostramos o resultado.

Como correr:
    python src/app.py
Variáveis de ambiente:
    TRYON_BACKEND = auto | catvton | inpaint   (default: auto)
    GRADIO_SHARE  = 1                            (gera link público — usar no Colab)
"""

import os

import gradio as gr

try:
    from tryon import DatasetLoader, VirtualTryOn
except ImportError:  # quando importado como pacote (ex.: a partir do notebook)
    from src.tryon import DatasetLoader, VirtualTryOn


def build_app():
    data = DatasetLoader()
    engine = VirtualTryOn(backend=os.environ.get("TRYON_BACKEND", "auto"))

    def run(person_id, cloth_id):
        if not person_id or not cloth_id:
            return None
        person = data.load_person(person_id)
        cloth = data.load_cloth(cloth_id)
        mask = data.load_mask(person_id)
        return engine.generate(person, cloth, mask)

    with gr.Blocks(title="Virtual Try-On") as app:
        gr.Markdown(
            f"# 👔 Virtual Try-On\n"
            f"Backend ativo: **{engine.backend_name}** "
            f"({'modelo fiel — GPU' if engine.backend_name == 'catvton' else 'baseline local — qualidade limitada'})"
        )
        with gr.Row():
            with gr.Column():
                person = gr.Dropdown(data.get_available_models(), label="1. Pessoa")
                cloth = gr.Dropdown(data.get_available_garments(), label="2. Peça de roupa")
                btn = gr.Button("⚡ Vestir", variant="primary")
            with gr.Column():
                out = gr.Image(label="Resultado")
        btn.click(run, inputs=[person, cloth], outputs=out)

    return app


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    build_app().launch(share=share)
