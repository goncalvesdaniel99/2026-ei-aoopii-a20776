import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# ==========================================
# 1. Gestão do Dataset (VITON-HD)
# ==========================================
class DatasetLoader:
    def __init__(self, base_dir="../data/test"):
        self.image_dir = os.path.join(base_dir, "image")
        self.agnostic_dir = os.path.join(base_dir, "agnostic")
        self.cloth_dir = os.path.join(base_dir, "cloth")
        self.mask_dir = os.path.join(base_dir, "cloth-mask")

    def get_available_models(self):
        if not os.path.exists(self.image_dir): return []
        return sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])

    def get_available_garments(self):
        if not os.path.exists(self.cloth_dir): return []
        return sorted([f for f in os.listdir(self.cloth_dir) if f.endswith(('.jpg', '.png'))])

    def load_assets(self, id_modelo, id_roupa):
        agnostic_path = os.path.join(self.agnostic_dir, id_modelo)
        cloth_path = os.path.join(self.cloth_dir, id_roupa)
        mask_path = os.path.join(self.mask_dir, id_roupa)

        if not all(os.path.exists(p) for p in [agnostic_path, cloth_path, mask_path]):
            raise FileNotFoundError("Ficheiros em falta. Verifica a estrutura do dataset.")

        img_agnostic = Image.open(agnostic_path).convert("RGB")
        img_cloth = Image.open(cloth_path).convert("RGB")
        
        cv_agnostic = cv2.cvtColor(np.array(img_agnostic), cv2.COLOR_RGB2BGR)
        lower_gray = np.array([100, 100, 100], dtype=np.uint8)
        upper_gray = np.array([150, 150, 150], dtype=np.uint8)
        mask_cinzenta = cv2.inRange(cv_agnostic, lower_gray, upper_gray)
        
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_cinzenta, kernel, iterations=1)
        
        # SALVAGUARDA EXTRA: Se a máscara estiver vazia, forçar uma caixa no centro
        if cv2.countNonZero(mask_dilated) == 0:
            print("⚠️ Máscara não detetada pelo limiar cinzento. A aplicar máscara de segurança.")
            h, w = mask_dilated.shape
            mask_dilated[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = 255

        img_mask = Image.fromarray(mask_dilated).convert("L")

        return img_agnostic, img_cloth, img_mask

# ==========================================
# 2. Motor de Deep Learning (Diffusion)
# ==========================================
class VirtualTryOnDiffusion:
    def __init__(self):
        print("⏳ A carregar o modelo de Deep Learning...")
        
        # Detetar hardware
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            # OBRIGATÓRIO PARA MAC: float32. O float16 causa o "ecrã castanho/preto" crónico neste modelo.
            self.dtype = torch.float32 
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            
        print(f"🚀 Hardware detetado: {self.device.upper()} | Precisão Base: {self.dtype}")
        
        model_id = "runwayml/stable-diffusion-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, 
            torch_dtype=self.dtype,
            safety_checker=None,              
            requires_safety_checker=False     
        ).to(self.device)

        # OTIMIZAÇÃO EXTREMA PARA MAC (Evita os 1000 segundos de congelamento no float32)
        self.pipe.enable_attention_slicing("max")
        print("✅ Motor de Inteligência Artificial Generativa pronto!")

    def generate_tryon(self, img_agnostic, img_cloth, img_mask):
        img_agnostic_resized = img_agnostic.resize((512, 512))
        img_mask_resized = img_mask.resize((512, 512))
        
        prompt = "A highly detailed, photorealistic fashion model wearing the target garment, perfect fit, natural fabric folds, high resolution, 8k"
        negative_prompt = "mutated, deformed, artifacts, naked, poorly drawn, extra limbs, bad proportions"

        print("🎨 A gerar a fotografia da peça de roupa... (Cerca de 60 a 90 segundos)")
        
        # Limpar lixo de memória do Mac antes de começar
        if self.device == "mps":
            torch.mps.empty_cache()

        # O pipeline padrão agora funciona perfeitamente porque a matemática é 32-bits pura
        resultado = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_agnostic_resized,
            mask_image=img_mask_resized,
            num_inference_steps=15, # REDUZIDO: de 25 para 15 para compensar a velocidade sem perder qualidade
            guidance_scale=7.5
        ).images[0]

        return resultado.resize(img_agnostic.size)

# ==========================================
# 3. Interface de Utilizador (Gradio)
# ==========================================
class AppInterface:
    def __init__(self):
        self.dataset = DatasetLoader()
        self.ai_engine = VirtualTryOnDiffusion()

    def process_images(self, id_modelo, id_roupa):
        if not id_modelo or not id_roupa:
            return None
        try:
            agnostic, cloth, mask = self.dataset.load_assets(id_modelo, id_roupa)
            resultado_final = self.ai_engine.generate_tryon(agnostic, cloth, mask)
            return resultado_final
        except Exception as e:
            print(f"❌ Erro: {e}")
            return None

    def launch(self):
        with gr.Blocks() as app:
            gr.Markdown("# 👔 Virtual Try-On (Deep Learning / Diffusion)")
            gr.Markdown("Projeto para a unidade curricular **Aprendizagem Organizacional - Opção II (Track A)**. Utiliza IA Generativa (*Inpainting*) para processar o dataset VITON-HD.")
            
            with gr.Row():
                with gr.Column():
                    dropdown_modelo = gr.Dropdown(choices=self.dataset.get_available_models(), label="1. Selecionar Modelo (Agnostic)")
                    dropdown_roupa = gr.Dropdown(choices=self.dataset.get_available_garments(), label="2. Selecionar Peça de Roupa")
                    btn_executar = gr.Button("⚡ Gerar Encaixe com IA", variant="primary")
                
                with gr.Column():
                    resultado_ui = gr.Image(label="Resultado Final (AI Generated)")

            btn_executar.click(
                fn=self.process_images, 
                inputs=[dropdown_modelo, dropdown_roupa], 
                outputs=resultado_ui
            )
        
        app.launch(share=False, theme=gr.themes.Monochrome())

if __name__ == "__main__":
    TryOnApp = AppInterface()
    TryOnApp.launch()