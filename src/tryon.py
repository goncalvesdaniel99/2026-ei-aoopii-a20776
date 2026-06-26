"""
Virtual Try-On — motor de inferência.

Este módulo concentra a lógica reutilizável do projeto (independente da interface).
Expõe a fachada `VirtualTryOn`, que escolhe um backend por trás da mesma API:

  - CatVTONBackend : modelo treinado especificamente para try-on. Resultados FIÉIS
                     (preserva a peça e adapta-a ao corpo). Requer GPU CUDA -> Colab.
  - InpaintBackend : Stable Diffusion inpainting genérico + IP-Adapter. Baseline que
                     corre em CPU/MPS para desenvolvimento. Qualidade limitada: o modelo
                     não foi treinado para try-on, logo "imagina" a peça em vez de a vestir.

Tanto a app (app.py) como o notebook do Colab usam `VirtualTryOn` sem saber qual o
backend ativo. A escolha é automática: GPU -> CatVTON, caso contrário -> baseline.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageFilter


# =====================================================================
# 1. Dataset (VITON-HD)
# =====================================================================
class DatasetLoader:
    """Carrega pessoa, peça e máscara a partir do dataset VITON-HD em data/test."""

    def __init__(self, base_dir=None):
        if base_dir is None:
            # Caminho relativo ao ficheiro -> funciona seja qual for o cwd.
            here = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(here, "..", "data", "test")
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, "image")        # pessoa (foto real)
        self.agnostic_dir = os.path.join(base_dir, "agnostic")  # pessoa com a roupa ocultada
        self.cloth_dir = os.path.join(base_dir, "cloth")        # peça de roupa isolada

    @staticmethod
    def _list(directory):
        if not os.path.isdir(directory):
            return []
        return sorted(f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".png")))

    def get_available_models(self):
        return self._list(self.image_dir)

    def get_available_garments(self):
        return self._list(self.cloth_dir)

    def load_person(self, person_id):
        return Image.open(os.path.join(self.image_dir, person_id)).convert("RGB")

    def load_cloth(self, cloth_id):
        return Image.open(os.path.join(self.cloth_dir, cloth_id)).convert("RGB")

    def load_mask(self, person_id, dilate=15, blur=9):
        """Máscara da zona a vestir.

        A imagem `agnostic` do VITON-HD tem a roupa original substituída por cinzento
        (~128,128,128). Essa região cinzenta é exatamente a zona onde queremos colocar a
        nova peça — ou seja, a máscara de inpainting. Extraímo-la por limiar de cor.
        """
        agnostic = Image.open(os.path.join(self.agnostic_dir, person_id)).convert("RGB")
        bgr = cv2.cvtColor(np.array(agnostic), cv2.COLOR_RGB2BGR)

        # O cinzento de ocultação é rigorosamente ~128. Limiar apertado primeiro...
        mask = cv2.inRange(bgr, (125, 125, 125), (131, 131, 131))
        # ...com fallback mais largo para variações de compressão JPEG.
        if mask.sum() == 0:
            mask = cv2.inRange(bgr, (110, 110, 110), (140, 140, 140))

        if dilate:
            kernel = np.ones((dilate, dilate), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        out = Image.fromarray(mask).convert("L")
        if blur:
            # Bordas suaves -> transição mais natural no inpainting.
            out = out.filter(ImageFilter.GaussianBlur(blur))
        return out


# =====================================================================
# 2. Backends de inferência
# =====================================================================
class TryOnBackend:
    """Interface comum. Todos os backends recebem PIL.Image e devolvem PIL.Image."""

    name = "base"

    def generate(self, person, cloth, mask, **kwargs):
        raise NotImplementedError


class CatVTONBackend(TryOnBackend):
    """Try-on fiel com o modelo CatVTON (treinado para a tarefa). Requer GPU CUDA.

    Depende do repositório de código do CatVTON estar no sys.path (o notebook do Colab
    clona-o para /content/CatVTON). Os pesos são descarregados do Hugging Face Hub.

    Nota: usamos a máscara que extraímos do dataset, por isso NÃO precisamos do AutoMasker
    do CatVTON (que exige detectron2/densepose, difíceis de instalar). Isto simplifica
    bastante a instalação no Colab.
    """

    name = "catvton"

    def __init__(
        self,
        catvton_repo="zhengchong/CatVTON",
        base_model="booksforcharlie/stable-diffusion-inpainting",
        width=768,
        height=1024,
        device="cuda",
    ):
        import torch
        # Importado do repo clonado do CatVTON (model/pipeline.py).
        from model.pipeline import CatVTONPipeline

        # A própria pipeline descarrega os pesos do HF Hub (snapshot_download interno)
        # quando attn_ckpt não é um caminho local. skip_safety_check evita carregar o
        # safety checker (poupa memória e dependências do modelo base).
        self.pipeline = CatVTONPipeline(
            base_ckpt=base_model,
            attn_ckpt=catvton_repo,
            attn_ckpt_version="mix",
            weight_dtype=torch.float16,
            device=device,
            skip_safety_check=True,
        )
        self.width, self.height = width, height
        self.device = device
        self._torch = torch

    def generate(self, person, cloth, mask, steps=50, guidance=2.5, seed=42):
        torch = self._torch
        size = (self.width, self.height)
        person = person.resize(size)
        mask = mask.resize(size)
        cloth = cloth.resize(size)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipeline(
            image=person,
            condition_image=cloth,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )[0]
        return result


class InpaintBackend(TryOnBackend):
    """Baseline local: Stable Diffusion inpainting genérico + IP-Adapter.

    Corre em CPU/MPS (Mac/PC sem GPU). Serve para desenvolvimento e para documentar a
    limitação da abordagem: como o modelo não foi treinado para try-on, a peça não é
    reproduzida fielmente — o IP-Adapter apenas dá uma "ideia" global da roupa.

    Otimizações de memória (cruciais com pouca RAM, ex. 8 GB):
      - enable_attention_slicing("max") e enable_vae_tiling() baixam o pico de memória;
      - mantemos float32 no MPS (float16 produz imagens castanhas/pretas neste modelo);
      - NÃO mexemos em PYTORCH_MPS_HIGH_WATERMARK_RATIO (pôr a 0.0 desliga o teto de
        segurança e provoca swap para disco -> as tais gerações de ~20 min).
    """

    name = "inpaint"

    def __init__(self, model_id="runwayml/stable-diffusion-inpainting",
                 device=None, use_ip_adapter=None):
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        # Permite desligar o IP-Adapter por ambiente (TRYON_IP_ADAPTER=0) para caber
        # em GPUs/memória pequenas (ex.: Mac de 8 GB): sem IP-Adapter ativa-se o
        # attention slicing e o consumo desce o suficiente para não rebentar.
        if use_ip_adapter is None:
            use_ip_adapter = os.environ.get("TRYON_IP_ADAPTER", "1") == "1"

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.use_ip_adapter = use_ip_adapter

        # fp16 só compensa em CUDA; no MPS/CPU usamos fp32 por estabilidade.
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if use_ip_adapter:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
            )
            self.pipe.set_ip_adapter_scale(0.7)
        else:
            # O attention slicing poupa memória, mas é INCOMPATÍVEL com o IP-Adapter:
            # substitui os attention processors do IP-Adapter e rebenta com
            # "'tuple' object has no attribute 'shape'". Só o ativamos sem IP-Adapter.
            self.pipe.enable_attention_slicing("max")

        try:
            self.pipe.enable_vae_tiling()
        except Exception:
            pass
        self.pipe = self.pipe.to(device)
        self._torch = torch

    def generate(self, person, cloth, mask, steps=20, guidance=7.5, seed=42):
        torch = self._torch
        person = person.resize((512, 512))
        mask = mask.resize((512, 512))
        if self.device == "mps":
            torch.mps.empty_cache()

        kwargs = dict(
            prompt="a person wearing the garment, photorealistic, natural fabric folds, high detail",
            negative_prompt="deformed, blurry, bad anatomy, artifacts, extra limbs",
            image=person,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        if self.use_ip_adapter:
            kwargs["ip_adapter_image"] = cloth
        return self.pipe(**kwargs).images[0]


# =====================================================================
# 3. Fachada
# =====================================================================
class VirtualTryOn:
    """Escolhe o backend e expõe um único método generate(person, cloth, mask)."""

    def __init__(self, backend="auto", **backend_kwargs):
        if backend == "auto":
            backend = self._auto_select()
        self.backend_name = backend

        if backend == "catvton":
            self.backend = CatVTONBackend(**backend_kwargs)
        elif backend == "inpaint":
            self.backend = InpaintBackend(**backend_kwargs)
        else:
            raise ValueError(f"Backend desconhecido: {backend!r} (usa 'catvton', 'inpaint' ou 'auto')")

    @staticmethod
    def _auto_select():
        """GPU disponível -> modelo fiel (CatVTON); caso contrário -> baseline local."""
        try:
            import torch
            if torch.cuda.is_available():
                return "catvton"
        except Exception:
            pass
        return "inpaint"

    def generate(self, person, cloth, mask, **kwargs):
        return self.backend.generate(person, cloth, mask, **kwargs)
