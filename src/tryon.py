import os

import cv2
import numpy as np
from PIL import Image, ImageFilter


class DatasetLoader:
    """Carrega pessoa, peca e mascara do dataset VITON-HD em data/test."""

    def __init__(self, base_dir=None):
        if base_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(here, "..", "data", "test")
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, "image")
        self.agnostic_dir = os.path.join(base_dir, "agnostic")
        self.cloth_dir = os.path.join(base_dir, "cloth")

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
        # Na imagem agnostic a roupa esta tapada com cinzento (~128). Essa zona e a mascara.
        agnostic = Image.open(os.path.join(self.agnostic_dir, person_id)).convert("RGB")
        bgr = cv2.cvtColor(np.array(agnostic), cv2.COLOR_RGB2BGR)

        mask = cv2.inRange(bgr, (125, 125, 125), (131, 131, 131))
        if mask.sum() == 0:
            mask = cv2.inRange(bgr, (110, 110, 110), (140, 140, 140))

        if dilate:
            kernel = np.ones((dilate, dilate), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        out = Image.fromarray(mask).convert("L")
        if blur:
            out = out.filter(ImageFilter.GaussianBlur(blur))
        return out


class TryOnBackend:
    name = "base"

    def generate(self, person, cloth, mask, **kwargs):
        raise NotImplementedError


class CatVTONBackend(TryOnBackend):
    # Try-on fiel com o CatVTON (precisa de GPU). Usa a nossa mascara, sem AutoMasker.
    name = "catvton"

    def __init__(self, catvton_repo="zhengchong/CatVTON",
                 base_model="booksforcharlie/stable-diffusion-inpainting",
                 width=768, height=1024, device="cuda"):
        import torch
        from model.pipeline import CatVTONPipeline

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
    # Baseline em CPU/MPS: SD inpainting + IP-Adapter. Nao faz try-on fiel.
    name = "inpaint"

    def __init__(self, model_id=None, device=None, use_ip_adapter=None):
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        if model_id is None:
            model_id = os.environ.get(
                "TRYON_INPAINT_MODEL", "stable-diffusion-v1-5/stable-diffusion-inpainting"
            )
        if use_ip_adapter is None:
            use_ip_adapter = os.environ.get("TRYON_IP_ADAPTER", "1") == "1"
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.use_ip_adapter = use_ip_adapter

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
            # attention slicing nao e compativel com o IP-Adapter; so o usamos sem ele.
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


class VirtualTryOn:
    """Escolhe o backend e expoe um unico metodo generate."""

    def __init__(self, backend="auto", **backend_kwargs):
        if backend == "auto":
            backend = self._auto_select()
        self.backend_name = backend

        if backend == "catvton":
            self.backend = CatVTONBackend(**backend_kwargs)
        elif backend == "inpaint":
            self.backend = InpaintBackend(**backend_kwargs)
        else:
            raise ValueError(f"Backend desconhecido: {backend!r}")

    @staticmethod
    def _auto_select():
        try:
            import torch
            if torch.cuda.is_available():
                return "catvton"
        except Exception:
            pass
        return "inpaint"

    def generate(self, person, cloth, mask, **kwargs):
        return self.backend.generate(person, cloth, mask, **kwargs)
