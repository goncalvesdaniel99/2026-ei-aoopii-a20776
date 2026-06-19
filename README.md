# Virtual Try-On: Clothes

Projeto da UC **Aprendizagem Organizacional — Opção II** (CP3 — Knowledge Extraction in
Dynamic Environments), IPVC/ESTG, 2025/2026.

## Membros do grupo
- Daniel Gonçalves — nº 20776

## Track
**A — Deep Learning** (Computer Vision)

## Descrição do projeto
Dada a **foto de uma pessoa** e a **foto de uma peça de roupa**, o sistema gera a pessoa
**a vestir essa peça**, adaptando-a ao corpo. Domínio: *e-commerce*, gestão de guarda-roupa
e recomendação de moda.

A abordagem é **diffusion-based**. O motor está desenhado com dois backends por trás da
mesma interface (`src/tryon.py`):

| Backend | Modelo | Qualidade | Hardware |
|---|---|---|---|
| `catvton`  | CatVTON (treinado para try-on) | Fiel — preserva a peça e adapta-a ao corpo | GPU (Colab) |
| `inpaint`  | Stable Diffusion Inpainting + IP-Adapter | Baseline (limitada) | CPU / Apple MPS |

O backend `inpaint` é o nosso **baseline** e documenta uma lição do projeto: um modelo de
inpainting *genérico* não faz try-on fiel, porque não foi treinado para a tarefa. O backend
`catvton` usa um modelo específico de try-on e dá os resultados fiéis. A seleção é
automática: havendo GPU usa `catvton`, caso contrário `inpaint`.

## Tech stack
- **Linguagem:** Python
- **Deep Learning:** PyTorch, 🤗 Diffusers, CatVTON, IP-Adapter
- **Visão computacional:** OpenCV, Pillow
- **Interface:** Gradio
- **Dataset:** VITON-HD (subconjunto de teste em `data/test/`)

## Estrutura
```
src/        código-fonte (motor tryon.py + interface app.py)
notebooks/  experiências + colab_tryon.ipynb (corre o modelo fiel na GPU do Colab)
data/       dataset VITON-HD (em .gitignore — ver "Como correr")
docs/       relatório
```

## Como correr

### Opção 1 — Resultado fiel, no Google Colab (recomendado)
Sem GPU local, esta é a forma de obter try-on fiel.
1. Abre `notebooks/colab_tryon.ipynb` no Google Colab.
2. `Runtime > Change runtime type > T4 GPU`.
3. Corre as células por ordem. Carrega o dataset (upload de zip ou Google Drive) para
   `data/test`.
4. A última célula lança a app Gradio com um **link público** para a demo.

### Opção 2 — Baseline local (CPU / Mac MPS)
Corre na própria máquina, mas com qualidade limitada (ver tabela acima).
```bash
pip install -r requirements.txt
python src/app.py
```
A app abre em `http://127.0.0.1:7860`. Variáveis úteis:
- `TRYON_BACKEND=auto|catvton|inpaint` (default `auto`)
- `GRADIO_SHARE=1` para gerar link público (usado no Colab)

### Dados
O `data/` está em `.gitignore`. Coloca o subconjunto de teste do VITON-HD em
`data/test/` com as subpastas `image/`, `agnostic/` e `cloth/`.
