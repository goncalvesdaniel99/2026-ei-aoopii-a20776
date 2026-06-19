# Virtual Try-On: Clothes — Relatório

**UC:** Aprendizagem Organizacional — Opção II (CP3 — Knowledge Extraction in Dynamic Environments)
**Instituição:** IPVC / ESTG — 2025/2026
**Track:** A — Deep Learning (Computer Vision)
**Autor:** Daniel Gonçalves — nº 20776

---

## 1. Introdução e objetivo

O *virtual try-on* responde a uma pergunta simples de enunciar e difícil de resolver:
**dada a foto de uma pessoa e a foto de uma peça de roupa, como gerar a pessoa a vestir
essa peça?** É um problema com aplicação direta em *e-commerce* (experimentar roupa sem a
vestir), gestão de guarda-roupa e recomendação de moda.

A dificuldade não é cosmética: é preciso respeitar a **pose do corpo**, deformar o **tecido**
de forma coerente, preservar a **identidade da peça** (cor, padrão, logótipo) e manter
intactas as zonas que não mudam (cara, mãos, fundo). É uma área de investigação ativa; o
objetivo deste trabalho não é superar o estado da arte, mas **compreender o problema** e
entregar uma solução funcional e fiel.

## 2. Dataset

Usámos o **VITON-HD**, um dataset padrão para try-on. Cada exemplo inclui:

| Pasta | Conteúdo |
|---|---|
| `image/`       | Foto real da pessoa (referência / *ground truth*) |
| `agnostic/`    | A mesma pessoa com a roupa do tronco **ocultada a cinzento** (~128,128,128) |
| `cloth/`       | A peça de roupa isolada, sobre fundo branco |
| `cloth-mask/`  | Máscara da peça |
| `image-parse/` | Segmentação semântica do corpo |

A imagem `agnostic` é central na nossa abordagem: a **zona cinzenta** marca exatamente
*onde* a nova peça deve ser colocada — ou seja, é a máscara de *inpainting* de que
precisamos.

## 3. Abordagem e arquitetura

A abordagem é **diffusion-based** (uma das duas vias sugeridas no enunciado, a par da
*warping-based*). O sistema separa claramente o **motor** (`src/tryon.py`) da **interface**
(`src/app.py`), e o motor expõe dois *backends* atrás de uma única fachada `VirtualTryOn`:

```
                         ┌────────────────────────────┐
   Pessoa + Peça  ─────► │      VirtualTryOn           │ ─────► Imagem final
                         │  (escolhe o backend)        │
                         └─────────────┬──────────────┘
                       GPU? │                    │ sem GPU
                            ▼                    ▼
                   ┌─────────────────┐   ┌──────────────────────────┐
                   │ CatVTONBackend  │   │ InpaintBackend (baseline)│
                   │ modelo treinado │   │ SD inpainting + IP-Adapter│
                   │ para try-on     │   │ genérico                  │
                   └─────────────────┘   └──────────────────────────┘
```

A seleção é automática: havendo GPU CUDA usa-se o `CatVTONBackend`; caso contrário, o
`InpaintBackend`. A interface e o notebook do Colab não sabem qual está ativo.

## 4. Metodologia

### 4.1. Extração da máscara
A máscara de *inpainting* é extraída por **limiar de cor** sobre a imagem `agnostic`
(OpenCV `inRange` em torno de 128,128,128), com um *fallback* mais largo para variações de
compressão JPEG. Aplicamos depois uma **dilatação** (para cobrir bem a fronteira da peça)
e um **desfoque gaussiano** nas bordas (para uma transição mais natural). Esta máscara
serve ambos os backends.

> Decisão importante: ao fornecermos a nossa própria máscara, dispensamos o `AutoMasker`
> do CatVTON (baseado em DensePose/SCHP), evitando a instalação de `detectron2`/`densepose`
> — a parte mais frágil do *setup*.

### 4.2. Backend *baseline*: SD Inpainting + IP-Adapter
Primeiro implementámos um *baseline* com `Stable Diffusion Inpainting` (modelo genérico)
condicionado pela peça através do **IP-Adapter**. **Conclusão (uma lição central do
projeto): não funciona para try-on fiel.** Razões:

1. O modelo **não foi treinado para try-on** — apenas preenche a região mascarada com algo
   plausível.
2. O **IP-Adapter** só injeta uma *impressão global* da peça (um embedding CLIP); **não
   preserva a geometria** nem o padrão exato (ex.: o desenho do cão na camisola perde-se).

Ou seja, o baseline tem um **teto de qualidade baixo por construção**. Mantivemo-lo no
projeto precisamente para documentar esta limitação e como modo de desenvolvimento local.

### 4.3. Backend principal: CatVTON
A solução fiel usa o **CatVTON**, um modelo de difusão **treinado especificamente para
try-on**. Em vez de "imaginar" a roupa, aprende a transferir a peça para o corpo
preservando a sua identidade. Recebe `(imagem da pessoa, imagem da peça, máscara)` e
devolve a pessoa vestida. Corre na GPU (T4) do **Google Colab**.

### 4.4. Decisões de engenharia (restrições de hardware)
O hardware disponível (MacBook Air M2 com 8 GB; desktop i5-6600K, 16 GB, **sem GPU
dedicada**) **não permite** correr um modelo de try-on a sério com desempenho aceitável:

- Com 8 GB, o SD entra em *swap* para disco → gerações de **~20 minutos**.
- Sem GPU, qualquer modelo pesado corre em CPU → dezenas de minutos por imagem.

Por isso, o modelo fiel corre na **GPU grátis do Colab**, e a app Gradio é lançada *dentro*
do Colab com `share=True`, gerando um **link público** para a demo. Isto mantém o `src/`
como o coração do projeto (o notebook apenas importa `src/`), dá resultados fiéis e não
depende de serviços de terceiros que possam estar *offline*.

## 5. Resultados

> **A preencher após execução no Colab.** Coloca as imagens em `docs/imagens/` e
> referencia-as abaixo. Sugestão: para cada exemplo, mostrar *pessoa + peça → resultado*,
> e incluir um caso de **transferência cruzada** (pessoa A com a peça de B) para evidenciar
> que a peça é mesmo vestida.

Exemplo de comparação a documentar:

| Pessoa | Peça | Baseline (Inpaint) | CatVTON |
|---|---|---|---|
| `00035_00` | `00071_00` | _(inserir)_ | _(inserir)_ |

<!-- ![Resultado CatVTON](imagens/resultado_catvton.png) -->

Aspetos a comentar nos resultados: fidelidade do padrão/cor, alinhamento ao corpo,
preservação da cara/mãos/fundo, e artefactos observados.

## 6. Limitações e desafios

- **Dependência de GPU externa:** o resultado fiel exige uma sessão Colab ativa; a GPU
  grátis tem limites de tempo.
- **Qualidade da máscara:** uma máscara mal extraída (bordas, dilatação) degrada o
  resultado; é ajustável em `DatasetLoader.load_mask`.
- **Baseline limitado:** mantido apenas como referência/desenvolvimento.

## 7. Trabalho futuro

- Geração automática da máscara para **fotos arbitrárias** (fora do VITON-HD), via
  segmentação de corpo (ex.: SCHP/DensePose ou SAM).
- Comparar o CatVTON com **OOTDiffusion** / **IDM-VTON** e medir qualidade.
- Suportar **upload** de pessoa e peça do utilizador na interface.

## 8. Como correr

Ver o [README](../README.md). Em resumo: para resultado fiel, abrir
`notebooks/colab_tryon.ipynb` no Colab com GPU T4; para baseline local,
`pip install -r requirements.txt` e `python src/app.py`.

## 9. Referências

- **VITON-HD** — Choi et al., *VITON-HD: High-Resolution Virtual Try-On via
  Misalignment-Aware Normalization*, CVPR 2021.
- **CatVTON** — Chong et al., *CatVTON: Concatenation Is All You Need for Virtual Try-On
  with Diffusion Models*, 2024. Repositório: https://github.com/Zheng-Chong/CatVTON
- **IP-Adapter** — Ye et al., *IP-Adapter: Text Compatible Image Prompt Adapter for
  Text-to-Image Diffusion Models*, 2023.
- **Stable Diffusion Inpainting** — Rombach et al., *High-Resolution Image Synthesis with
  Latent Diffusion Models*, CVPR 2022.
