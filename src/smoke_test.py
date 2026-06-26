"""
Smoke test — valida o dataset e a extração de máscara SEM carregar nenhum modelo.
Corre em segundos. Serve para confirmar que os caminhos, o VITON-HD e a máscara estão OK
antes de gastar tempo a descarregar/correr o modelo de difusão.

Uso:
    python src/smoke_test.py
"""

import os
import sys

import numpy as np

try:
    from tryon import DatasetLoader
except ImportError:
    from src.tryon import DatasetLoader


def main():
    data = DatasetLoader()
    models = data.get_available_models()
    garments = data.get_available_garments()

    print(f"Pasta de dados : {os.path.abspath(data.base_dir)}")
    print(f"Pessoas (image): {len(models)}")
    print(f"Peças  (cloth) : {len(garments)}")

    if not models or not garments:
        print("\n❌ Dataset vazio. Confirma que data/test/image e data/test/cloth têm imagens.")
        sys.exit(1)

    person_id = models[0]
    cloth_id = garments[0]
    print(f"\nA testar com pessoa={person_id} | peça={cloth_id}")

    person = data.load_person(person_id)
    cloth = data.load_cloth(cloth_id)
    mask = data.load_mask(person_id)

    mask_arr = np.array(mask)
    ratio = (mask_arr > 10).mean()  # fração de pixéis "ligados" na máscara
    print(f"  pessoa : {person.size}")
    print(f"  peça   : {cloth.size}")
    print(f"  máscara: {mask.size} | pixéis ativos: {ratio:.1%}")

    # Guarda uma pré-visualização da máscara para inspeção visual.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mask_preview.png")
    mask.save(out)
    print(f"\n🖼️  Pré-visualização da máscara guardada em: {os.path.abspath(out)}")

    if ratio < 0.02:
        print("\n⚠️  Máscara quase vazia — a extração de cinzento pode estar a falhar.")
        print("   Ajusta os limiares em DatasetLoader.load_mask.")
        sys.exit(2)

    print("\n✅ Smoke test OK — dataset e máscara prontos. Podes correr a app.")


if __name__ == "__main__":
    main()
