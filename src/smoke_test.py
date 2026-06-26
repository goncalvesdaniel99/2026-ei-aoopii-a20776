# Valida o dataset e a extracao de mascara sem carregar modelos. Corre em segundos.
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
    print(f"Pecas  (cloth) : {len(garments)}")

    if not models or not garments:
        print("\nDataset vazio. Confirma que data/test/image e data/test/cloth tem imagens.")
        sys.exit(1)

    person_id = models[0]
    cloth_id = garments[0]
    print(f"\nA testar com pessoa={person_id} | peca={cloth_id}")

    person = data.load_person(person_id)
    cloth = data.load_cloth(cloth_id)
    mask = data.load_mask(person_id)

    mask_arr = np.array(mask)
    ratio = (mask_arr > 10).mean()
    print(f"  pessoa : {person.size}")
    print(f"  peca   : {cloth.size}")
    print(f"  mascara: {mask.size} | pixeis ativos: {ratio:.1%}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mask_preview.png")
    mask.save(out)
    print(f"\nPre-visualizacao da mascara guardada em: {os.path.abspath(out)}")

    if ratio < 0.02:
        print("\nMascara quase vazia. Ajusta os limiares em DatasetLoader.load_mask.")
        sys.exit(2)

    print("\nSmoke test OK. Dataset e mascara prontos.")


if __name__ == "__main__":
    main()
