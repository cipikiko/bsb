"""
Experiment 2: Rozpoznávanie tváre – FAR, FRR, EER a útok fotkou

Čo skript robí:
  1. Stiahne dataset LFW (1 000 dvojíc fotiek: 500 tej istej osoby, 500 rôznych).
  2. Modelom Facenet (knižnica DeepFace) vypočíta vzdialenosť tvárí v každej dvojici.
     Vzdialenosti uloží do CSV, takže pri ďalšom spustení sa už nepočítajú znova.
  3. Vypočíta FAR a FRR pre rôzne prahy, nájde EER a nakreslí grafy 3 a 4.
  4. Ak existuje priečinok data/spoof/, otestuje útok fotkou (tabuľka 4).

Štruktúra fotiek pre útok fotkou (mená bez diakritiky):
  data/spoof/<meno>/real/    5 skutočných selfie (prvá slúži ako referencia)
  data/spoof/<meno>/attack/  5 fotiek tej selfie zobrazenej na monitore

Spustenie:  python face.py
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # potlačí zbytočné hlášky TensorFlow

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # grafy sa ukladajú do súborov, neotvárajú sa okná
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepface import DeepFace
from sklearn.datasets import fetch_lfw_pairs

# ---------------------------------------------------------------- nastavenia
MODEL = "Facenet"
DEFAULT_THRESHOLD = 0.40  # predvolený prah DeepFace pre Facenet + kosínusovú vzdialenosť
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
SPOOF_DIR = BASE / "data" / "spoof"
DISTANCES_CSV = RESULTS / "exp2_vzdialenosti_lfw.csv"
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------- pomocné funkcie
def cosine_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embedding_from_array(img_rgb):
    """Embedding z fotky LFW (tvár je už orezaná, detekcia sa preskakuje)."""
    img = img_rgb * 255 if img_rgb.max() <= 1.0 else img_rgb
    bgr = np.ascontiguousarray(img[:, :, ::-1]).astype(np.uint8)  # DeepFace chce BGR
    rep = DeepFace.represent(img_path=bgr, model_name=MODEL,
                             detector_backend="skip", enforce_detection=False)
    return rep[0]["embedding"]


def embedding_from_file(path):
    """Embedding z vlastnej fotky; tvár sa najprv vyhľadá. None = tvár nenájdená."""
    try:
        rep = DeepFace.represent(img_path=str(path), model_name=MODEL,
                                 detector_backend="opencv", enforce_detection=True)
        return rep[0]["embedding"]
    except ValueError:
        return None


def list_images(folder):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


# ---------------------------------------------------------------- 1. a 2. LFW
def compute_lfw_distances():
    if DISTANCES_CSV.exists():
        print(f"Načítavam uložené vzdialenosti z {DISTANCES_CSV}")
        return pd.read_csv(DISTANCES_CSV)

    print("Sťahujem dataset LFW (prvýkrát to môže trvať niekoľko minút)...")
    lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0)
    print(f"Počet dvojíc: {len(lfw.pairs)}. Počítam vzdialenosti...")

    rows = []
    for i, (pair, label) in enumerate(zip(lfw.pairs, lfw.target)):
        d = cosine_distance(embedding_from_array(pair[0]), embedding_from_array(pair[1]))
        rows.append({"dvojica": i, "ta_ista_osoba": int(label), "vzdialenost": d})
        if (i + 1) % 50 == 0:
            print(f"  hotovo {i + 1}/{len(lfw.pairs)}")

    df = pd.DataFrame(rows)
    df.to_csv(DISTANCES_CSV, index=False)
    return df


# ---------------------------------------------------------------- 3. FAR, FRR, EER
def far_frr(df, thresholds):
    same = df.loc[df.ta_ista_osoba == 1, "vzdialenost"].to_numpy()
    diff = df.loc[df.ta_ista_osoba == 0, "vzdialenost"].to_numpy()
    far = np.array([(diff <= t).mean() * 100 for t in thresholds])  # cudzí prijatý
    frr = np.array([(same > t).mean() * 100 for t in thresholds])   # správny odmietnutý
    return far, frr


def analyze(df):
    thresholds = np.linspace(0, df.vzdialenost.max(), 2001)
    far, frr = far_frr(df, thresholds)
    idx = int(np.argmin(np.abs(far - frr)))
    eer_thr, eer = float(thresholds[idx]), float((far[idx] + frr[idx]) / 2)

    pd.DataFrame({"prah": thresholds, "FAR_%": far, "FRR_%": frr}).to_csv(
        RESULTS / "exp2_far_frr_krivka.csv", index=False)

    # Tabuľka 3: FAR a FRR pri vybraných prahoch
    chosen = [(0.30, "prísnejší prah"), (DEFAULT_THRESHOLD, "predvolený prah modelu"),
              (eer_thr, "prah EER"), (0.50, "benevolentnejší prah")]
    chosen.sort(key=lambda x: x[0])
    t_far, t_frr = far_frr(df, [t for t, _ in chosen])
    table3 = pd.DataFrame({
        "prah": [round(t, 3) for t, _ in chosen],
        "popis": [p for _, p in chosen],
        "FAR_%": np.round(t_far, 2),
        "FRR_%": np.round(t_frr, 2),
    })
    table3.to_csv(RESULTS / "exp2_tabulka3_far_frr.csv", index=False)

    # Graf 3: FAR a FRR podľa prahu
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far, label="FAR (prijatie cudzej osoby)")
    plt.plot(thresholds, frr, label="FRR (odmietnutie správnej osoby)")
    plt.axvline(DEFAULT_THRESHOLD, color="gray", linestyle="--",
                label=f"predvolený prah ({DEFAULT_THRESHOLD})")
    plt.scatter([eer_thr], [eer], color="red", zorder=5,
                label=f"EER = {eer:.1f} % (prah {eer_thr:.3f})")
    plt.xlabel("Prah kosínusovej vzdialenosti")
    plt.ylabel("Chybovosť [%]")
    plt.title("FAR a FRR modelu Facenet na datasete LFW")
    plt.xlim(0, min(1.2, df.vzdialenost.max()))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "graf3_far_frr.png", dpi=200)
    plt.close()

    # Graf 4: histogram vzdialeností
    plt.figure(figsize=(8, 5))
    plt.hist(df.loc[df.ta_ista_osoba == 1, "vzdialenost"], bins=40, alpha=0.6,
             label="tá istá osoba")
    plt.hist(df.loc[df.ta_ista_osoba == 0, "vzdialenost"], bins=40, alpha=0.6,
             label="iná osoba")
    plt.axvline(DEFAULT_THRESHOLD, color="gray", linestyle="--", label="predvolený prah")
    plt.xlabel("Kosínusová vzdialenosť tvárí")
    plt.ylabel("Počet dvojíc")
    plt.title("Rozdelenie vzdialeností v datasete LFW")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "graf4_histogram.png", dpi=200)
    plt.close()

    print("\nTabuľka 3:")
    print(table3.to_string(index=False))
    print(f"\nEER = {eer:.2f} % pri prahu {eer_thr:.3f}")
    return eer_thr


# ---------------------------------------------------------------- 4. útok fotkou
def spoof_test(eer_thr):
    if not SPOOF_DIR.exists():
        print(f"\nPriečinok {SPOOF_DIR} neexistuje, útok fotkou preskakujem.")
        return

    rows = []
    for person in sorted(p for p in SPOOF_DIR.iterdir() if p.is_dir()):
        real, attack = list_images(person / "real"), list_images(person / "attack")
        if len(real) < 2:
            print(f"  {person.name}: treba aspoň 2 skutočné fotky, preskakujem.")
            continue
        ref = embedding_from_file(real[0])
        if ref is None:
            print(f"  {person.name}: na referenčnej fotke sa nenašla tvár, preskakujem.")
            continue

        for kind, files in (("skutočná fotka", real[1:]), ("útok fotkou", attack)):
            for f in files:
                emb = embedding_from_file(f)
                d = cosine_distance(ref, emb) if emb is not None else np.nan
                rows.append({
                    "osoba": person.name, "typ": kind, "subor": f.name,
                    "tvar_najdena": emb is not None,
                    "vzdialenost": d,
                    "prijata_predvoleny_prah": bool(emb is not None and d <= DEFAULT_THRESHOLD),
                    "prijata_prah_EER": bool(emb is not None and d <= eer_thr),
                })

    if not rows:
        print("\nŽiadne použiteľné fotky pre útok fotkou.")
        return

    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "exp2_utok_fotkou_detail.csv", index=False)

    # Tabuľka 4: súhrn pre každú osobu
    table4 = detail.groupby(["osoba", "typ"]).agg(
        pocet=("subor", "count"),
        tvar_najdena=("tvar_najdena", "sum"),
        prijate_predvoleny_prah=("prijata_predvoleny_prah", "sum"),
        prijate_prah_EER=("prijata_prah_EER", "sum"),
        priemerna_vzdialenost=("vzdialenost", "mean"),
    ).round(3).reset_index()
    table4.to_csv(RESULTS / "exp2_tabulka4_utok_fotkou.csv", index=False)

    print("\nTabuľka 4 (útok fotkou):")
    print(table4.to_string(index=False))


# ---------------------------------------------------------------- spustenie
if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    distances = compute_lfw_distances()
    threshold_eer = analyze(distances)
    spoof_test(threshold_eer)
    print(f"\nHotovo. Výsledky sú v priečinku '{RESULTS}/'.")
