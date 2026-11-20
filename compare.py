"""Experiment 3. Spustenie: python compare.py (vyžaduje matplotlib)."""

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
# všetky vstupy sú výstupy passwords.py a face.py v priečinku results/
PASSWORD_SCORES = Path("results") / "table1_passwords.csv"
CRACK_TIMES = Path("results") / "crack_times_by_category.csv"
LFW_DISTANCES = Path("results") / "exp2_vzdialenosti_lfw.csv"
PHOTO_RESULTS = Path("results") / "exp2_tabulka4_utok_fotkou.csv"

ATTEMPTS = 5
FACE_THRESHOLD = 0.40
CONFIDENCE = 0.95
CONFIDENCE_LABEL = f"{100 * CONFIDENCE:.0f} %"
ATTEMPT_METRIC = f"Aspoň jeden úspech pri {ATTEMPTS} pokusoch"
FACE_VARIANT = f"Prah {FACE_THRESHOLD:.2f}".replace(".", ",")
PASSWORD_GROUPS = {
    "1_velmi_slabe": "veľmi slabé",
    "2_slabe": "slabé",
    "3_stredne": "stredné",
    "4_silne": "silné",
    "5_velmi_silne": "veľmi silné",
}


@dataclass(frozen=True)
class Result:
    scenar: str
    metoda: str
    variant: str
    ukazovatel: str
    typ_hodnoty: str
    hodnota: float | None = None
    jednotka: str = ""
    pocet_pripadov: int | None = None
    velkost_vzorky: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    poznamka: str = ""
    zdroj: str = ""


def read_rows(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def password_estimates():
    by_group = {label: [] for label in PASSWORD_GROUPS.values()}
    for row in read_rows(PASSWORD_SCORES):
        guesses = float(row["guesses"])
        if not math.isfinite(guesses) or guesses < 1:
            raise ValueError(f"Neplatný odhad počtu pokusov v {PASSWORD_SCORES}.")
        by_group[row["category_label"]].append(100 * min(ATTEMPTS / guesses, 1))

    results = []
    for group, values in by_group.items():
        if not values:
            raise ValueError(f"V {PASSWORD_SCORES} chýbajú heslá zo skupiny {group!r}.")
        results.append(Result(
            "Online hádanie", "Heslo", group, ATTEMPT_METRIC, "modelový medián",
            median(values), "%", velkost_vzorky=len(values),
            minimum=min(values), maximum=max(values),
            poznamka=(f"Model min(1, {ATTEMPTS}/G) predpokladá G rovnako pravdepodobných "
                      "možností a rôzne pokusy. Odhad zxcvbn slúži ako aproximácia G."),
            zdroj=PASSWORD_SCORES.as_posix(),
        ))
    return results


def face_verification():
    distances = {"0": [], "1": []}
    for row in read_rows(LFW_DISTANCES):
        distance = float(row["vzdialenost"])
        if not math.isfinite(distance):
            raise ValueError(f"Neplatná vzdialenosť v {LFW_DISTANCES}.")
        distances[row["ta_ista_osoba"]].append(distance)

    different, same = distances["0"], distances["1"]
    if not different or not same:
        raise ValueError(f"V {LFW_DISTANCES} musia byť dvojice oboch druhov.")

    false_accepts = sum(value <= FACE_THRESHOLD for value in different)
    false_rejects = sum(value > FACE_THRESHOLD for value in same)
    far = false_accepts / len(different)
    far_observed = Result(
        "Online hádanie", "FaceNet", FACE_VARIANT, "FAR pri jednom porovnaní",
        "pozorovaný podiel", 100 * far, "%", false_accepts, len(different),
        poznamka="Dvojice rôznych osôb z LFW.", zdroj=LFW_DISTANCES.as_posix(),
    )
    frr = Result(
        "Odmietnutie oprávneného používateľa", "FaceNet", FACE_VARIANT,
        "FRR pri jednom porovnaní", "pozorovaný podiel",
        100 * false_rejects / len(same), "%", false_rejects, len(same),
        poznamka="Dvojice tej istej osoby z LFW.", zdroj=LFW_DISTANCES.as_posix(),
    )

    online = []
    if false_accepts == 0:
        # Exact one-sided binomial upper bound for zero observed events.
        bound_label = f"jednostranná {CONFIDENCE_LABEL} horná hranica"
        log_failure = math.log1p(-CONFIDENCE) / len(different)
        upper_far = -math.expm1(log_failure)
        upper_multi = -math.expm1(ATTEMPTS * log_failure)
        online.append(Result(
            "Online hádanie", "FaceNet", FACE_VARIANT, "FAR pri jednom porovnaní",
            bound_label, 100 * upper_far, "%",
            false_accepts, len(different),
            poznamka="Presná binomická hranica pri nezávislých dvojiciach a 0 prijatiach.",
            zdroj=LFW_DISTANCES.as_posix(),
        ))
        multi = Result(
            "Online hádanie", "FaceNet", FACE_VARIANT, ATTEMPT_METRIC,
            bound_label, 100 * upper_multi, "%",
            false_accepts, len(different),
            poznamka=f"Prepočet hornej hranice FAR. Model predpokladá {ATTEMPTS} nezávislých pokusov.",
            zdroj=LFW_DISTANCES.as_posix(),
        )
    else:
        multi = Result(
            "Online hádanie", "FaceNet", FACE_VARIANT, ATTEMPT_METRIC,
            "modelový odhad", 100 * (1 - (1 - far) ** ATTEMPTS), "%",
            false_accepts, len(different),
            poznamka=f"Prepočet pozorovaného FAR pre {ATTEMPTS} nezávislých pokusov.",
            zdroj=LFW_DISTANCES.as_posix(),
        )
    online.append(multi)
    return far_observed, online, frr


def offline_estimates():
    for row in read_rows(CRACK_TIMES):
        yield Result(
            "Únik databázy", "Heslo", f"{PASSWORD_GROUPS[row['category']]} / {row['algorithm']}",
            "Geometrický priemer odhadu času prelomenia", "odhad Experimentu 1",
            float(row["crack_time_s"]), "s",
            poznamka="Závisí od odhadu zxcvbn a rýchlosti hashovania na testovacom CPU.",
            zdroj=CRACK_TIMES.as_posix(),
        )
    yield Result(
        "Únik databázy", "Biometria tváre", "Biometrická šablóna",
        "Následky úniku šablóny", "diskusia",
        poznamka="Následky závisia od ochrany a možnosti výmeny uloženej šablóny.",
    )


def photo_acceptance():
    photos = [row for row in read_rows(PHOTO_RESULTS) if row["typ"] == "útok fotkou"]
    tested = sum(int(row["pocet"]) for row in photos)
    accepted = sum(int(row["prijate_predvoleny_prah"]) for row in photos)
    if not 0 <= accepted <= tested or tested == 0:
        raise ValueError(f"Neplatný počet pokusov s fotkou v {PHOTO_RESULTS}.")
    return Result(
        "Podvrh", "FaceNet", FACE_VARIANT, "Prijatie fotky v prezentačnom teste",
        "pozorovaný podiel", 100 * accepted / tested, "%", accepted, tested,
        poznamka=f"Laboratórny test na {len(photos)} osobách.", zdroj=PHOTO_RESULTS.as_posix(),
    )


def vendor_references():
    return [
        Result(
            "Referenčné údaje výrobcov", "Apple Face ID", "Jeden zaregistrovaný vzhľad",
            "Pravdepodobnosť odomknutia náhodnou osobou", "výrobca uvádza menej než",
            0.0001, "%", poznamka="Údaj Apple pre iPhone alebo iPad Pro.",
            zdroj="https://support.apple.com/en-us/102381",
        ),
        Result(
            "Referenčné údaje výrobcov", "Microsoft Windows Hello", "Windows 11 s IR snímačom",
            "FAR pri rozpoznávaní tváre", "výrobca uvádza menej než", 0.001, "%",
            poznamka="Hodnota z dokumentácie Microsoftu pre náhodného používateľa.",
            zdroj="https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/"
                  "windows-hello-face-authentication#how-accuracy-is-measured",
        ),
    ]


def plot_comparison(passwords, face, observed_far):
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    fig.subplots_adjust(left=0.20, right=0.95, top=0.79, bottom=0.30)
    axis_rows = list(range(len(passwords)))
    values = [row.hodnota for row in passwords]
    password_markers = ax.errorbar(
        values, axis_rows,
        xerr=[[value - row.minimum for row, value in zip(passwords, values)],
              [row.maximum - value for row, value in zip(passwords, values)]],
        fmt="o", markersize=6, capsize=3, color="#245982", elinewidth=1.5,
        label="Heslá · medián a rozsah v skupine", zorder=3,
    )
    is_upper = observed_far.pocet_pripadov == 0
    face_label = (f"FaceNet · {CONFIDENCE_LABEL} horná hranica" if is_upper
                  else "FaceNet · modelový odhad")
    face_marker = ax.scatter(face.hodnota, len(passwords), c="#9D410E",
                             marker="<" if is_upper else "D", s=85, label=face_label, zorder=3)
    ax.annotate(f"{face.hodnota:.2f} %".replace(".", ","),
                (face.hodnota, len(passwords)), xytext=(-9, 11), textcoords="offset points",
                color="#873907", ha="right", fontsize=10)

    ax.set_xscale("log")
    ax.set_xlim(min(face.hodnota, *(row.minimum for row in passwords)) / 4, 125)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.set_yticks([*axis_rows, len(passwords)], [*(row.variant for row in passwords), "FaceNet"])
    ax.invert_yaxis()
    ax.set_xlabel("Pravdepodobnosť aspoň jedného úspechu [%]", labelpad=10)
    ax.grid(axis="x", color="#DAE3EA", alpha=0.9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BECAD4")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", color="#BECAD4")
    ax.legend(handles=[password_markers, face_marker], loc="lower left",
              bbox_to_anchor=(0, 1.015), ncols=2, borderaxespad=0, frameon=False)

    fig.text(0.20, 0.92, f"Online overenie pri {ATTEMPTS} pokusoch",
             fontsize=17, weight="bold", color="#142633")
    threshold = f"{FACE_THRESHOLD:.2f}".replace(".", ",")
    fig.text(0.20, 0.075,
             f"FaceNet pri prahu {threshold} prijal {observed_far.pocet_pripadov} z "
             f"{observed_far.velkost_vzorky} cudzích dvojíc. Prepočet predpokladá nezávislé pokusy.\n"
             "Model hesiel predpokladá rovnako pravdepodobné možnosti, ich počet aproximuje zxcvbn.",
             fontsize=9, color="#354F61", linespacing=1.5, va="bottom")
    fig.savefig(RESULTS / "graf5_porovnanie.png", dpi=200)
    plt.close(fig)


def main():
    passwords = password_estimates()
    observed_far, face_online, frr = face_verification()
    rows = [
        *passwords, observed_far, *face_online, *offline_estimates(),
        Result("Podvrh", "Heslo", "Phishing a odpozorovanie", "Získanie hesla", "diskusia"),
        photo_acceptance(),
        Result("Odmietnutie oprávneného používateľa", "Heslo", "Zadanie hesla",
               "Chybovosť pri zadávaní", "bez merania"),
        frr, *vendor_references(),
    ]

    RESULTS.mkdir(exist_ok=True)
    plot_comparison(passwords, face_online[-1], observed_far)
    table = RESULTS / "tabulka5_porovnanie.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=[field.name for field in fields(Result)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    print(f"Graf 5 a tabuľka 5 sú v {RESULTS}")


if __name__ == "__main__":
    main()
