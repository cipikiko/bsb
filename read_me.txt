POROVNANIE BIOMETRICKEJ A HESLOVEJ AUTENTIFIKÁCIE Z HĽADISKA BEZPEČNOSTI
Praktická časť semestrálneho zadania
=========================================================================

Projekt obsahuje tri experimenty, každý v samostatnom skripte. Všetky výsledky
(tabuľky CSV a grafy PNG) sa ukladajú do priečinka results/. Čísla tabuliek
a grafov zodpovedajú technickej správe.

  passwords.py  Experiment 1: sila hesiel a rýchlosť hashovania
                (SHA-256, bcrypt, Argon2id)       -> tabuľky 3.1-3.2, grafy 3.1-3.2
  face.py       Experiment 2: rozpoznávanie tváre (FaceNet, dataset LFW),
                FAR, FRR, EER a útok fotkou       -> tabuľky 4.1-4.2, grafy 4.1-4.2
  compare.py    Experiment 3: porovnanie oboch metód
                                                  -> tabuľka 5.1, graf 5.1


ŠTRUKTÚRA PRIEČINKA
-------------------
  passwords.py, face.py, compare.py   skripty experimentov
  data/passwords.csv                  15 vymyslených hesiel v 5 kategóriách
  data/spoof/                         vlastné fotky tvárí (nie sú zverejnené)
  results/                            vygenerované tabuľky a grafy
  requirements.txt                    zoznam potrebných knižníc


POŽIADAVKY
----------
  - Python 3.11 alebo novší
  - internet pri prvom spustení face.py (stiahne sa dataset LFW a model
    FaceNet, spolu niekoľko stoviek MB)


INŠTALÁCIA
----------
  python -m venv .venv
  .venv\Scripts\activate          (Windows)
  source .venv/bin/activate       (Linux, macOS)
  pip install -r requirements.txt


SPUSTENIE
---------
Skripty sa spúšťajú z tohto priečinka v uvedenom poradí:

  python passwords.py     približne 1-3 minúty
  python face.py          prvé spustenie 10-30 minút, ďalšie pár sekúnd
  python compare.py       pár sekúnd

compare.py nič nemeria, iba spracuje výsledky prvých dvoch skriptov, preto sa
spúšťa ako posledný. Počas merania v passwords.py nespúšťajte iné náročné
programy. Rýchlosť hashovania závisí od procesora.


FOTKY PRE ÚTOK FOTKOU
---------------------
Fotky tvárí nie sú súčasťou repozitára. Pri spustení bez nich sa útok fotkou
preskočí a ostane uložená tabuľka results/exp2_tabulka4_utok_fotkou.csv.
Na vlastný test treba fotky uložiť takto (mená bez diakritiky):

  data/spoof/<meno>/real/     01.jpg ... 05.jpg  (01.jpg je referenčná fotka)
  data/spoof/<meno>/attack/   01.jpg ... 05.jpg  (fotky tváre zobrazenej na monitore)


HLAVNÉ VÝSTUPY V PRIEČINKU results/
-----------------------------------
  table1_passwords.csv             tabuľka 3.1  heslá a odhad zxcvbn
  table2_hash_speed.csv            tabuľka 3.2  rýchlosť hashovania
  graph1_hash_speed.png            graf 3.1
  graph2_crack_time.png            graf 3.2     čas prelomenia
  exp2_tabulka3_far_frr.csv        tabuľka 4.1  FAR a FRR pri prahoch
  exp2_tabulka4_utok_fotkou.csv    tabuľka 4.2  útok fotkou
  graf4_histogram.png              graf 4.1
  graf3_far_frr.png                graf 4.2
  tabulka5_porovnanie.csv          tabuľka 5.1  porovnanie scenárov
  graf5_porovnanie.png             graf 5.1

Ostatné súbory obsahujú surové merania a medzivýsledky.


OBMEDZENIA
----------
  - Hashovanie sa meria na jednom vlákne procesora. Útočník s grafickou
    kartou je pri SHA-256 rádovo rýchlejší.
  - Odhad sily hesla knižnicou zxcvbn je len približný.
  - FaceNet nemá detekciu živosti, výsledky neplatia pre systémy
    s 3D alebo infračerveným snímaním (Face ID, Windows Hello).
  - Vzorky sú malé (500 dvojíc cudzích osôb, 15 útokov fotkou).
  - Všetky heslá sú vymyslené.
