# FastStrata: co się zmieniło, wyniki, ustawienia

Szczegóły do [README](../README.pl.md): skąd pochodzi FastStrata, każda zmiana i co dała, cztery presety, komputer
testowy, każdy wynik, karta 4 GB, co setup proponuje według karty i opcje silnika. Jak zmierzono każdą liczbę,
przebieg po przebiegu: [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) (po angielsku).

**Spis treści:** [Skąd pochodzi](#skąd-pochodzi) · [Co zmieniła FastStrata](#co-zmieniła-faststrata) ·
[Modele i presety](#modele-i-presety) · [Komputer testowy](#komputer-testowy) · [Wyniki](#wyniki) ·
[Karty 4 GB](#na-karcie-4-gb-do-gier) · [Czego potrzebujesz](#czego-potrzebujesz) · [Konfiguracja](#konfiguracja) ·
[Jak to zmierzono](#jak-to-zmierzono)

## Skąd pochodzi

FastStrata jest trzecim projektem w linii:

| Projekt | Autor | Czym jest |
| --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) | Niko1221 i współtwórcy | Oryginał. Silnik `strata` uruchamia Qwen3.8-Flash-Next, model z 24 576 ekspertów, z których każde słowo potrzebuje 10: karta graficzna trzyma ekspertów wybieranych najczęściej i w trakcie używania uczy się, którzy to są, wszyscy eksperci są w RAM, a procesor w tym samym czasie liczy tych, których karta nie ma; warstwa szkicu (draft) zgaduje kolejne słowa, a model je sprawdza. Do tego: instalator jednym kliknięciem, serwer z API zgodnym z OpenAI i Anthropic oraz aplikacja webowa z czatem i podglądem na żywo. |
| [PolyStrata](https://github.com/VecSzn/PolyStrata) | VecSzn | Dodaje drugi silnik, `strata-poly`, który czyta plik GGUF modelu tam, gdzie leży, bez drugiej kopii: najpierw GLM 5.3 Flash, potem Qwen3.6-35B-A3B, Ornith 1.5, Gemma 4, Tiel Coder i DeepSeek V4 Flash. Gęste wagi są w VRAM; eksperci routowani są dzieleni między pamięć podręczną w VRAM, wypełnianą według tego, jak często każdy ekspert jest używany, i podążającą za tym, czego używa odpowiedź, a procesor, który liczy pozostałych z RAM, gdy karta robi swoją część. Własny blok szkicu modelu (MTP) zgaduje do trzech tokenów naprzód, a model sprawdza je w jednym przebiegu. |
| **FastStrata** | ten fork | Zawęża PolyStrata do dwóch z tych modeli, które mieszczą się na komputerze do gier, Qwen3.6-35B-A3B i Ornith 1.5, razem czterech presetów, i sprawia, że `strata-poly` jest szybki na małej karcie: dziesięć zmierzonych kroków, dziewięć zmian w silniku i jego ustawieniach ([niżej](#co-zmieniła-faststrata)). |

PolyStrata trzymała pamięć podręczną uwagi w 16 bitach, 22 KB na token dla Qwen3.6, co na karcie 6 GB mieści 72K.
FastStrata dodała pamięć 8-bitową, `--kv-q8` (17.10.2026, 11,7 KB na token: 144K na karcie 6 GB), i 4-bitową,
`--kv-q4` (krok 9: 184K, a z kluczami także w 4 bitach cały kontekst modelu, 256K).

Do Qwen3.8-Flash-Next użyj [Strata](https://github.com/Niko1221/Strata). Pozostałe modele PolyStrata nie są częścią
FastStrata: setup nadal je instaluje, gdy wskaże się je przez `--family`, tak jak ustawiała je PolyStrata, i informuje,
że FastStrata ich nie wspiera.

## Co zmieniła FastStrata

Każdą zmianę zmierzono na [komputerze testowym](#komputer-testowy) z Qwen3.6-35B-A3B UD-Q4_K_M, uruchamiając na zmianę
wersję sprzed zmiany i wersję ze zmianą. „Bezstratna” znaczy, że model liczy te same sumy, więc odpowiedzi zmieniają się
nie bardziej niż między dwoma przebiegami tej samej wersji.

| Krok | Opcja | Co robi | Co dała na komputerze testowym | Odpowiedzi |
| ---: | --- | --- | --- | --- |
| | `--kv-q8` (17.10.2026) | Pamięć podręczna uwagi w 8 bitach: 11,7 KB na token zamiast 22 KB. | Kontekst 144K na karcie 6 GB, gdzie 16 bitów mieści 72K. | Stratna: średnia dywergencja KL 0,0025–0,0038 względem pamięci 16-bitowej; w 4500 krokach zmienił się jeden token, którego tamta była pewna w 60% lub więcej. |
| 1 | `--threads 8` | Jeden wątek procesora na rdzeń. Silnik brał wszystkie 16 wątków, przez co jego własny wątek, ten, który podaje karcie pracę, nie miał rdzenia. | Po prompcie 37K 46,3 i 46,7 tok/s zamiast 41,7 i 42,6. | Bezstratna. |
| 2 | pamięć rozmów 12 GiB, 8 miejsc | Więcej rozmów agenta programistycznego wraca bez ponownego czytania (było 8 GiB, 4 miejsca). | Nie mierzono pod kątem szybkości. | Bezstratna. |
| 3 | `--attn-split` | Uwaga odpowiedzi w jednym przejściu po pamięci podręcznej, na rdzeniach tensorowych. | Po 37K 60,2 i 61,7 tok/s zamiast 46,1 i 48,5; jedno wywołanie uwagi przy 37K 167 µs zamiast 649. | Iloczyny liczone na połówkach 16-bitowych: najbardziej prawdopodobny token zmienił się w 4 z 256 kroków, za każdym razem przy prawie remisie. |
| 4 | `--graph-experts` | Każda warstwa odpowiedzi jako jeden graf CUDA, razem z ekspertami w VRAM; karta i procesor dają sobie znać przez pamięć przypiętą (pinned). | O 10% krótszy czas okna tokenów po 2K, o 7% po 37K. | Bezstratna: te same sumy, bit w bit. |
| 5 | `--elastic-prefill 1024` | Prompt czytany w wywołaniach po 2048 tokenów zamiast 512, w VRAM, którą pamięć ekspertów pożycza na czas czytania. | Czytanie 2K x1,49, 37K x1,30 (36,0 s zamiast 47,0), 139K x1,07; pisanie bez zmian; 144K nadal się mieści. | Sumy takie jak przy `--prompt-batch 2048`. |
| 6 | (zmierzone, odrzucone) | Miara tego, ile ustawienie kosztuje odpowiedzi, i dwa nowe stratne pokrętła. | Zob. [niżej](#zmierzone-i-odrzucone). | |
| 7 | `--suffix-drafts 256` | Zgadywanie na podstawie wszystkich rozmów, które silnik widział od startu, w 160 MiB RAM (SuffixDecoding). | Zadanie agenta podane ponownie w nowej rozmowie 99,3 tok/s zamiast 88,9; zwykły czat bez zmian. | Bezstratna: model sprawdza każdy zgadnięty token. |
| 8 | `--cpu-multi` | Eksperci Q4_K, Q5_K i Q6_K na procesorze: każdy wiersz czytany i rozpakowywany raz dla wszystkich tokenów okna. | Eksperci procesora z prędkością 40–43 GB/s zamiast 30–40, z 42–43, które daje zwykły odczyt RAM; po 2K 78,2 tok/s zamiast 74,6 (1–2% w późniejszym sprawdzeniu). | Bezstratna: bity ggml. |
| 9 | `--kv-q4`, `--kv-q4-keys q4` | Wartości pamięci podręcznej uwagi w 4 bitach (8,9 KB na token), potem także klucze (6,2 KB), dla karty za małej na kontekst. | 184K i cały kontekst 256K na karcie 6 GB, gdzie `--kv-q8` mieści 144K. | Stratna: średnia KL 0,0040–0,0079 i 0,0090–0,0127; 8 i 10 zmienionych pewnych tokenów w 4500 krokach. Wyłączona, chyba że kontekst jej wymaga. |
| 10 | cztery presety | Stary silnik kontra nowy dla każdego presetu, ile mniejsze pliki kosztują odpowiedzi, i symulowana karta 4 GB. | [Wyniki](#wyniki). | |

`--tail-skip 6`, które pomija eksperta spoza VRAM, gdy każdy token okna stawia go na siódmym lub ósmym miejscu, to
ustawienie PolyStrata dla Qwen3.6 i zostaje: x1,15 na tym samym tekście przy średniej KL 0,0060 (krok 6). Ornith 1.5
działa bez niego, tak jak w PolyStrata: ile kosztuje odpowiedzi Ornith, nie zmierzono.

### Zmierzone i odrzucone

- **`--weight-cut P`** (krok 6), które zostawia ekspertów, dopóki nie zbiorą części P wagi routingu tokenu: wolniejsze
  od tail skip przy tym samym koszcie (x1,09 zamiast x1,15 przy KL 0,006).
- **`--defer-experts K`** (krok 6, Expert Deferral z KTransformers), ostatni eksperci procesora liczeni, gdy karta
  pracuje dalej: z `--tail-skip 6 --defer-experts 5` o 5% szybciej na tym samym tekście w sprawdzeniu, ale w 3000
  krokach zmieniło 3 tokeny, których referencja była pewna w 60% lub więcej, gdzie dzisiejsze ustawienie nie zmieniło
  żadnego. Zmieniony pewny token w kodzie to błędna nazwa albo nawias, więc zostaje wyłączone, w silniku, dopóki
  benchmark zadań nie pokaże, że jest nieszkodliwe. `--tail-skip 5` (x1,20, KL 0,0102, dwa zmienione pewne tokeny)
  odrzucono z tego samego powodu.
- **Strony 2 MB i zablokowany RAM** (krok 8): eksperci procesora skopiowani do stron 2 MB nie czytają się szybciej niż
  plik modelu w stronach 4 KB (najlepsze rundy 0,8–1,6% od siebie) i zablokowałyby 18,7 GiB RAM, których Windows nie
  może wyrzucić do pliku stronicowania.
- **Kopia ekspertów przepakowana dla procesora** (krok 8): `--cpu-multi` czyta ich już z prędkością RAM, a kopia
  zajmuje kolejne 18,7 GiB.
- **Czytanie z wyprzedzeniem ekspertów następnej warstwy**, gdy karta liczy bieżącą (krok 8): w granicach rozrzutu
  między przebiegami.

## Modele i presety

| Preset | Plik | RAM w użyciu podczas pracy | VRAM | Eksperci | Koszt w odpowiedziach |
| --- | ---: | ---: | --- | --- | --- |
| **Qwen3.6-35B-A3B UD-Q4_K_M** ([Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) | 22,7 GB | 24,1–27,2 GB | 4,7 GB przed jakimkolwiek ekspertem; 962 ekspertów (1,7 GiB) mieści się na karcie 6 GB na starcie | Q4_K, Q5_K, Q6_K: działa na nich każda opcja | punkt odniesienia |
| **Qwen3.6-35B-A3B UD-Q3_K_M** | 17,1 GB | 18,5–21,7 GB | 4,8 GB; 1341 ekspertów (1,7 GiB) | IQ3_XXS, IQ4_XS: ani `--cpu-multi`, ani `--graph-experts` | względem UD-Q4_K_M: średnia KL 0,020, 5 zmienionych pewnych tokenów w 1500 krokach, wszystkie w prozie lub interpunkcji |
| **Ornith 1.5 APEX-I-QUALITY** ([mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF)) | 23,7 GB | 25,5–28,6 GB | 4,4 GB; 1080 ekspertów (1,9 GiB) | Q6_K, Q5_K, IQ4_XS: `--cpu-multi` na 20 z 41 warstw | punkt odniesienia |
| **Ornith 1.5 APEX-I-COMPACT** | 17,4 GB | 19,2–22,4 GB | 4,2 GB; 1823 ekspertów (2,4 GiB) | Q3_K, Q4_K: `--cpu-multi` na 10 z 41 warstw | względem APEX-I-QUALITY: średnia KL 0,088, 26–27 zmienionych pewnych tokenów w 1320 krokach, w tym nazwy w kodzie |

- **RAM w użyciu** to zestaw roboczy (working set) silnika zmierzony na komputerze testowym (krok 10): od jednej rozmowy
  do czterech zaparkowanych, a w nim prawie cały plik modelu, bo silnik czyta z niego każdego eksperta spoza VRAM
  podczas pisania. GB to 10⁹ bajtów.
- **VRAM** to szacunek setupu, ile preset zajmuje przed jakimkolwiek ekspertem przy kontekście 32K (zmierzony na RTX
  5090), a potem liczba ekspertów, która mieści się na testowej karcie 6 GB na starcie.
- **UD-Q3_K_M nie jest szybszy od UD-Q4_K_M w FastStrata**: 66,2 zamiast 71,8 tok/s po 2K, 58,6 zamiast 66,9 po 37K,
  45,6 zamiast 48,8 po 139K. Jego eksperci IQ3_XXS nie dostają ani `--cpu-multi`, ani `--graph-experts`, a procesor
  liczy każdego z nich tak samo długo albo dłużej. Szybciej czyta prompty. Jest dla komputera z około 5 GB mniej RAM.
- **APEX-I-COMPACT nie jest polecany do programowania.** Kosztuje 4,3 raza tyle, co UD-Q3_K_M (raz na 50 tokenów
  zmienia się token, którego większy plik był pewny), a wśród tych zmian są nazwy w kodzie (`fake_ctypes.c` ->
  `fake_ctypes.Win`, `.Struct` -> `.w`). Na tym samym tekście pisze x1,22 szybciej niż APEX-I-QUALITY.
- Ornith 1.5 ma architekturę Qwen3.6-35B-A3B z innymi wagami, a jego blok szkicu zgaduje trafnie rzadziej, więc mniej
  zyskuje na zgadywaniu. Oba modele czytają obrazy (około 1 GB więcej do pobrania).
- **Do Qwen3.8-Flash-Next użyj oryginalnej [Strata](https://github.com/Niko1221/Strata).**

## Komputer testowy

| | |
| --- | --- |
| Laptop | AMD Ryzen 7 5800H (Zen 3, 8 rdzeni, 16 wątków, AVX2), 64 GB DDR4-3200 w dwóch kanałach (42–43 GB/s odczytu przez 8 wątków) |
| Karta graficzna | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB GDDR6 (336 GB/s), na PCIe 3.0 x8 |
| Dyski | SSD NVMe; plik każdego modelu był wczytany do pamięci podręcznej plików przed jego przebiegami |
| System | Windows 11 Pro, CUDA 13.0, silnik zbudowany dla sm_86 |
| Silnik | stary: wersja z 17.10.2026 z `--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6 --conversation-cache-mib 8192`, 16 wątków; FastStrata: kv-q8 w 150beb0b z flagami, które setup daje temu laptopowi ([Konfiguracja](#konfiguracja)) |

## Wyniki

Stary silnik kontra FastStrata na komputerze testowym, 20.10.2026, odpowiedzi zachłanne (greedy), silnik sam, tak jak
steruje nim serwer. Każda liczba to średnia z dwóch przebiegów, prompt 139K po jednym przebiegu na silnik. *Ten sam
tekst*: każdy silnik pisze te same odpowiedzi referencyjne do pięciu promptów agenta po 1,3K–3,7K tokenów, więc to, jak
często zgadywanie trafia, nie wpływa już na szybkość; to najstabilniejsza liczba. *Benchmark agenta*: sesja w stylu
OpenCode z 35 odpowiedziami (agent edytuje trzy pliki, subagent je przegląda, to samo zadanie w nowej rozmowie i zwykły
czat), prompty do 14 268 tokenów.

| Pisanie, tok/s, stary -> FastStrata | Po 2K | Ten sam tekst | Po 37K | Po 139K | Benchmark agenta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-Q4_K_M | 56,4 -> **71,8** (x1,27) | 62,3 -> **72,9** (x1,17) | 41,9 -> **66,9** (x1,60) | 24,7 -> **48,8** (x1,98) | 62,4 -> **82,8** (x1,33) |
| Qwen3.6 UD-Q3_K_M | 58,8 -> **66,2** (x1,13) | 67,3 -> **69,4** (x1,03) | 37,0 -> **58,6** (x1,58) | 26,4 -> **45,6** (x1,73) | 68,4 -> **81,2** (x1,19) |
| Ornith 1.5 APEX-I-QUALITY | 45,8 -> **51,0** (x1,11) | 46,4 -> **52,4** (x1,13) | 28,9 -> **39,6** (x1,37) | 21,5 -> **36,8** (x1,71) | 47,0 -> **61,8** (x1,32) |
| Ornith 1.5 APEX-I-COMPACT | 60,8 -> **63,1** (x1,04) | 57,9 -> **63,6** (x1,10) | 41,8 -> **56,2** (x1,34) | 24,6 -> **43,6** (x1,77) | 60,4 -> **76,1** (x1,26) |

Im dłuższy prompt, tym więcej daje FastStrata. Po krótkich promptach najwięcej zyskuje UD-Q4_K_M, jedyny plik, do
którego sięga każdy krok. Każdy plik zmierzono na jego własnym tekście i pliki nie są tu porównywane między sobą.

### Qwen3.6-35B-A3B UD-Q4_K_M (22,7 GB)

| | Stary | FastStrata | |
| --- | ---: | ---: | ---: |
| Pisanie po prompcie 2K | 56,4 tok/s | 71,8 tok/s | x1,27 |
| Pisanie tego samego tekstu | 62,3 tok/s | 72,9 tok/s | x1,17 |
| Pisanie po prompcie 37K | 41,9 tok/s | 66,9 tok/s | x1,60 |
| Pisanie po prompcie 139K | 24,7 tok/s | 48,8 tok/s | x1,98 |
| Czytanie promptu 37K | 47,2 s | 36,4 s | x1,30 |
| Czytanie promptu 139K | 265,1 s | 248,2 s | x1,07 |
| Benchmark agenta, wszystkie 35 odpowiedzi | 62,4 tok/s | 82,8 tok/s | x1,33 |
| To samo zadanie ponownie w nowej rozmowie | 65,3 tok/s | 93,6 tok/s | x1,43 |
| Karta 4 GB, kontekst 32K, `--kv-q8` | nie uruchamiano | 55,0 tok/s (jeden z dwóch przebiegów przerwany) | |
| RAM w użyciu podczas pracy | 24,1–27,2 GB | 24,1–27,2 GB | |

### Qwen3.6-35B-A3B UD-Q3_K_M (17,1 GB)

| | Stary | FastStrata | |
| --- | ---: | ---: | ---: |
| Pisanie po prompcie 2K | 58,8 tok/s | 66,2 tok/s | x1,13 |
| Pisanie tego samego tekstu | 67,3 tok/s | 69,4 tok/s | x1,03 |
| Pisanie po prompcie 37K | 37,0 tok/s | 58,6 tok/s | x1,58 |
| Pisanie po prompcie 139K | 26,4 tok/s | 45,6 tok/s | x1,73 |
| Czytanie promptu 37K | 45,7 s | 34,8 s | x1,31 |
| Czytanie promptu 139K | 258,8 s | 241,2 s | x1,07 |
| Benchmark agenta, wszystkie 35 odpowiedzi | 68,4 tok/s | 81,2 tok/s | x1,19 |
| To samo zadanie ponownie w nowej rozmowie | 72,8 tok/s | 91,5 tok/s | x1,26 |
| Karta 4 GB, 32K, `--kv-q8` | 35,3 tok/s | 44,7 tok/s | x1,27 |
| Karta 4 GB, 16K / 48K, `--kv-q8` | nie uruchamiano | 54,9 / 42,0 tok/s | |
| Odpowiedzi względem UD-Q4_K_M | | średnia KL 0,020, najbardziej prawdopodobny token zmieniony w 4,1% kroków, 5 pewnych w 1500 (własny tail skip UD-Q4_K_M: 0,006, 1,4%, żaden) | |
| RAM w użyciu podczas pracy | 18,5–21,7 GB | 18,5–21,7 GB | |

### Ornith 1.5 35B-A3B APEX-I-QUALITY (23,7 GB), bez tail skip

| | Stary | FastStrata | |
| --- | ---: | ---: | ---: |
| Pisanie po prompcie 2K | 45,8 tok/s | 51,0 tok/s | x1,11 |
| Pisanie tego samego tekstu | 46,4 tok/s | 52,4 tok/s | x1,13 |
| Pisanie po prompcie 37K | 28,9 tok/s | 39,6 tok/s | x1,37 |
| Pisanie po prompcie 139K | 21,5 tok/s | 36,8 tok/s | x1,71 |
| Czytanie promptu 37K | 47,7 s | 36,7 s | x1,30 |
| Czytanie promptu 139K | 269,2 s | 236,9 s | x1,14 |
| Benchmark agenta, wszystkie 35 odpowiedzi | 47,0 tok/s | 61,8 tok/s | x1,32 |
| To samo zadanie ponownie w nowej rozmowie | 47,9 tok/s | 73,6 tok/s | x1,54 |
| Karta 4 GB | nie uruchamiano | nie uruchamiano | |
| RAM w użyciu podczas pracy | 25,5–28,6 GB | 25,5–28,6 GB | |

### Ornith 1.5 35B-A3B APEX-I-COMPACT (17,4 GB), bez tail skip

| | Stary | FastStrata | |
| --- | ---: | ---: | ---: |
| Pisanie po prompcie 2K | 60,8 tok/s | 63,1 tok/s | x1,04 |
| Pisanie tego samego tekstu | 57,9 tok/s | 63,6 tok/s | x1,10 |
| Pisanie po prompcie 37K | 41,8 tok/s | 56,2 tok/s | x1,34 |
| Pisanie po prompcie 139K | 24,6 tok/s | 43,6 tok/s | x1,77 |
| Czytanie promptu 37K | 46,5 s | 35,2 s | x1,32 |
| Czytanie promptu 139K | 268,6 s | 224,6 s | x1,20 |
| Benchmark agenta, wszystkie 35 odpowiedzi | 60,4 tok/s | 76,1 tok/s | x1,26 |
| To samo zadanie ponownie w nowej rozmowie | 63,5 tok/s | 89,4 tok/s | x1,41 |
| Karta 4 GB, 32K, `--kv-q8` | nie uruchamiano | 47,8 tok/s | |
| Odpowiedzi względem APEX-I-QUALITY | | średnia KL 0,088, najbardziej prawdopodobny token zmieniony w 7,7% kroków, 26–27 pewnych w 1320 (APEX-I-QUALITY uruchomiony ponownie: 0,010, 2,0%, 1–4) | |
| RAM w użyciu podczas pracy | 19,2–22,4 GB | 19,2–22,4 GB | |

Czego te liczby nie obejmują: benchmarku zadań (HumanEval, odsetka udanych wywołań narzędzi), włączonego myślenia
(thinking), odpowiedzi losowanych (sampling), drugiego przebiegu przy 139K (drugi przebieg tej samej wersji różnił się
tam o kilka procent), serwera i aplikacji agenta przed silnikiem, innych kart i procesorów, Linuksa.

## Na karcie 4 GB do gier

Nie było pod ręką karty 4 GB. Karta 4 GB w tej sekcji to **karta 6 GB komputera testowego, której 2 GiB VRAM zajmuje
inny proces** (`test_scripts/vram_hold.py --card 4096`), co zostawiło silnikowi 3082 MiB wolnej pamięci, mniej więcej
tyle, ile zostawia karta 4 GB, gdy działa na niej pulpit. Pokazuje to, co mieści się w tej VRAM. Nie pokazuje
przepustowości pamięci prawdziwej karty 4 GB, jej łącza PCIe (tu 3.0 x8) ani tego, ile z niej zajmuje konkretny pulpit.

FastStrata, prompt wypełniający kontekst, potem 256 napisanych tokenów:

| Preset | Kontekst | `--kv-q8` | `--kv-q4` | Prompt przeczytany w |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-Q3_K_M | 16K | 54,9 tok/s | 56,0 tok/s | 17,8 s |
| | 32K | 39,6, 49,8 tok/s (dwa przebiegi) | 48,0, 48,4 tok/s | 37,3–37,5 s |
| | 48K | 42,0 tok/s | 47,9 tok/s | 60,2–60,5 s |
| Qwen3.6 UD-Q4_K_M | 32K | 55,0 tok/s; 53,5 bez `--graph-experts` | | 39,2–39,3 s |
| Ornith 1.5 APEX-I-COMPACT | 32K | 47,8 tok/s | | 39,7 s |

- **Każdy przebieg się zmieścił i nic nie trafiło do pamięci systemowej.** Żaden plik Qwen3.6 nie zostawia miejsca na
  choćby jednego eksperta w VRAM obok gęstych wag i pamięci podręcznych, więc wszystkich liczy procesor; mniejsze gęste
  wagi APEX-I-COMPACT zostawiają mu na starcie 398 ekspertów.
- **Stary -> nowy przy 32K z UD-Q3_K_M: 35,3 -> 44,7 tok/s, x1,27**; prompt w 39,1 -> 37,4 s.
- **UD-Q4_K_M pisze tam szybciej niż UD-Q3_K_M**, 55,0 zamiast 44,7 tok/s przy 32K: gdy wszyscy eksperci są na
  procesorze, eksperci Q4_K dostają `--cpu-multi`, a IQ3_XXS nie. Potrzebuje jednak 24–27 GB RAM zamiast 19–22.
- **`--kv-q8` mieści 48K**, więc do tej długości pamięć 4-bitowa nie jest potrzebna. Powyżej 48K na symulowanej karcie
  nie uruchamiano; przy 64K (krok 9, UD-Q4_K_M) pamięć karty trafiła do pamięci systemowej, a pisanie spadło do
  19,1 tok/s z `--kv-q8` i 31,0 z `--kv-q4`.
- **Jeden przebieg UD-Q4_K_M się zatrzymał** z komunikatem "graph experts: the card gave up waiting for the host";
  drugi napisał 55,0 tok/s, a przebieg bez `--graph-experts` 53,5. Przyczyny nie znaleziono, więc setup nie daje
  `--graph-experts` karcie poniżej 5 GB.
- **RAM w komputerze z 32 GB:** silnik z plikiem 17 GB (UD-Q3_K_M, APEX-I-COMPACT) zajmuje podczas pracy 19–22 GB RAM,
  co zostawia komputerowi z 32 GB (34,4 GB) 12–15 GB na Windows i inne programy; Windows i pulpit komputera testowego
  zajmowały tego dnia 15,6–20,5 GB, więc potrzebny jest lżejszy pulpit. Pliki 23 GB zajmują 24–29 GB i zostawiają
  komputerowi z 32 GB za mało. Nie mierzono na komputerze z 32 GB.
- **Setup na karcie 4 GB** proponuje te presety z kontekstami zmierzonymi tutaj i pamięcią 8-bitową: UD-Q3_K_M do
  48K, UD-Q4_K_M i APEX-I-COMPACT przy 32K. Jego własny szacunek pamięci, robiony dla większych kart, uznaje kartę 4 GB
  za za małą na jakikolwiek kontekst, więc poniżej 5,5 GB setup kieruje się tymi pomiarami.

## Czego potrzebujesz

### Jaka karta

| VRAM | Co daje | Skąd to wiadomo |
| --- | --- | --- |
| **4 GB: minimum** | Krótki kontekst. UD-Q3_K_M działa z 16K do 48K, UD-Q4_K_M i APEX-I-COMPACT z 32K, wszystkie z pamięcią 8-bitową, pisząc 42–56 tok/s. 16K to najmniejszy kontekst, jaki proponuje setup; 32K to najmniej, z czym agent programistyczny taki jak OpenCode pracuje wygodnie. | Zmierzone na symulowanej karcie 4 GB (3082 MiB wolne), [wyżej](#na-karcie-4-gb-do-gier) |
| **6 GB: rozsądnie** | **Q4 (UD-Q4_K_M) z pamięcią 8-bitową przy 144K**: każdy wynik na tej stronie. | Zmierzone: [komputer testowy](#komputer-testowy) |
| **8 GB: optymalnie** | Q4 z pamięcią 8-bitową przy około 248K (Ornith całe okno 256K) albo 144K z pamięcią 16-bitową, i więcej ekspertów w VRAM. | Szacunek setupu pomniejszony o 10%; nie mierzono z FastStrata (PolyStrata uruchamiała te presety na laptopowej karcie 8 GB) |

Ustawienie do wybrania: **Q4 i pamięć 8-bitowa** (`--kv-q8`). Setup poleca je na karcie 6 lub 8 GB i komputerze z
około 25 GB RAM wolnymi dla niego; na karcie 4 GB poleca mniejszy Q3.

| | |
| --- | --- |
| **Karta graficzna** | Jedna karta NVIDIA GeForce z serii RTX 20 lub nowszej. Zmierzono na laptopowej karcie 6 GB i na symulowanych na niej 4 GB. PolyStrata uruchamiała te presety na laptopowej karcie 8 GB i na karcie 32 GB bez opcji FastStrata ([docs/BACKENDS.md](BACKENDS.md#benchmark)). |
| **RAM** | 24–29 GB w użyciu dla plików 23 GB i 19–22 GB dla plików 17 GB, na karcie 6 GB (wyżej). Karta z większą VRAM potrzebuje mniej. |
| **Procesor** | x86-64 z AVX2 dla `--cpu-multi`; mierzono na 8 rdzeniach. |
| **Dysk** | 17–24 GB na plik modelu, 1 GB więcej na czytanie obrazów. SSD: eksperci są czytani z pliku. |
| **System** | Windows 10 / 11 lub Linux, aktualny sterownik NVIDIA. |

Co menu setupu mówi o każdym presecie, według karty: najdłuższy kontekst, jaki proponuje z pamięcią 16- lub
8-bitową, i RAM, którego chce (w GiB, tak jak podaje go system; komputer „32 GB” podaje trochę mniej niż 32). Wiersze
6 GB i 4 GB zmierzono przy całej wolnej karcie; pozostałe, i karta, której część VRAM zajmuje pulpit, dostają szacunek
setupu pomniejszony o 10%, a menu pisze „estimated”. Zgłoszenia z innych komputerów zastąpią szacunki:

| Karta | Qwen3.6-35B-A3B | Ornith 1.5 |
| --- | --- | --- |
| 4 GB | UD-Q4_K_M 32K od 26,1 GB RAM, UD-Q3_K_M 48K od 20,6 | APEX-I-QUALITY 24K (szacunek) od 27,6, APEX-I-COMPACT 32K od 21,8 |
| 6 GB | UD-Q4_K_M 144K od 24,7, UD-Q3_K_M 144K od 19,3 | APEX-I-QUALITY 144K od 25,9, APEX-I-COMPACT 144K od 19,9 |
| 8 GB | UD-Q4_K_M 248K od 22,6, UD-Q3_K_M 240K od 17,2 | APEX-I-QUALITY 256K od 23,7, APEX-I-COMPACT 256K od 17,7 |
| 12 GB | UD-Q4_K_M od 18,3, UD-Q3_K_M od 12,9 | APEX-I-QUALITY od 19,4, APEX-I-COMPACT od 13,4 |
| 16 GB | UD-Q4_K_M od 14,0 | APEX-I-QUALITY od 15,1 |

Setup poleca pierwszy preset z menu, który działa na tym komputerze z 32K lub więcej, i najdłuższy kontekst do 144K:
dłuższy czyta pełny prompt przez minuty (139K w około cztery na komputerze testowym).

Karta AMD na Linuksie dostaje opcje działające na procesorze (`--threads`, `--suffix-drafts`, `--cpu-multi`, pamięć
rozmów); pozostałe to kod CUDA, niezbudowany dla AMD. Nie mierzono z FastStrata. Nie działa na Macu.

## Konfiguracja

Setup zapisuje opcje silnika w konfiguracji uruchomienia modelu, `strata-<rodzina>-<preset>.json` (na przykład
`strata-qwen36-ud-q4_k_m.json`), w `"args"`. Na komputerze testowym, dla UD-Q4_K_M przy 144K:

```text
--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6 --threads 8 --attn-split --graph-experts
--elastic-prefill 1024 --suffix-drafts 256 --cpu-multi --conversation-cache-mib 12288 --conversation-cache-slots 8
```

Każdą tam, gdzie komputer i preset ją przyjmują; setup wypisuje każdą razem z tym, co dała:

| Opcja | Komu setup ją daje |
| --- | --- |
| `--threads N` | procesorowi o jednakowych rdzeniach z dwoma wątkami na rdzeń: jeden wątek na rdzeń (hybrydowy procesor Intel zachowuje własną regułę) |
| `--attn-split` | NVIDIA od Turinga (RTX 20) wzwyż |
| `--graph-experts` | jednej karcie NVIDIA od Volty wzwyż z co najmniej 5 GB VRAM; nie UD-Q3_K_M, którego ekspertów nie czyta |
| `--elastic-prefill 1024` | jednej karcie NVIDIA |
| `--suffix-drafts 256` | komputerowi z RAM, którego według szacunku setupu preset potrzebuje |
| `--cpu-multi` | procesorowi z AVX2 i presetowi z ekspertami K-quant (nie UD-Q3_K_M) |
| `--kv-q8`, `--prompt-batch 512` | NVIDIA, gdy żądany kontekst nie mieści się na karcie w 16 bitach; setup pyta |
| `--kv-q4`, potem `--kv-q4-keys q4` | NVIDIA, tylko gdy nie mieści się nawet `--kv-q8` z partiami po 512; setup podaje, ile to kosztuje, i pyta |
| pamięć rozmów 12 GiB, 8 miejsc | komputerowi z 24 GB RAM ponad to, czego chce preset |

- **Wszystkie przyspieszenia wyłączone:** `START-HERE.bat --setup --speedups off` (`./setup.sh --setup --speedups off`)
  zapisuje konfigurację tak, jak robił to setup PolyStrata, a kolejne uruchomienia setupu zachowują ten wybór;
  `--speedups on` włącza je z powrotem. Pojedyncza opcja: usuń ją z `"args"`; kolejne uruchomienie z `--setup` zapisze
  ją ponownie.
- **Długi kontekst na małej karcie:** gdy `--kv-q8` się nie mieści, zamień `"--kv-q8"` na `"--kv-q4"` (lub
  `"--kv-q4-keys", "q4"`) i zwiększ `--max-context`. To kosztuje odpowiedzi ([krok 9](OPTIMIZATION_LOG.md)),
  więc używaj tego tylko dla kontekstu, który inaczej się nie mieści.
- **Czego setup nigdy nie włącza:** `--weight-cut`, `--defer-experts`, głębszego tail skip
  ([wyżej](#zmierzone-i-odrzucone)).
- Każda opcja szczegółowo, z własnymi pomiarami (po angielsku): [docs/FASTSTRATA_OPTIONS.md](FASTSTRATA_OPTIONS.md).
  Regułę zawiera `poly_speedups()` w [setup.py](../setup.py), testy `tools/test_setup_speedups.py`.

## Jak to zmierzono

- [docs/OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md): każdy krok, co zmierzono, jak, względem czego i czego nie
  zmierzono.
- Skrypty, które wymienia log (`test_scripts/`: `run_serve.py`, sam silnik; `agent_bench.py`, sesja agenta;
  `quality.py`, ile ustawienie kosztuje odpowiedzi; `fit_run.py`, jak długi kontekst się mieści; `vram_hold.py`,
  symulacja mniejszej karty; `proc_mem.py`, RAM silnika; `faststrata_runs.py`, przebiegi kroku 10), należą do
  repozytorium roboczego i nie są tu publikowane; zapytaj w zgłoszeniu o ten, który chcesz uruchomić.
- Stary i nowy silnik na zmianę, świeża kopia profilu ekspertów w każdym przebiegu, nic innego na karcie, odpowiedzi
  zachłanne.
- Raport z twojego komputera jest mile widziany: `BENCH.bat` (`./bench.sh`) mierzy czat, długie prompty i sesję w
  stylu agenta ([jak](COMMUNITY_BENCHMARKS.md#run-the-benchmark)). Napisz, czego nie testowałeś.
