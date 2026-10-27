<h1 align="center">FastStrata</h1>

<p align="center"><b>Model 35B typu mixture-of-experts do programowania w OpenCode na komputerze do gier z małą kartą graficzną</b><br>
Qwen3.6-35B-A3B i Ornith 1.5 · własna bramka dla OpenCode · Windows i Linux · NVIDIA · MIT</p>

<p align="center">
<a href="https://github.com/Jarson1408/FastStrata/actions/workflows/setup-tests.yml"><img src="https://github.com/Jarson1408/FastStrata/actions/workflows/setup-tests.yml/badge.svg" alt="testy setupu"></a>
<a href="https://github.com/Jarson1408/FastStrata/releases"><img src="https://img.shields.io/github/v/release/Jarson1408/FastStrata?label=release" alt="wydanie"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="licencja MIT"></a>
</p>

<p align="center"><a href="README.md">English</a> · <b>Polski</b></p>

<!-- Demo: OpenCode piszący kod na laptopie 6 GB, obok licznik prędkości. Gdy docs/media/faststrata-opencode-6gb.gif
będzie w repozytorium, wstaw tu tę linię, i w README.md:
<p align="center"><img src="docs/media/faststrata-opencode-6gb.gif" width="900" alt="OpenCode piszący kod z Qwen3.6-35B-A3B na laptopie z RTX 3060 6 GB, obok licznik prędkości"></p>
-->

FastStrata uruchamia Qwen3.6-35B-A3B, model do programowania z 35 miliardami wag, z których na każdy token pracują 3
miliardy, na laptopie do gier z kartą graficzną 6 GB. Na RTX 3060 Laptop z 64 GB RAM pisze **72 tok/s po krótkim
prompcie, 67 po prompcie 37K tokenów i 49 po 139K**, z kontekstem 144K. Karta trzyma warstwy uwagi (attention) i
ekspertów, których odpowiedź używa najczęściej; resztę liczy procesor z RAM.

**Zrobiona dla [OpenCode](https://opencode.ai), z własną bramką.** [FastStrata Gateway](https://github.com/Jarson1408/FastStrata-Gateway) jest częścią tego
projektu: to aplikacja między OpenCode a modelem, napisana pod OpenCode, którą setup instaluje razem z modelem. Setup
znajduje też OpenCode (albo go instaluje), dodaje do jego konfiguracji dostawcę **FastStrata** i tworzy na pulpicie
skrót **FastStrata**: jedno kliknięcie uruchamia bramkę, a bramka uruchamia model, gdy OpenCode o niego prosi.
Zamienia wywołania narzędzi modelu na takie, które OpenCode czyta, przerywa turę, która się zapętla, przełącza
zainstalowane modele i potrafi trzymać mały model przy pamięci projektu
([LODESTAR](#lodestar-pamięć-projektu-dla-małego-modelu)). [Więcej](#opencode-i-faststrata-gateway)

To fork [PolyStrata](https://github.com/VecSzn/PolyStrata), który na tym samym laptopie działa do dwóch razy
szybciej: x1,27 po krótkim prompcie, x1,60 po 37K, x1,98 po 139K i x1,33 w sesji agenta programistycznego. Każdą liczbę
zmierzono na tym jednym laptopie, uruchamiając stary i nowy silnik na zmianę: [wyniki](docs/RESULTS.pl.md),
[jak mierzono](docs/OPTIMIZATION_LOG.md) (po angielsku).

**Spis treści:** [Czy mój komputer da radę?](#czy-mój-komputer-da-radę) · [Instalacja w trzech
krokach](#instalacja-w-trzech-krokach) · [Szybkość](#szybkość) · [Używanie](#używanie) ·
[OpenCode i bramka](#opencode-i-faststrata-gateway) · [LODESTAR](#lodestar-pamięć-projektu-dla-małego-modelu) ·
[Gdy setup się zatrzyma](#gdy-setup-się-zatrzyma) · [Odinstalowanie](#odinstalowanie) ·
[Konfiguracja](#konfiguracja) · [Skąd pochodzi](#skąd-pochodzi) · [Podziękowania i licencja](#podziękowania-i-licencja)

## Czy mój komputer da radę?

Jedna karta NVIDIA, GeForce z serii RTX 20 lub nowsza:

| VRAM | Co daje | Skąd to wiadomo |
| --- | --- | --- |
| **4 GB: minimum** | Krótki kontekst, 16K do 48K z pamięcią 8-bitową, pisanie z prędkością 42–56 tok/s. 32K to najmniej, z czym agent programistyczny taki jak OpenCode pracuje wygodnie. Który plik i jaki kontekst: [niżej](#karta-4-gb). | Zmierzone na symulowanej karcie 4 GB: karcie 6 GB, której 2 GiB zajmuje inny program |
| **6 GB: rozsądnie** | **Kontekst 144K** z modelem Q4 i pamięcią 8-bitową, pisanie z prędkością 49–72 tok/s. | Zmierzone: [laptop testowy](docs/RESULTS.pl.md#komputer-testowy) |
| **8 GB: optymalnie** | Około 248K z modelem Q4 i pamięcią 8-bitową (144K z 16-bitową) i więcej ekspertów w VRAM. | Szacunek setupu pomniejszony o 10%; nie mierzono z FastStrata |

**Q4 i pamięć 8-bitowa to dwie różne rzeczy.** *Q4* (albo *Q3*) to **plik modelu**: ile bitów zajmują jego wagi
(UD-Q4_K_M 23 GB, UD-Q3_K_M 17 GB); od tego zależy, ile RAM potrzebuje i jak jest dokładny. **Pamięć** to **pamięć
kontekstu**, pamięć uwagi (attention cache), która trzyma rozmowę: w **8 bitach** (`--kv-q8`) token zajmuje 11,7 KB
zamiast 22 KB przy 16 bitach i dzięki temu 144K mieści się na karcie 6 GB. Setup dobiera pamięć do karty: 16 bitów,
gdzie się mieszczą, 8 tam, gdzie nie, a 4 bity (`--kv-q4`, mniej dokładne) tylko tam, gdzie nie mieści się nawet 8.

| | |
| --- | --- |
| **RAM** | Podczas pracy silnik zajmuje 24–29 GB z plikami modelu 23 GB i 19–22 GB z plikami 17 GB (zmierzone na laptopie 6 GB, który ma 64 GB). Setup sprawdza, ile Twój komputer ma wolnego. Komputera z 32 GB nie mierzono. |
| **Dysk** | 17–24 GB na model, na SSD: eksperci są czytani z pliku. |
| **Procesor** | x86-64, z AVX2 dla `--cpu-multi`; mierzono na 8 rdzeniach. |
| **System** | Windows 10 / 11 lub Linux, aktualny sterownik NVIDIA. Karta AMD: [zobacz niżej](#karty-amd). Nie działa na Macu. |

Setup sam to wszystko sprawdza i mówi, co się mieści, zanim cokolwiek pobierze: `START-HERE.bat --check` tylko
pokazuje wynik. Pełna tabela według karty i presetu: [docs/RESULTS.pl.md](docs/RESULTS.pl.md#czego-potrzebujesz).

### Karta 4 GB

Działa, z krótkim kontekstem. Karty 4 GB nie było pod ręką: te przebiegi to karta 6 GB laptopa testowego, z której
inny program zajął 2 GiB (`test_scripts/vram_hold.py --card 4096`), co zostawiło silnikowi 3082 MiB, mniej więcej
tyle, ile zostaje na karcie 4 GB, gdy działa na niej pulpit. Prompt wypełniający kontekst, potem 256 napisanych
tokenów ([wszystkie przebiegi](docs/RESULTS.pl.md#na-karcie-4-gb-do-gier)):

| Plik modelu | Pamięć kontekstu | 16K | 32K | 48K | 64K |
| --- | --- | --- | --- | --- | --- |
| **Q4**: Qwen3.6 UD-Q4_K_M, 22,7 GB | **8 bitów** | | **mieści się, 50–55 tok/s** | nie uruchamiano | za duży: 19 tok/s |
| | 4 bity | | mieści się, 55 tok/s | nie uruchamiano | za duży: 31 tok/s |
| | 16 bitów | | jedyny przebieg się zatrzymał | | |
| **Q3**: Qwen3.6 UD-Q3_K_M, 17,1 GB | **8 bitów** | mieści się, 55 tok/s | mieści się, 40–50 tok/s | **mieści się, 42 tok/s** | nie uruchamiano |
| | 4 bity | mieści się, 56 tok/s | mieści się, 48 tok/s | mieści się, 48 tok/s | nie uruchamiano |
| Ornith 1.5 APEX-I-COMPACT, 17,4 GB | 8 bitów | | mieści się, 48 tok/s | nie uruchamiano | nie uruchamiano |

„Za duży” znaczy, że kontekst nie zmieścił się na karcie: Windows przeniósł pamięć karty do pamięci systemowej
i pisanie zwolniło kilkukrotnie. Prompt 32K był czytany od początku w 37–40 s.

| Twój komputer | Weź |
| --- | --- |
| **32 GB RAM** | **Q3** (UD-Q3_K_M) z pamięcią 8-bitową, 32K; do 48K na dłuższe sesje |
| **48 GB RAM lub więcej** | **Q4** (UD-Q4_K_M) z pamięcią 8-bitową, 32K; Q3 tam, gdzie chcesz do 48K |

- **Który plik.** Q4, jeśli komputer ma na niego RAM. Na karcie 4 GB żaden ekspert żadnego z plików Qwena nie mieści
  się w VRAM, więc liczy je wszystkie procesor, a eksperci Q4 idą szybszą ścieżką CPU (`--cpu-multi`), której eksperci
  Q3 nie mają: przy 32K Q4 pisał 55 tok/s, Q3 45. Q4 jest też dokładniejszy (Q3 względem niego zmienił najbardziej
  prawdopodobny token w 4,1% kroków, średnia KL 0,020). Ale w trakcie pracy Q4 zajmuje 24–27 GB RAM, a Q3 19–22 GB:
  na komputerze z 32 GB Q4 zostawia Windowsowi i innym programom za mało.
- **Która pamięć kontekstu.** 8 bitów (`--kv-q8`). Mieści 32K z Q4 i 48K z Q3, a z małych formatów najmniej kosztuje
  odpowiedzi: względem 16 bitów najbardziej prawdopodobny token zmienił się w 1,2–2,3% kroków. Pamięć 4-bitowa
  (`--kv-q4`) pisała tu odrobinę szybciej, ale kosztuje więcej (1,9–3,3%) i w tym, co uruchomiono, nie zmieściła
  dłuższego kontekstu: jest na kontekst, którego 8 bitów nie pomieści. Pamięć 16-bitowa nie jest na kartę 4 GB.
- **Jaki kontekst.** 32K, najmniej, przy którym agent do kodu taki jak OpenCode pracuje wygodnie: Q4 przy 32K albo Q3
  do 48K. Nie ustawiaj więcej: ponad to, co się mieści, pisanie zwalnia kilkukrotnie. Między 48K a 64K niczego nie
  uruchamiano.
- **Co robi setup.** Na karcie z 3 do 4,5 GB wolnej pamięci menu proponuje Q4 przy 32K (od 26,1 GB wolnego RAM), Q3
  przy 48K (od 20,6), Ornith APEX-I-COMPACT przy 32K (od 21,8) i APEX-I-QUALITY przy 24K (szacunek, od 27,6), wszystkie
  z pamięcią 8-bitową, i poleca pierwszy, który uruchamia 32K lub więcej i mieści się w Twoim RAM. Karta poniżej 5 GB
  nie dostaje `--graph-experts` (z nim jeden przebieg z dwóch się tam zatrzymał, a dawał około 3%) ani `--stage-next`
  (jej cache prawie nie ma czego mu pożyczyć, a tam go nie uruchamiano).
- **Od v0.4.0:** `--lend-staging`, które setup dopisuje na jednej karcie NVIDIA, umieszcza na takiej karcie
  około jednej trzeciej ekspertów w VRAM, gdzie wcześniej nie mieścił się żaden: z Q4 i pamięcią 8-bitową 11–19%
  szybciej bez zgadywania ([krok 13](docs/OPTIMIZATION_LOG.md)). Mierzone z włączonym `--graph-experts`, którego setup
  tu nie daje.
- **Niezmierzone:** prawdziwa karta 4 GB (jej przepustowość pamięci, łącze PCIe, ile VRAM zajmuje jej pulpit; setup
  pyta, czy pulpit działa na karcie NVIDIA), konteksty między 48K a 64K, komputer z 32 GB RAM.

### Karty AMD

Przyspieszenia FastStrata są napisane dla kart NVIDIA RTX (CUDA) i większość z nich nie działa na karcie AMD. Z kartą
AMD zajrzyj do [PolyStrata](https://github.com/VecSzn/PolyStrata), projektu, z którego wywodzi się FastStrata. Co FastStrata robi na takiej karcie:

- **Windows:** Qwen3.6 i Ornith nie działają: silnik dla kart AMD (HIP) buduje się tylko na Linuksie. Setup mówi to,
  zanim cokolwiek pobierze.
- **Linux z ROCm** (z jego pakietami deweloperskimi): setup kompiluje silnik, bo gotowy silnik z wydania jest dla
  NVIDIA. Działają przyspieszenia po stronie procesora (`--threads`, `--cpu-multi`, `--suffix-drafts`, pamięć rozmów),
  nie działają te na karcie (`--kv-q8`, `--kv-q4`, `--attn-split`, `--graph-experts`, `--elastic-prefill`,
  `--lend-staging`, `--stage-next`). Pamięć uwagi zostaje 16-bitowa, 22 KB na token zamiast 11,7, więc ta sama ilość VRAM mieści
  mniej więcej połowę kontekstu.
- **Niezmierzone** na żadnej karcie AMD. Przeniesienie przyspieszeń karty na HIP albo pomiar z karty AMD są mile
  widziane: [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## Instalacja w trzech krokach

1. **Pobierz** [zip][zip] i rozpakuj go gdziekolwiek albo zrób `git clone` [repozytorium][repo].
2. **Uruchom setup:** na Windows kliknij dwukrotnie **`START-HERE.bat`**; na Linuksie uruchom w folderze
   **`./setup.sh`**. Naciskaj Enter, żeby przyjąć bezpieczny wybór.
3. **Używaj:** uruchom skrót **FastStrata**, otwórz OpenCode i wybierz model dostawcy **FastStrata** (setup
   zainstalował bramkę i wpisał konfigurację OpenCode) albo rozmawiaj w przeglądarce pod `http://127.0.0.1:8080`.

W kroku 2 setup najpierw pyta, czy karta NVIDIA wyświetla pulpit (czy jest do niej podłączony ekran); sam to wykrywa i
podaje jako odpowiedź. Karta, która wyświetla pulpit, trzyma na niego trochę VRAM, więc wtedy bezpieczny kontekst jest o
10% mniejszy od najdłuższego, który się mieści (128K zamiast 144K na karcie 6 GB). Potem pokazuje bezpieczny wybór dla
tego komputera, preset z kontekstem: Enter instaluje go bez dalszych pytań. Albo wybierz sam z czterech presetów
ocenionych dla tego komputera, z prędkością zmierzoną na laptopie testowym i z tym, ile każdy kosztuje odpowiedzi: jeden
albo kilka (`1,3`), każdy z własnym kontekstem i długością odpowiedzi (8K, 16K albo 32K; bezpieczna to własna długość
presetu, Qwen3.6 16K i Ornith 8K, a odpowiedź zajmuje najwyżej ćwierć kontekstu). Działają po jednym naraz (bramka
przełącza je, gdy OpenCode prosi o inny), więc każdy jest oceniany osobno; ich pobrania się sumują. Potem setup pobiera
gotowy silnik z najnowszego wydania FastStrata (155 MB: Windows, karta NVIDIA od RTX 20 do RTX 50 ze sterownikiem 580
lub nowszym, procesor z AVX2; sprawdzony z SHA-256, które podaje GitHub) albo, na każdym innym komputerze, go kompiluje
(około pięciu minut na testowym laptopie, jednorazowo; na Windows proponuje instalację kompilatora Visual Studio 2022 i
CUDA toolkit), pobiera modele (17–24 GB każdy, z możliwością wznowienia), zapisuje konfigurację każdego z opcjami, które
przyjmuje ten komputer, i proponuje [bramkę dla OpenCode](#opencode-i-faststrata-gateway). Na ekranie są krótkie zdania;
`--details` pokazuje też każdy pomiar i polecenie, a [log](#gdy-setup-się-zatrzyma) ma je zawsze.

**Następnym razem** otwórz skrót **FastStrata** na pulpicie (z bramką) albo uruchom ponownie `START-HERE.bat`:
uruchamia zainstalowany model i niczego nie pobiera dwa razy. Pierwszy start po uruchomieniu komputera czyta plik
modelu z SSD: na laptopie testowym silnik wczytał się w 9,9 s, a pierwsza odpowiedź przyszła z prędkością 27 tok/s.
Nowszy FastStrata rozpakowany obok zainstalowanego ustawia się przy pierwszym uruchomieniu tak samo: jego modele z ich
kontekstami i bramka (przejęta albo zaproponowana); pliki modeli są użyte ponownie.

Dla skryptu albo asystenta AI `--yes` wybiera każdą bezpieczną odpowiedź (bezpieczny preset i jego kontekst, pulpit taki, jak wykryto); pełne
polecenie setupu laptopa testowego to

```bat
START-HERE.bat --family qwen36 --model UD-Q4_K_M --context 147456 --yes
```

Kilka presetów naraz: `--models UD-Q4_K_M,APEX-I-QUALITY` (albo dwa razy `--model`), z `--context N` dla wszystkich
albo `--context UD-Q4_K_M=147456,APEX-I-QUALITY=65536` dla każdego z osobna; `--max-output N` (albo `MODEL=N`) tak samo
ustawia długość odpowiedzi (`./setup.sh` przyjmuje na Linuksie te same opcje). Albo wklej to do Claude Code, Cursor, Codex lub Copilot:

```text
Set up FastStrata on this PC for me: https://github.com/Jarson1408/FastStrata - follow docs/AI_SETUP.md there.
```

## Szybkość

Pisanie, tokeny na sekundę, na laptopie testowym (RTX 3060 Laptop 6 GB, Ryzen 7 5800H, 64 GB DDR4), 20.10.2026,
odpowiedzi zachłanne (greedy), silnik, od którego FastStrata wyszła, kontra FastStrata, każda liczba to średnia z
dwóch przebiegów (139K: z jednego):

| Preset | Po 2K | Po 37K | Po 139K | Benchmark agenta |
| --- | ---: | ---: | ---: | ---: |
| **Qwen3.6 UD-Q4_K_M** (22,7 GB, polecany) | 56,4 -> **71,8** | 41,9 -> **66,9** | 24,7 -> **48,8** | 62,4 -> **82,8** |
| Qwen3.6 UD-Q3_K_M (17,1 GB) | 58,8 -> **66,2** | 37,0 -> **58,6** | 26,4 -> **45,6** | 68,4 -> **81,2** |
| Ornith 1.5 APEX-I-QUALITY (23,7 GB) | 45,8 -> **51,0** | 28,9 -> **39,6** | 21,5 -> **36,8** | 47,0 -> **61,8** |
| Ornith 1.5 APEX-I-COMPACT (17,4 GB) | 60,8 -> **63,1** | 41,8 -> **56,2** | 24,6 -> **43,6** | 60,4 -> **76,1** |

- Benchmark agenta to sesja w stylu OpenCode z 35 odpowiedziami i promptami do 14 268 tokenów.
- UD-Q3_K_M nie jest szybszy od UD-Q4_K_M w FastStrata: jest dla komputera z około 5 GB mniej RAM.
- APEX-I-COMPACT nie jest polecany do programowania: zmienia nazwy w kodzie, które APEX-I-QUALITY pisze dobrze.
- Nie mierzono: benchmarku zadań, włączonego myślenia (thinking), odpowiedzi losowanych, innych kart i procesorów,
  Linuksa.

Co dała każda zmiana, każdy preset w całości, czasy czytania promptu i ile mniejsze pliki kosztują odpowiedzi:
[docs/RESULTS.pl.md](docs/RESULTS.pl.md).

## Używanie

- **W przeglądarce:** `http://127.0.0.1:8080`, z czatem (Chat), podglądem modelu i komputera na żywo (Monitor), a w
  nim prędkością pisania, oraz informacjami (About).
- **Aplikacje i agenci programistyczni:** dostawca zgodny z OpenAI pod `http://127.0.0.1:8080/v1` (dowolny klucz,
  dowolna nazwa modelu); API Anthropic pod `/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` i
  `"fit_max_tokens": true` w konfiguracji, [dlaczego](docs/FAQ.md#can-i-use-it-with-claude-code-codex-or-other-apps));
  Responses API pod `/v1/responses`; `/completion` i `/tokenize` w stylu llama.cpp.
- **Z innego urządzenia:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <sekret>`. Nigdy bez klucza.

## OpenCode i FastStrata Gateway

FastStrata jest zrobiona do pracy w [OpenCode](https://opencode.ai), a jej własna bramka sprawia, że to działa.
[FastStrata Gateway](https://github.com/Jarson1408/FastStrata-Gateway) to aplikacja między OpenCode a modelem, napisana pod OpenCode i zintegrowana z tym
projektem: setup instaluje ją tutaj w `gateway\`, przypiętą do jednego wydania i sprawdzoną przez SHA-256 jej plików.
Zamienia wywołania narzędzi modelu na takie, które OpenCode czyta, przerywa turę, która się zapętla, uruchamia model,
gdy OpenCode o niego prosi, i przełącza go na inny zainstalowany preset, oraz prowadzi pamięć projektu
[LODESTAR](#lodestar-pamięć-projektu-dla-małego-modelu) (na starcie wyłączoną, z przełącznikiem). Po modelu setup ją
proponuje; po odpowiedzi „tak” setup:

- umieszcza bramkę w `gateway\` w tym folderze, z własnym środowiskiem Pythona (około 300 MB);
- wpisuje zainstalowane presety do jej ustawień, każdy z kontekstem, który setup wybrał dla tego komputera;
- znajduje OpenCode tam, gdzie jest zainstalowany, albo proponuje jego instalację (przez npm albo scoop, zależnie od
  tego, co jest na komputerze);
- wpisuje presety do konfiguracji OpenCode (`~/.config/opencode/opencode.json`, dostawca `faststrata`), zachowując
  kopię `.bak` i nie ruszając reszty pliku; folder tworzy, gdy go nie ma;
- tworzy na pulpicie skrót **FastStrata**, który otwiera aplikację bramki; aplikacja uruchamia model, gdy OpenCode o
  niego prosi, a jej zamknięcie zwalnia kartę i RAM. Start modelu trwa około minuty na testowym laptopie 6 GB: pierwsza
  odpowiedź po otwarciu aplikacji albo po przełączeniu na inny model czeka na niego, a OpenCode czeka razem z nią. Od
  v0.6.0 rozmowy, które model przeczytał, wracają potem z dysku (`--conversation-disk`): na laptopie testowym następna
  tura rozmowy 70K po restarcie zaczęła się po 7,2 s zamiast 83,0.

W OpenCode wybierz model dostawcy FastStrata. Później albo na instalacji zrobionej bez niej: `START-HERE.bat --gateway
on` (`./setup.sh --gateway on`). Z `--yes` setup ją pomija, chyba że podasz `--gateway on`; `--opencode install|skip`
i `--no-shortcut` odpowiadają na resztę.

### LODESTAR: pamięć projektu dla małego modelu

Model z 3B aktywnych parametrów czyta pliki z poleceniami projektu (`README.md`, `AGENTS.md`), ale nie trzyma się ich
w długim zadaniu: pomija kroki i zapomina, co ustalono. LODESTAR to protokół bramki, który temu zapobiega. Po instalacji
jest wyłączony; pole **Enforce LODESTAR Memory Protocol** w aplikacji (albo *LODESTAR Protocol* w menu w zasobniku)
włącza go i wyłącza w każdej chwili.

- **Wdrożenie do projektu.** Przycisk **Deploy Template** w aplikacji (albo *Deploy LODESTAR Template...* w zasobniku)
  kopiuje `AGENTS.md`, `CLAUDE.md` i folder `.context/` do wybranego folderu projektu. Protokół działa w folderze, do
  którego go wdrożono.
- **Pamięć w sesji.** Z włączonym LODESTAR bramka trzyma model w jednym cyklu na zadanie: zanim zmieni plik, czyta
  zasady i pamięć projektu i zapisuje cel oraz kroki w `.context/memory/active-context.md`; na końcu zapisuje, co
  zrobiono, i zamyka cel. Przerwana sesja albo nowa zaczyna tam, gdzie skończyła poprzednia.
- **Pamięć semantyczna dla długich projektów.** Lekcja z każdego skończonego zadania trafia do
  `.context/semantic-memory/` i jest wyszukiwana na początku następnego. Działa przez
  `python .context/semantic-memory/embed.py`, który potrzebuje `sentence-transformers` w tym Pythonie
  (`pip install sentence-transformers`; pobiera PyTorch). Własne środowisko bramki go nie instaluje.

## Gdy setup się zatrzyma

Każde uruchomienie zapisuje to, co setup wypisał, do `logs\setup-<data>-<godzina>.log` (zostaje dziesięć najnowszych),
bez nazwy Twojego konta i komputera i bez kluczy API. Tam, gdzie setup się zatrzymuje, podaje ten plik: dołącz go do
[zgłoszenia][issues] i napisz, co robiłeś. Więcej (po angielsku): [docs/INSTALL.md](docs/INSTALL.md),
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Odinstalowanie

Najpierw zamknij aplikację FastStrata (ikona w zasobniku > Quit). Potem kliknij dwukrotnie **`UNINSTALL.bat`** w
folderze FastStrata (Linux: `python3 setup.py --uninstall`). Wypisuje, co zrobi, i pyta raz:

- ustawienia bramki, `%APPDATA%\faststrata-gateway` (Linux: `~/.config/faststrata-gateway`), trafiają do Kosza;
- skrót **FastStrata** z pulpitu trafia do Kosza;
- z konfiguracji OpenCode znika dostawca `faststrata` i nic więcej; plik sprzed zmiany zostaje obok jako `.bak` (albo
  `.bak-<data>-<godzina>`, gdy `.bak` już tam jest).

To, co ustawił inny folder FastStrata na tym komputerze, zostaje, chyba że zdecydujesz inaczej. Na końcu wypisuje, co
masz usunąć sam: sam folder FastStrata (silnik, bramkę, ich środowiska Pythona) i pliki modelu, z ich rozmiarem; inne
foldery FastStrata na tym komputerze mogą korzystać z tych samych plików modelu. Zostają też OpenCode, Python i
narzędzia do kompilacji, które setup mógł zainstalować.

## Konfiguracja

Setup zapisuje opcje silnika w konfiguracji uruchomienia modelu, `strata-<rodzina>-<preset>.json`, w `"args"`. Na
laptopie testowym, dla UD-Q4_K_M przy 144K:

```text
--max-context 147456 --kv-q8 --prompt-batch 512 --tail-skip 6 --threads 8 --attn-split --graph-experts
--elastic-prefill 1024 --lend-staging --stage-next --suffix-drafts 256 --cpu-multi
--conversation-cache-mib 12288 --conversation-cache-slots 8 --prompt-cache-message 2048
--conversation-disk <folder danych>\conversations --conversation-disk-mib 4096
```

Każdą tam, gdzie komputer i preset ją przyjmują ([który komputer dostaje którą](docs/RESULTS.pl.md#konfiguracja)), a
setup wypisuje każdą razem z tym, co dała. Wszystkie wyłączone: `START-HERE.bat --setup --speedups off`, tak jak
konfigurację zapisywał setup PolyStrata; `--speedups on` włącza je z powrotem. Każda opcja szczegółowo (po angielsku):
[docs/FASTSTRATA_OPTIONS.md](docs/FASTSTRATA_OPTIONS.md).

## Skąd pochodzi

FastStrata jest trzecim projektem w linii. [Strata](https://github.com/Niko1221/Strata) (Niko1221 i współtwórcy)
uruchamia Qwen3.8-Flash-Next z ekspertami podzielonymi między kartę a procesor i wniosła instalator, serwer i aplikację
webową. [PolyStrata](https://github.com/VecSzn/PolyStrata) (VecSzn) dodała `strata-poly`, silnik, który czyta plik GGUF
modelu tam, gdzie leży, dla Qwen3.6-35B-A3B, Ornith 1.5 i innych. FastStrata zawęża go do dwóch modeli, które mieszczą
się na komputerze do gier, razem czterech presetów, i sprawia, że `strata-poly` jest szybki na małej karcie: czternaście
zmierzonych kroków, dwanaście zmian w silniku i jego ustawieniach ([każda z nich](docs/RESULTS.pl.md#co-zmieniła-faststrata)).

Do Qwen3.8-Flash-Next użyj [Strata](https://github.com/Niko1221/Strata). Plany: profile ekspertów dołączone do każdego
presetu; tryb dla komputerów z 16–24 GB
RAM. Co zmieniło się w każdym wydaniu (po angielsku): [CHANGELOG.md](CHANGELOG.md).

## Podziękowania i licencja

- [Strata](https://github.com/Niko1221/Strata), autor Niko1221 i współtwórcy: silnik Qwen, instalator, serwer i
  aplikacja webowa.
- [PolyStrata](https://github.com/VecSzn/PolyStrata), autor VecSzn: silnik `strata-poly`, który FastStrata przyspiesza.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): czytnik GGUF, kernele, na których opiera się
  `strata-poly`, i arytmetyka procesora, którą `--cpu-multi` zachowuje co do bitu.
- Qwen3.6-35B-A3B od [zespołu Qwen](https://huggingface.co/Qwen), skwantyzowany przez
  [Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF); Ornith 1.5 od ornith-ai, w kwantyzacji APEX
  autorstwa [mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF). Do plików każdego modelu
  stosuje się jego własna licencja.
- Pomysły: SuffixDecoding (Oliaro i in., NeurIPS 2025) dla `--suffix-drafts`; Expert Deferral (KTransformers, SOSP
  2025), zmierzony w kroku 6; KIVI i KVQuant dla układów pamięci 4-bitowej.

FastStrata jest otwartym oprogramowaniem na [licencji MIT](LICENSE), tak jak Strata i PolyStrata; licencja zachowuje
informację o prawach autorskich Strata. Kilka części ma własne licencje ([które](docs/HOW_IT_WORKS.md#license)).
Wszystkie podziękowania: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).

Znalazłeś problem? Otwórz [zgłoszenie (issue)][issues] i dołącz log setupu albo `strata-<rodzina>-<preset>.log` z
folderu FastStrata. Wynik z Twojego komputera jest mile widziany: `BENCH.bat` (`./bench.sh`),
[jak](docs/COMMUNITY_BENCHMARKS.md) (po angielsku). Jak zabezpieczony jest serwer: [SECURITY.md](SECURITY.md),
dokument PolyStrata.

[repo]: https://github.com/Jarson1408/FastStrata
[zip]: https://github.com/Jarson1408/FastStrata/archive/refs/heads/main.zip
[issues]: https://github.com/Jarson1408/FastStrata/issues
