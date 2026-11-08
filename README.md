> **Repository pubblico: solo il codice.** I dati non sono inclusi: risultati e quote di football-data,
> quote dei bookmaker (OddsPapi), xG (Sportmonks), previsioni e registri delle giocate. Si scaricano o si
> ricreano con `python -m forecast1x2 update-data` (le quote dei bookmaker richiedono una propria chiave
> OddsPapi). Le cifre citate più sotto vengono dal progetto originale, che resta privato.

# forecast1x2

Previsioni 1X2 con un modello Dixon-Coles per Serie A, Premier League, Liga, Bundesliga e Ligue 1.
Include un backtest walk-forward contro le quote di chiusura e una dashboard Streamlit.

## Uso rapido (macOS)

1. **Prima volta:** doppio clic su `installa.command`. Crea l'ambiente Python, scarica i dati e calcola le previsioni; ci vuole qualche minuto.
2. **Ogni volta:** doppio clic su `Avvia dashboard.command`. Si apre il browser su http://localhost:8501; chiudi la finestra del Terminale per fermare la dashboard.
3. **Prima di ogni giornata:** nella dashboard premi **«Aggiorna dati e previsioni»** nella barra a sinistra, oppure da terminale lancia `.venv/bin/python -m forecast1x2 aggiorna`.
   - Il momento migliore è venerdì pomeriggio, quando football-data pubblica le quote pre-partita.
   - La dashboard avvisa se i dati hanno più di 3 giorni o se le partite sono già iniziate.

### Aggiornamento automatico in base alle partite

**Su GitHub (in uso da ottobre 2026).** Il workflow `.github/workflows/aggiorna.yml` lancia `python -m forecast1x2 auto` ogni ora e salva nel repository i dati, le previsioni e il registro dei consigli aggiornati; la dashboard su Streamlit Cloud si aggiorna da sola. Gira senza Sportmonks, quindi le previsioni usano solo i gol (vedi la sezione xG). Si può lanciare a mano dalla scheda Actions → «Aggiornamento automatico» → Run workflow. Se lo usi, non tenere attivo anche l'aggiornamento sul Mac.

**Sul Mac (alternativa).** Doppio clic su `attiva aggiornamento automatico.command`. Da quel momento macOS lancia `python -m forecast1x2 auto` ogni 30 minuti.

**Quando aggiorna davvero:** il comando scarica e ricalcola solo se, dall'ultimo aggiornamento, è passato uno di questi momenti:

- 24 ore prima di una partita;
- 3 ore prima di una partita;
- 3 ore dopo una partita;
- 24 ore dopo una partita;
- in ogni caso, se l'ultimo aggiornamento ha più di 24 ore.

Negli altri controlli esce subito, senza accedere alla rete.

**Se qualcosa va storto:**

- un aggiornamento fallito (per esempio perché sei offline) viene ritentato al controllo successivo;
- si esegue un solo aggiornamento alla volta: se ne parte un secondo, per esempio dal pulsante della dashboard mentre è in corso quello automatico, viene rifiutato con un messaggio;
- le fonti facoltative non bloccano mai l'aggiornamento: xG, quote pre-partita e calendario di riserva. Se mancano si usano gli ultimi dati disponibili, e la dashboard mostra un avviso giallo;
- le quote pre-partita si usano solo se la data coincide con quella della partita e sono state scaricate da meno di 7 giorni;
- un file di una stagione chiusa a cui mancano più di 2 partite blocca l'aggiornamento: va riscaricato con `update-data --force`;
- non si fanno previsioni per squadre senza dati storici: quasi sempre è un errore di mappatura dei nomi.
- il registro delle operazioni è in `logs/auto.log`;
- la dashboard mostra nella barra a sinistra se l'aggiornamento automatico è attivo e quando è previsto il prossimo.

**Note:**

- Funziona solo quando il Mac è acceso e hai effettuato l'accesso. I controlli saltati durante lo stop vengono recuperati al risveglio.
- Per spegnerlo: doppio clic su `disattiva aggiornamento automatico.command`.
- Se sposti la cartella del progetto, rilancia lo script di attivazione.

Se macOS blocca i file `.command` alla prima apertura: tasto destro sul file → Apri → Apri.

La dashboard è raggiungibile solo da questo computer (`localhost`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Testato con Python 3.9.6, l'unico disponibile su questa macchina. Con Python ≥ 3.10 funzionano anche versioni più recenti delle librerie.

## Comandi

```bash
.venv/bin/python -m forecast1x2 aggiorna             # update-data + predict, il comando di tutti i giorni
.venv/bin/python -m forecast1x2 update-data          # scarica e normalizza dati e calendari
.venv/bin/python -m forecast1x2 update-data --force  # riscarica anche le stagioni chiuse
.venv/bin/python -m forecast1x2 backtest             # calibra ξ, λ e w (2022/23 + 2023/24), backtest su 2024/25 e 2025/26 (~4 min)
.venv/bin/python -m forecast1x2 predict              # previsioni della prossima giornata
.venv/bin/python -m forecast1x2 consigli-backtest    # verifica storica della regola dei consigli in prova
.venv/bin/python -m forecast1x2 quote-giuste-verifica # parametri e verifica delle quote giuste dei mercati (~1 min)
.venv/bin/streamlit run app/dashboard.py --server.address localhost   # dashboard su http://localhost:8501
.venv/bin/python -m pytest                           # test
```

Uso tipico durante la stagione: `update-data`, poi `predict`, poi aggiornare la dashboard.
Se `outputs/hyperparams.json` non esiste, `predict` calibra prima gli iperparametri.
Le quote pre-partita (`fixtures.csv` di football-data) escono pochi giorni prima di ogni giornata, di solito il venerdì: conviene rilanciare `update-data` e `predict` a ridosso del weekend.

## Dati

| Fonte | Uso |
|---|---|
| football-data.co.uk `mmz4281/{SSSS}/{DIV}.csv`, dal 2020/21 al 2026/27 | risultati, quote di apertura e di chiusura (prima divisione) |
| football-data.co.uk, stesse stagioni, I2/E1/SP2/D2/F2 | risultati di seconda divisione, per stimare le neopromosse |
| football-data.co.uk `fixtures.csv` | quote pre-partita delle prossime partite (solo come quote, non come calendario) |
| fixturedownload.com `feed/json/{slug}-2026` | calendario (fonte primaria) |
| openfootball `2026-27/{it.1,…}.json` | calendario (fallback) |

- **Download:** un file viene sovrascritto solo se la risposta è HTTP 200, l'intestazione è quella attesa e c'è almeno una riga. Le stagioni chiuse non si riscaricano, salvo con `--force`.
- **Quote di chiusura:** media di mercato (`AvgCH/CD/CA`), con `B365CH/CD/CA` come fallback riga per riga. La colonna `odds_source` indica quale è stata usata.
- **Quote di apertura:** media di mercato (`AvgH/D/A`), con `B365H/D/A` come fallback. football-data le raccoglie il venerdì (o il martedì per le partite infrasettimanali), quindi prima della partita: si possono usare senza leakage.
- **Risultati mancanti:** se manca in prima divisione, la pipeline si ferma. In seconda divisione la riga viene scartata e segnalata; a ottobre 2026 succede per Bastia–Red Star, Ligue 2 2025/26.
- **Nomi delle squadre:** `data/team_names.csv` mappa i nomi dei calendari sui nomi di football-data, separatamente per campionato e per fonte. Se una squadra non è mappata, la pipeline si ferma con un errore.
- **Validazione del calendario:** un calendario deve contenere tutte le coppie casa/trasferta senza ripetizioni, e ogni squadra deve giocare n−1 partite in casa e n−1 in trasferta. Se la fonte primaria non supera il controllo, si usa openfootball e viene stampato un avviso. A ottobre 2026 succede per la Ligue 1: fixturedownload riporta Rennes–PSG due volte e non ha PSG–Rennes.
- **Partite giocate:** una partita del calendario conta come giocata quando compare nei risultati di football-data, oppure quando il calendario stesso riporta già il punteggio (football-data può arrivare con uno o due giorni di ritardo).
- **Prossima giornata:** è il `RoundNumber` più basso con almeno metà delle partite ancora da giocare. Le partite non giocate di giornate precedenti vengono mostrate con un'etichetta:
  - *recupero*, se hanno una data futura;
  - *rinviata*, se la data è già passata senza risultato.

## Modello

Il modello è un Dixon-Coles stimato separatamente per ogni campionato. La stima usa sia le partite di prima divisione sia quelle della seconda divisione corrispondente. Il tasso di gol della squadra di casa è

log λ = μ + δ·[seconda divisione] + casa + attacco_casa + difesa_trasferta + κ·(neo_trasf − neo_casa)

e per la squadra in trasferta vale lo stesso, senza il termine casa e con il segno di κ invertito.

- **Seconda divisione:** le squadre che salgono o scendono collegano le due scale, così una neopromossa parte da un rating basato sui dati e non dalla media del campionato. δ è il diverso livello di gol in seconda divisione.
- **κ:** misura quanto rende di meno una squadra nella prima stagione dopo la promozione rispetto a quanto direbbe il suo rating. Serve perché la forza non resta costante quando si cambia divisione. Viene stimato dai dati: a ottobre 2026 vale tra 0,03 e 0,13 a seconda del campionato.
- **Correzione dei punteggi bassi:** τ corregge 0-0, 1-0, 0-1 e 1-1, con il parametro ρ limitato all'intervallo [−0.2, 0.2].
- **Identificabilità:** Σattacco = Σdifesa = 0, imposta tramite la parametrizzazione.
- **Pesi e penalità:** ogni partita pesa exp(−ξ·giorni), con una penalità L2 di peso λ su attacco e difesa.
- **Stima:** L-BFGS-B con gradiente analitico, verificato numericamente nei test.
- **Probabilità:** dalla matrice dei punteggi 0–10 rinormalizzata si ricavano 1/X/2, i gol attesi e il risultato più probabile.

### Iperparametri (`outputs/hyperparams.json`)

- **ξ e λ:** sono comuni a tutti i campionati e scelti su una griglia, minimizzando la log-loss walk-forward sul 2022/23 e sul 2023/24, quindi prima del backtest. Ho provato anche a sceglierli per campionato, ma è overfitting: scegliendoli sul 2022/23 e valutandoli sul 2023/24, la log-loss era 0,9857 per campionato contro 0,9751 con valori comuni.
- **w:** è il peso del modello nel pool logaritmico con le quote di apertura senza margine, P ∝ P_modello^w · P_mercato^(1−w). È calibrato sulle stesse stagioni e oggi vale **0**: il modello non aggiunge informazione alle quote di apertura.

## Valutazione

- **Walk-forward:** il backtest procede per settimane di calendario (da lunedì). Ogni settimana si ristima il modello usando solo le partite con data anteriore al lunedì.
- **Confronti, sulle stesse partite:**
  - modello + quote di apertura (pool con peso w);
  - quote di apertura senza margine;
  - quote di chiusura senza margine, il riferimento principale (normalizzazione proporzionale; il metodo di Shin come confronto);
  - frequenze storiche 1/X/2 del campionato, calcolate solo sul passato.
- **Metriche:** log-loss, Brier multiclasse e RPS, per campionato, per stagione e in totale.
- **Intervallo di confidenza:** IC 95% bootstrap sulla differenza tra ciascuna fonte e il mercato di chiusura. Si ricampionano settimane intere, perché le partite della stessa settimana non sono indipendenti.

## File in outputs/

| File | Contenuto |
|---|---|
| `predictions.csv` | previsioni della prossima giornata, con recuperi e partite rinviate; quando le quote pre-partita sono disponibili, anche le probabilità di mercato (`market_*`) e il pool (`comb_*`) |
| `ratings.csv` | attacco e difesa stimati per le squadre della stagione in corso, con l'indicazione delle neopromosse; κ non è incluso |
| `meta.json` | data dell'aggiornamento dei dati, ξ, vantaggio casa e ρ per campionato |
| `hyperparams.json` | ξ, λ, w e le griglie di calibrazione |
| `backtest_matches.csv` | probabilità per partita di tutte le fonti, con `train_max_date` per il controllo del leakage |
| `backtest_metrics.csv` | metriche aggregate |
| `backtest_bootstrap.csv` | IC bootstrap della differenza modello − mercato |
| `calibration.csv` | dati per la calibrazione |

La dashboard legge solo da questi file e non rifà le stime.

## Ensemble Dixon-Coles + Elo e affidabilità

**Storico.** Si parte dalla stagione 2015/16, per prima e seconda divisione. Le stagioni 2019/20 di Ligue 1 e Ligue 2, interrotte per il COVID, sono un'eccezione dichiarata nel controllo di completezza. Nel Dixon-Coles le partite con peso sotto 0,001 sono escluse dalla stima, perché non contano nulla e la rallentano.

**Calibrazione.** Gli iperparametri si scelgono sulle stagioni 2021/22, 2022/23 e 2023/24, tutte precedenti al backtest. Valori attuali:
- Dixon-Coles: ξ = 0,002 (emivita circa 347 giorni), λ = 0,25.
- Elo: K = 15, regressione verso la media tra le stagioni 0,1.
- Ensemble: 0,6 Dixon-Coles + 0,4 Elo.
- Peso degli xG: 0,5.

**Il secondo modello.** Accanto al Dixon-Coles c'è un modello diverso (`src/forecast1x2/elo.py`): rating Elo aggiornati partita per partita, con un moltiplicatore per la differenza reti, più una regressione logistica ordinale che trasforma la differenza di rating in P(1), P(X), P(2). Le neopromosse partono dalla media delle squadre retrocesse. La regressione ordinale usa solo le settimane precedenti.

**L'OLS** (regressione lineare con minimi quadrati) è incluso solo come confronto: è il peggiore dei modelli testati.

**Risultati** su 3.504 partite (2024/25 e 2025/26):

| Fonte | Log-loss | Esito più probabile indovinato | Errore medio delle probabilità | Capacità vs mercato |
|---|---|---|---|---|
| **Ensemble + xG (in uso)** | **0,9857** | 52% | ±1,4 pp | **84%** |
| Ensemble senza xG | 0,9872 | 53% | ±1,4 pp | 83% |
| Dixon-Coles | 0,9888 | 52% | ±1,7 pp | 81% |
| Elo | 0,9890 | 52% | ±1,4 pp | 81% |
| OLS | 0,9977 | 52% | ±1,7 pp | 73% |
| Mercato di chiusura | 0,9691 | 54% | ±1,3 pp | 100% |
| Baseline (frequenze 1/X/2) | 1,0748 | 43% | ±0,8 pp | 0% |

- Lo storico dal 2015 ha migliorato tutti i modelli rispetto a quello dal 2020: ensemble −0,0017, Elo −0,0029.
- Rispetto al solo Dixon-Coles, l'ensemble con xG migliora la log-loss di −0,0031, con IC 95% da −0,0053 a −0,0012: un miglioramento significativo.
- Il distacco dal mercato di chiusura resta significativo (+0,017).

**Candidati provati e scartati.** Sono stati valutati solo sulle stagioni di calibrazione, con la regola fissata prima: si tiene un candidato solo se migliora la log-loss dell'ensemble di almeno 0,0005. I pesi scelti sugli stessi dati sono stati misurati lasciando fuori una stagione a rotazione.

| Candidato | Guadagno | Esito |
|---|---|---|
| Correzione dei pareggi | +0,00001 | scartato |
| Pi-ratings (rating dinamici) come terzo modello | +0,00037 | scartato, sotto soglia |
| Vantaggio casa per squadra | da −0,0006 a −0,0009 | scartato, peggiora |
| Giorni di riposo, solo dalle partite di campionato | −0,00004 | scartato |
| Coppe europee e calendario fitto | — | non provato: su openfootball c'è solo la Champions 2024/25 |

## Livelli di rischio e simulazione scommesse

**Le previsioni non sono consigli di scommessa.** La dashboard lo ricorda in ogni pagina delle previsioni.

**Livelli di rischio.** Ogni partita ha un livello, in base alla probabilità dell'esito più probabile:

| Livello | Probabilità del favorito | Partite (backtest 2024-26) | Il favorito ha vinto (modello + xG) |
|---|---|---|---|
| 🟢 1, basso | ≥ 60% | 23% | 72% |
| 🟡 2, medio | 45–60% | 43% | 51% |
| 🔴 3, alto | < 45% | 35% | 40% |

Le percentuali si aggiornano a ogni backtest (`outputs/risk_tiers.csv`). La colonna «Modello − mercato» mostra dove il modello dà a un esito più probabilità delle quote pre-partita: è un'informazione, non un suggerimento.

**Simulazione scommesse** (`outputs/betting_sim.csv`, pagina Backtest). Sono state simulate giocate da 1 unità a partita, alle quote medie pre-partita, sul 2024/25 e sul 2025/26:

| Strategia | Rendimento | IC 95% |
|---|---|---|
| Gioca dove il modello «vede valore» (soglie 0%, 5%, 10%) | da −17% a −22% | sempre sotto zero |
| Gioca sempre il favorito del mercato (riferimento) | −3,4% | circa il margine del bookmaker |

Le cifre valgono sia per il modello sui soli gol sia per quello con xG. Il modello «vede valore» soprattutto dove sbaglia rispetto al mercato.

## Quote giuste per tutti i mercati sul risultato

Pagina «Quote giuste e strategia» della dashboard; codice in `src/forecast1x2/markets.py`. Copre 181 mercati del regolamento scommesse calcio, nei 10 campionati:
- 1X2, Doppia chance, Draw no bet, Under/Over da 0,5 a 5,5, Goal/No goal, Multigoal (anche di squadra), Under/Over di squadra, Pari/Dispari, Handicap, Risultato esatto;
- Combo 1X2 + U/O, 1X2 + GG/NG, DC + U/O, DC + GG/NG, DC + Multigoal, 1X2 + Multigoal, GG/NG + U/O;
- primo e secondo tempo, Tempo/finale, Tempo con più gol, Gol in entrambi i tempi.

Marcatori, angoli, cartellini e simili non ci sono, perché mancano i dati.

**Come si calcola.**
1. Dalle quote medie 1X2 e Over/Under 2.5 del venerdì (i mercati più precisi) si tolgono i margini.
2. Si cercano i gol attesi delle due squadre che riproducono quelle probabilità.
3. Da questi si ricava la probabilità di ogni altro mercato.

I gol di una squadra seguono una Poisson corretta (Conway-Maxwell-Poisson, ν = 1,1: meno zeri e meno goleade, come nella realtà), con la correzione dei punteggi bassi di Dixon-Coles (ρ = −0,05). Nel primo tempo cade il 43,8% dei gol. I parametri sono scelti sul 2015/16-2023/24; il file `outputs/markets_params.json` si rigenera con `python -m forecast1x2 quote-giuste-verifica`.

**Verifica 2024/25-2025/26** (7.514 partite, `outputs/markets_calibration.csv`):
- tra probabilità prevista e frequenza reale ci sono in media 0,55 punti percentuali, al massimo 1,7 (Handicap casa −1);
- con la Poisson semplice erano 0,82 in media, e il Goal veniva sottostimato di 2,5 punti.

La pagina mostra l'errore di ogni mercato accanto alla sua quota giusta.

**Uso.** Si sceglie la partita e si scrive la quota del proprio bookmaker (va bene sia 2,40 sia 2.40). La pagina mostra:
- il **valore** (probabilità × quota − 1);
- la **puntata** suggerita: un quarto del criterio di Kelly, al massimo il 5% del budget;
- un giudizio: «valore», «valore entro l'errore» (non affidabile) oppure «nessun valore». Una quota oltre il doppio di quella giusta viene segnalata come probabile errore di battitura.

Le giocate con valore si possono aggiungere a una **schedina**, che confronta le singole con la multipla.

**Quote automatiche di 3 bookmaker (OddsPapi, piano gratuito).** Con una chiave [OddsPapi](https://oddspapi.io) si scaricano le quote di **Planetwin365, Sisal/Snai e Bet365**, e per ogni giocata si usa la quota migliore dei tre («Dove» dice quale). Codice in `src/forecast1x2/oddspapi.py`.
- **Perché questi tre:** molti marchi italiani usano le stesse quote. Planetwin365 = Goldbet = Lottomatica = Betflag; Snai = Sisal = PokerStars. Questi tre coprono i listini principali diversi.
- **Margine misurato il 23/10/2026, quote 1,01-2,50** (rendimento atteso rispetto alla quota giusta):

  | Bookmaker | Rendimento atteso |
  |---|---|
  | Planetwin365 | −7,2% |
  | Sisal/Snai | −6,8% |
  | Bet365 | −5,4% |
  | migliore dei tre | −5,2% |

  Bet365 ha la quota migliore in circa 2 casi su 3.
- **Chiave:** variabile `ODDSPAPI_API_KEY` (segreto di GitHub con lo stesso nome) oppure file `~/.config/forecast1x2/oddspapi_key`. Senza chiave tutto il resto funziona come prima.
- **Consumo:** l'API accetta un solo bookmaker e al massimo 5 campionati per richiesta, quindi servono 2 richieste per bookmaker. Si scarica al massimo una volta ogni 23 ore (3 × 2 richieste al giorno, circa 180 al mese). Ogni 30 giorni servono 3 richieste per i riferimenti (`data/oddspapi_reference.json`). Il contatore in `data/oddspapi_state.json` si ferma a 200 richieste al mese, sotto le 250 del piano gratuito, e non scarica mai metà di un bookmaker.
- **Mercati:** 1X2, Doppia chance, Under/Over, Goal/No goal, Draw no bet, Pari/Dispari, Under/Over di squadra, primo e secondo tempo, se il bookmaker li offre su OddsPapi. Multigoal e Combo vanno scritti a mano.
- **Abbinamento:** le partite si abbinano per campionato, orario (±3 ore) e somiglianza dei nomi, entrambe le squadre. Le partite non abbinate restano senza quota.
- **Download manuale:** `python -m forecast1x2 quote-planetwin`. Le quote vengono scaricate in `outputs/bookmaker_odds.csv`.

**Registro dei consigli e CLV** (`src/forecast1x2/tracking.py`, pagina «Verifica»). A ogni aggiornamento si salvano, prima delle partite e una volta sola:
- per ogni partita, la giocata più probabile e la più conveniente;
- una volta al giorno, le quattro schedine pronte.

Il registro è in `data/tips/picks_register.csv` e `slips_register.csv`. Dopo la partita si segna l'esito, dal risultato finale e del primo tempo, e si calcola il **CLV**: quota presa × probabilità del mercato alla chiusura − 1, con le quote medie di chiusura di football-data passate nello stesso motore delle quote giuste.
- Un CLV medio positivo dice che i consigli prendono quote migliori di quelle finali: è il segnale più rapido, leggibile dopo 3-4 settimane.
- Il rendimento, invece, richiede una stagione.
- Nelle schedine una partita rimborsata o annullata vale 1,00, come da regolamento.

**Dipendenza tra i gol delle due squadre: provata, non adottata.** Il parametro κ, che moltiplica per exp(κ) i risultati in cui segnano entrambe, è stato stimato sul 2015-24 insieme a ρ e ν: il valore migliore è κ = 0. Sul 2015-24 il Goal è già calibrato (−0,01 punti). Il suo scarto sul 2024-26 (+1,0) coincide con quello dell'Over 2,5 (+1,1): in quelle stagioni si è segnato più di quanto il mercato prevedesse, non manca un pezzo al modello. Il parametro resta nel codice con valore 0.

**Limiti.**
- Le probabilità sono quelle del mercato: il valore nasce solo dove il proprio bookmaker paga più del mercato. Succede soprattutto sui mercati «esotici» (Combo, Multigoal), dove il margine è più alto e le quote sono meno curate.
- Per questi mercati non abbiamo quote storiche: si è verificato che le probabilità sono giuste, non che giocarli avrebbe reso.

## Diagnosi econometrica (ottobre 2026)

Metodi da Hill, Griffiths e Lim, *Principles of Econometrics* (5ª ed., 2018): logit binario e condizionale (16.2, 16.4), test di Wald e del rapporto di verosimiglianza (16.2.5-16.2.6), errori standard robusti raggruppati (15.3, appendice 15A). Gli errori sono raggruppati per settimana, perché le partite della stessa settimana non sono indipendenti.

**1. Il mercato 1X2 è calibrato?** Logit condizionale P(j) ∝ exp(a_j + b·log q_j) sulle probabilità senza margine delle quote medie del venerdì, stimato sul 2015/16-2023/24 (35.047 partite).
- b = 1,112 (es 0,018) invece di 1: le quote schiacciano le probabilità verso il centro, quindi i favoriti vincono un po' più spesso di quanto dicano (favourite-longshot bias).
- a_X = +0,059 (es 0,016): il pareggio è un po' sottostimato.
- Test di Wald di «a = 0, b = 1»: p ≈ 10⁻¹⁰. Il test LR concorda. La calibrazione è la stessa in prima e seconda divisione (p = 0,25).
- Fuori campione (2024/25-2025/26): log-loss 1X2 da 1,00496 a 1,00386, differenza −0,0011 con IC 95% da −0,0020 a −0,0002.
- Per l'Over/Under 2,5 lo stesso logit trova b = 1,14, ma fuori campione la correzione non migliora (+0,0002, IC tra −0,0004 e +0,0008): non si usa.

**2. Dove si applica.** Portare la correzione nei gol attesi peggiora tutti gli altri mercati: sul 2024-26 lo scarto medio passa da 0,55 a 0,69 punti e Goal/No goal peggiora di 0,7 punti, perché un modello sui gol non può rappresentare la correzione del pareggio. Si applica quindi solo ai mercati che dipendono dal 1X2: 1X2, Doppia chance e Draw no bet (`markets.override_1x2`, parametri in `outputs/markets_params.json` → `calib_1x2`). Sul 2024-26 l'errore di calibrazione (ECE) scende:
- pareggio da 0,95 a 0,18 punti;
- vittoria in trasferta da 1,30 a 0,58;
- Draw no bet da 1,81 a 0,98.

Gli altri mercati restano invariati.

**3. Il nostro modello aggiunge informazione al mercato?** Test di encompassing: si aggiunge log p_modello al logit condizionale (2024/25-2025/26, prima divisione). Il coefficiente del modello è negativo e significativo per tutte e tre le versioni:
- ensemble −0,37 (p = 0,002);
- Dixon-Coles −0,32 (p = 0,005);
- Elo −0,32 (p = 0,003).

Quando il modello si discosta dal mercato sbaglia in modo sistematico: conferma che il modello non va usato per «correggere» le quote (peso w = 0 nel pool, simulazione delle scommesse in perdita).

## Schedina consigliata

Due pagine della dashboard; codice in `src/forecast1x2/advisor.py`, con le quote Planetwin365, la probabilità di vincere e il rendimento atteso.
- **«Schedine pronte»** (pagina iniziale) combina le partite di tutti i 10 campionati.
  - Filtri: periodo (prossimi 7 giorni, oggi, domani, fino a domenica) e campionati.
  - In cima, le quattro più probabili: singola ≥ 1,50, doppia ≥ 1,50, doppia ≥ 2,00, tripla ≥ 3,00.
  - «Costruisci la tua»: da 1 a 5 partite, quota totale minima e obiettivo (più probabile oppure miglior rendimento atteso). Mostra 3 alternative senza partite in comune.
  - La ricerca è esatta: per ogni partita tiene le giocate non battute sia in probabilità sia in quota, e prova tutte le combinazioni tra le partite migliori (80 per le singole, 30 per le doppie, 16 per le triple, 11 per le quadruple, 9 per le cinquine). Dura meno di un secondo.
- **«Giocate per partita»**: per ogni partita la giocata più probabile e la più conveniente, più Combo e Multigoal da cercare.

**Cosa dice il regolamento scommesse** (documento Sisal sul calcio, 572 pagine):
- più esiti della stessa partita si combinano con i mercati **Combo** (1X2 + U/O, DC + GG/NG, DC + Multigoal…), che sono una scommessa sola; solo i mercati segnati «(LEGABILE)», quasi tutti sui giocatori, si legano liberamente;
- in una multipla una partita rinviata o annullata vale **quota 1,00**;
- conta solo il tempo regolamentare, recupero compreso.

**Cosa dicono i dati** (ottobre 2026):

Margine di Planetwin365 sulle 2.111 quote del 23/10/2026, misurato come rendimento atteso rispetto alla quota giusta:

| Fascia di quota | Rendimento atteso |
|---|---|
| 1,01-1,60 | circa −6% |
| 1,60-3,00 | da −8% a −10% |
| 3,00-5,00 | −12% |
| oltre 5 | −29% |

I mercati più convenienti sono Over 1,5, 1X2, Under 3,5, Goal e Doppia chance 12 (circa −5/−6%); i peggiori Under 0,5 e Over 5,5 (circa −38%).

Storico 2015-2026 alle quote Bet365 (1X2 e Over/Under 2,5), verifica 2024/25-2025/26:

| Come giocare | Vinte | Rendimento |
|---|---|---|
| Singola: la più probabile di ogni partita | 60% | −3,6% |
| …con quota 1,01-1,30 | 83% | +0,1% |
| Doppia delle più probabili | 37% | −4,4% |
| Tripla | 21% | −10,0% |
| Quadrupla | 13% | −11,6% |
| Cinquina | 8% | −15,9% |
| «Valore» contro il consenso su un solo bookmaker (soglia +5%, scelta sulla calibrazione dove rendeva +9,4%) | 10% | −49,5% |

**Regole del consigliere**, che seguono da questi numeri:
- giocata più probabile di ogni partita, con quota tra 1,20 e 2,50 e costo atteso non peggiore di −6%;
- per ogni partita anche la «più conveniente»: tra quelle con almeno il 60% di probabilità, quella su cui il bookmaker trattiene meno;
- schedine con al massimo 3 partite diverse; la combinazione più probabile che raggiunge la quota totale richiesta si cerca in modo esatto tra le 15 partite con le giocate più probabili;
- Combo e Multigoal (che Planetwin365 non pubblica su OddsPapi): le più probabili di ogni partita, con la quota minima sotto cui non conviene giocarle (quota giusta − 6%).

Nessun modo di giocare ha guadagnato in modo affidabile: il consigliere sceglie quello che perde meno, non uno che vince.

## Consigli in prova: over/under 2.5

Un registro di giocate **simulate** (1 unità, nessun soldo vero) per verificare in avanti l'unica idea che nello storico ha mostrato un segnale. Codice in `src/forecast1x2/tips.py`, pagina «Consigli in prova» nella dashboard.

**Ricerca (ottobre 2026).** Sullo stesso storico 2015-2026 dei 10 campionati sono state provate le strategie con un fondamento nella letteratura. Le soglie sono state scelte sul 2015/16-2023/24 e verificate sul 2024/25-2025/26. Nessuna batte il mercato in modo affidabile:

| Strategia (quota migliore disponibile) | Verifica 2024-26 |
|---|---|
| Modello del progetto contro la quota migliore | da −5% a −18% |
| 1X2 con la quota migliore molto sopra la media (Kaunitz et al., 2017) | −28% |
| Sempre gli sfavoriti (probabilità < 20%) | −15% |
| Sempre il pareggio (quota ≥ 2,62) / sempre i favoriti forti | circa 0% |
| **Over/under 2.5 con la quota migliore molto sopra la media** | **segnale, ma su pochi casi** |

**Regola `ou25-v1`, congelata il 23/10/2026.** Con le quote del venerdì di football-data (`fixtures.csv`):
- probabilità «giusta» = quote medie over/under senza margine;
- vantaggio = quota migliore × probabilità giusta − 1, sul lato migliore;
- consiglio se il vantaggio supera il 2%, in prima e seconda divisione.

La soglia è stata scelta sul periodo di calibrazione (`python -m forecast1x2 consigli-backtest`, risultati in `outputs/tips_backtest.csv`):

| Periodo | Giocate | Rendimento | IC 95% |
|---|---|---|---|
| Calibrazione 2015-24, tutte | 1.197 | +3,8% | da −3,4% a +11,4% |
| di cui seconda divisione | 385 | +16,0% | da +3,9% a +28,4% |
| Verifica 2024-26, tutte | 105 | +5,1% | da −18,9% a +30,6% |
| di cui seconda divisione | 42 | +32,4% | da −5,4% a +68,0% |

Il segnale c'è solo in seconda divisione, con circa 100 casi a stagione in tutto: per questo è una prova e non un consiglio.

**Come funziona il registro** (`data/tips/ou_register.csv`):
- si aggiorna da solo a ogni `update-data`/`aggiorna`/`auto`; un errore del registro non blocca mai le previsioni;
- una partita entra una volta sola, almeno un'ora prima dell'inizio, con le quote di quel momento, e non viene più modificata;
- dopo la partita la giocata diventa «vinta» o «persa»; se dopo 7 giorni manca il risultato (partita rinviata) diventa «annullata».

**Limiti.**
- La quota migliore è la più alta tra i bookmaker raccolti da football-data, che non dice quale. Spesso coincide con Betfair Exchange, dove sulle vincite si paga una commissione.
- Chi vince con regolarità viene limitato dai bookmaker: è successo anche agli autori dello studio del 2017.
- Prima di qualsiasi conclusione serve almeno una stagione intera di prova.

## xG (expected goals): disattivati da ottobre 2026

**Decisione (23/10/2026).** Sportmonks è stato tolto dall'aggiornamento automatico su GitHub. Il guadagno degli xG è piccolo e non statisticamente certo: ensemble 0,9857 con xG contro 0,9872 senza; sul solo Dixon-Coles −0,0012 di log-loss, con IC 95% da −0,0035 a +0,0011. In pratica le previsioni cambiano di pochi decimi di punto percentuale. Senza xG le previsioni usano automaticamente il peso xG 0 e la dashboard mostra le statistiche dell'ensemble senza xG. Il registro dei consigli non usa gli xG.

Il codice è rimasto: con la chiave Sportmonks (file o variabile qui sotto) gli xG tornano attivi, ad esempio lanciando gli aggiornamenti dal Mac. Quanto segue descrive come funzionavano.

Gli xG per partita arrivano da Sportmonks (piano Starter + add-on xG).
La chiave si legge da `~/.config/forecast1x2/sportmonks_token`, un file fuori dal progetto e leggibile solo dall'utente; in alternativa dalla variabile `SPORTMONKS_API_TOKEN`.

- **Copertura:** sul nostro abbonamento gli xG partono dal **2024/25** (controllato a ottobre 2026). Le stagioni precedenti non restituiscono partite.
- **Test incrociato:** il 2024/25 viene previsto con il peso degli xG scelto sul 2025/26 (0,5), e il 2025/26 con quello scelto sul 2024/25 (0,75). Le stime del modello usano sempre solo partite passate; l'unica scelta che attraversa le stagioni è il peso. Risultati dell'ottobre 2026 su 3.504 partite:

  | Fonte | Log-loss |
  |---|---|
  | Modello, solo gol | 0,9902 |
  | Modello + xG | 0,9873 |
  | Mercato di chiusura | 0,9691 |

  - Rispetto al modello sui soli gol, gli xG migliorano di −0,0029 (IC 95% da −0,0058 a +0,0002): l'effetto va nella stessa direzione in entrambe le stagioni ed è quasi significativo.
  - Il distacco dal mercato resta significativo: +0,018.
- **Previsioni:** usano il peso scelto sulle due stagioni insieme, 0,75.
- **Dove si vede:** il backtest principale resta sui soli gol, per essere confrontabile con i risultati precedenti. Il test xG è in `outputs/xg_metrics.csv` e `outputs/xg_bootstrap.csv`, e nella dashboard (pagina Backtest).
- **Nomi delle squadre:** 121 nomi di Sportmonks mappati in `data/team_names.csv` (`source = sportmonks`). La mappatura è stata ricavata confrontando le partite con football-data e ritrova il 100% delle partite giocate.

Fonti gratuite scartate:

| Fonte | Motivo |
|---|---|
| Understat | `robots.txt` con `Disallow: /` |
| FBref | risponde 403 alle richieste automatiche |
| Sofascore | risponde 403 alle richieste automatiche |
| FotMob | vieta `/api/` nel `robots.txt` |
| xgscore.io | i termini d'uso vietano di copiare i dati |
