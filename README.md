# VisionSoftwareMDE

Machine vision systeem voor geautomatiseerde visuele inspectie op de productielijn van MDE Automation. Het systeem combineert OCR en objectdetectie om producten te beoordelen (OK/NOK) en communiceert de uitslag via een web-dashboard en optioneel via een TCP-koppeling met de PLC.

---

## Vereisten

- Windows 10/11
- Python 3.10+
- Tesseract-OCR geïnstalleerd op `C:/Program Files/Tesseract-OCR/tesseract.exe`
- Webcam of industriële camera (USB)

---

## Installatie

```powershell
git clone <repo-url>
cd VisionSoftwareMDE

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the project root and replace the placeholder values before starting the application:

```dotenv
ROBOFLOW_API_KEY=your_key_here
MAINTENANCE_PASSWORD=your_password_here
```

---

## Starten

```powershell
python -m backend.main
```

Het dashboard is beschikbaar op `http://localhost:5000`.  
Druk op `Q` in de terminal om handmatig een inspectiecyclus te starten.

---

## Configuratie

Alle instellingen staan in `config/default.yaml`. De meest relevante opties:

| Instelling | Beschrijving |
|---|---|
| `vision_mode` | `"ocr"` of `"object_detection"` |
| `object_detection.backend` | `"classifier"`, `"yolo"`, `"template"` of `"roboflow"` |
| `camera.index` | Cameranummer (0 = eerste USB-camera) |
| `roi` | Regio van interesse als genormaliseerde coördinaten (0–1) |
| `trigger.enabled` | `true` om TCP-koppeling met PLC te activeren |
| `ocr.keywords` | Tekst die herkend moet worden voor een OK-beoordeling |

Voor de Roboflow-backend: sla de API-sleutel op als omgevingsvariabele (`ROBOFLOW_API_KEY`) en verwijs daar in de config naar, zodat de sleutel niet in de repository staat.

Voor onderhoudsfuncties: zet het wachtwoord in `MAINTENANCE_PASSWORD` in plaats van in `config/default.yaml`.

---

## Projectstructuur

```
VisionSoftwareMDE/
├── backend/
│   ├── core/           # Camera, vision-engine, AppState, TCP-trigger
│   ├── detection/      # OCR- en objectdetectie-implementaties
│   └── output/         # Resultaatopslag (JSONL)
├── frontend/           # Web-dashboard (Flask + HTML/JS)
├── config/             # Configuratiebestand (YAML)
├── data/
│   ├── dataset/        # Afbeeldingen voor classifier-training
│   ├── results/        # Inspectieresultaten per dag (JSONL)
│   ├── templates/      # Referentieafbeeldingen voor template matching
│   └── logs/           # Logbestanden
├── models/             # Getrainde modellen (classifier, YOLO)
├── tools/              # Hulpscripts: dataset opnemen, splitsen, trainen
├── benchmarks/         # Snelheidsmeting OCR en objectdetectie
├── tests/              # Testscripts
└── docs/               # Uitgebreide documentatie
```

---

## Classifier trainen

```powershell
# 1. Afbeeldingen opnemen (OK en defective)
python .\tools\capture_dataset.py

# 2. Dataset splitsen (train/val/test)
python .\tools\split_dataset.py

# 3. Model trainen
python .\tools\train_classifier.py
```

Het getrainde model wordt opgeslagen in `models/classifier/final/`.  
Zie `docs/classifier-and-object-detection.md` voor meer informatie.

---

## PLC-koppeling (TCP)

Activeer in `config/default.yaml`:

```yaml
trigger:
  enabled: true
  port: 5001
  trigger_byte: "0x01"
```

De PLC stuurt byte `0x01` → het systeem voert een inspectiecyclus uit → de PLC ontvangt `OK\n` of `NOK\n`.

---

## Resultaten

Inspectieresultaten worden per dag opgeslagen in `data/results/` als JSONL-bestanden:

```json
{"timestamp": "2026-06-04T10:30:45", "status": "OK", "confidence": 0.95, "cycle_time_ms": 125}
```

---

## Documentatie

| Document | Inhoud |
|---|---|
| `docs/architecture.md` | Systeemarchitectuur, threading-model en data flow |
| `docs/configuration.md` | Alle configuratie-opties in detail |
| `docs/development.md` | Uitbreiden en debuggen; debugging-instructies |
| `docs/classifier-and-object-detection.md` | Modeltraining en backend-keuze (classifier, YOLO, Roboflow, template) |
| `docs/project-map.md` | Compleet projectoverzicht met alle modules en functies |
| `docs/roadmap.md` | Toekomstige verbeteringen en plannen |

---

## Testen

Eenheidstests en integratietests draaien met:

```powershell
# Alle tests
python -m unittest discover -s tests -p "test_*.py"

# Specifieke test-module
python -m unittest tests.test_unit_core
python -m unittest tests.test_integration_vision_pipeline
```

Bestaande tests:
- `tests/test_unit_core.py` – Eenheidstests voor `AppState` en `InspectionEngine`
- `tests/test_integration_vision_pipeline.py` – Integratietests voor vision pipeline en resultaatopslag
- `tests/test_inspection_pipeline.py` – Legacy handmatige tests met validatie

---

*Ontwikkeld tijdens afstudeerstage bij MDE Automation, februari–juni 2026.*