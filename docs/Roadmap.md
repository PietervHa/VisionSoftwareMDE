# Roadmap – VisionSoftwareMDE

Dit document beschrijft bekende verbeterpunten en mogelijke uitbreidingen. Items zijn verdeeld in **nu op te lossen** (kleine bugs of veiligheidsrisico's) en **toekomstige doorontwikkeling** (features voor na de oplevering).

---

## Nu op te lossen (voor of direct na oplevering)

### Beveiliging – API-sleutel in config
**Probleem:** De Roboflow API-sleutel (`SwXK94LMoGhQfZsk8xCG`) staat als plaintext in `config/default.yaml` en daarmee in de Git-repository.  
**Oplossing:** Sleutel verplaatsen naar een omgevingsvariabele (`ROBOFLOW_API_KEY`) en in de config verwijzen met `${ROBOFLOW_API_KEY}`. Voeg `.env` toe aan `.gitignore`.

### Beveiliging – Maintenance-wachtwoord in config
**Probleem:** Het maintenance-wachtwoord (`@Welkom01`) staat als plaintext in de config.  
**Oplossing:** Zelfde aanpak als de API-sleutel: omgevingsvariabele of aparte secrets-file buiten de repository.

### Bug – Foutlogging bij camerafout (FTC 45)
**Probleem:** Wanneer de camera wegvalt tijdens een inspectiecyclus, wordt de fout niet altijd correct gelogd naar `data/logs/`.  
**Oplossing:** Afvangen in `camera.py` met expliciete error-logging en een duidelijke ERROR-status in het inspectieresultaat.

---

## Toekomstige doorontwikkeling (roadmap)

### 1. Stabiliteitstest uitvoeren (NFR2)
De 8-uur durende stabiliteitstest (FTC vereiste) is niet uitgevoerd voor oplevering. Aanbevolen: systeem aansluiten op de productieline en minimaal één volledige werkdag zonder handmatige interventie laten draaien. Logbestanden monitoren op geheugenlekken of frame-drop.

### 2. Migratie naar FastAPI
Het webframework Flask (synchroon) kan onder hoge polling-frequentie een bottleneck vormen. FastAPI (asynchroon, ASGI) kan doorvoer en responsiviteit van het dashboard verbeteren, met name bij hogere TCP-trigger-frequenties.  
Proof of concept is al overwogen tijdens de realisatiefase; de architectuur leent zich goed voor deze migratie.

### 3. Dataset uitbreiden voor de classifier
De huidige classifier is getraind op een relatief kleine dataset van OK- en defective-producten van één producttype. Bij introductie van nieuwe producttypes of verpakkingsvarianten moet de dataset worden uitgebreid en het model opnieuw worden getraind.  
`tools/capture_dataset.py` en `tools/train_classifier.py` zijn hiervoor al aanwezig.

### 4. Databaseopslag voor inspectieresultaten
Inspectieresultaten worden nu opgeslagen als JSONL-bestanden per dag in `data/results/`. Voor langetermijn-analyses en koppeling met een MES of ERP is opslag in een database (bijv. SQLite of PostgreSQL) beter geschikt.  
`backend/output/result_writer.py` is het aanknopingspunt voor deze uitbreiding.

### 5. Automatische modelselectie op basis van producttype
Op dit moment wordt het detectiebackend handmatig ingesteld in de config. Als MDE Automation meerdere producttypes gaat inspecteren, is automatische modelselectie op basis van een productnummer (uit het TCP-trigger-signaal of een barcode) een logische stap.

### 6. Exportfunctie in het dashboard
Het analytics-dashboard toont resultaten per dag. Een exportknop (CSV of PDF) zou het eenvoudiger maken om rapportages te genereren voor kwaliteitscontroles of klantcommunicatie.

### 7. GPU-versnelling inschakelen (CUDA)
Het systeem draait momenteel op CPU. Op een IPC met een NVIDIA GPU kan CUDA worden ingeschakeld voor de classifier en YOLO, wat de cyclustijd met 50–70% kan reduceren. Dit vereist installatie van CUDA-toolkit en de juiste PyTorch-versie met GPU-ondersteuning.

---

*Laatste update: 5 juni 2026 – Pieter van Haaften*