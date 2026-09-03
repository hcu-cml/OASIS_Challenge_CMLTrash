# Roadside Trash Agent

An autonomous, privacy-first GeoAI agent that detects, segments, and maps roadside waste accumulations from street-level imagery. 

The agent pairs Vision-Language Models (**Gemma 4**) with instance segmentation (**SAM3**) in a verified human-in-the-loop workflow.

---

## What the Agent Does

1. **Acquires Imagery**: Resolves any place name or user-drawn bounding box and audits available street-level imagery via the Mapillary API.
2. **Multimodal Screening**: Analyzes street scenes with Gemma 4 to identify trash accumulations, debris piles, and generate targeted segmentation prompts.
3. **Instance Segmentation**: Runs SAM3 to produce pixel-precise instance masks for detected waste.
4. **Privacy-First Redaction**: Identifies human subjects in candidate frames and strongly blurs/redacts faces and persons **before any data is published or mapped**.
5. **Human-in-the-Loop Gate**: Pauses and presents an interactive review card whenever models disagree (e.g. uncertain detections or count discrepancies).
6. **Open Geospatial Output**: Exports human-verified sightings directly into **CityGML 2.0** (`GenericCityObject`) and interactive map layers.

---

## Option 1: Instant Offline Demo (Docker)

If you just want to test and review the agent without setting up GPU dependencies, API tokens, or Ollama, we provide a pre-baked Docker demo containing **6,024 cached images from Dhaka, Bangladesh**, along with full LLM screening outputs and SAM3 masks.

```bash
# 1. Run the demo container in the background (Docker assigns a free host port)
docker run --rm -d -P --name demo-app socialgood/trash-agent-demo:latest

# 2. Check the assigned port
docker port demo-app
```

*Example output:* `5006/tcp -> 0.0.0.0:32768`  
Open **http://localhost:32768** in your browser and click **Run Survey** to explore the complete workflow!

To stop the container:
```bash
docker stop demo-app
```

---

## Option 2: Live Mode (Survey Any Location Worldwide)

In live mode, the agent fetches real-time Mapillary imagery for any city or drawn bounding box, queries your local Ollama instance with Gemma 4, and runs live SAM3 segmentation.

### 1. Prerequisites

- **Python 3.10+** (Python 3.12 or 3.13 recommended)
- **Mapillary Client Token** (free developer token from [Mapillary Developers](https://www.mapillary.com/dashboard/developers))
- **HuggingFace Token (`HF_TOKEN`)** with approved access to [`facebook/sam3`](https://huggingface.co/facebook/sam3) for SAM3 segmentation & privacy redaction weights
- **Ollama** installed and running with `gemma4`

---

### 2. Setting Up Ollama & Gemma 4

1. Install [Ollama](https://ollama.com/).
2. Pull the Gemma 4 vision model:
   ```bash
   ollama pull gemma4
   ```
3. Make sure Ollama is running (`ollama serve` or via the background system service). By default, Ollama serves its OpenAI-compatible API at `http://localhost:11434/v1`.

> **Remote Ollama Server**: If Ollama is running on another machine or GPU server on your LAN, start Ollama with `OLLAMA_HOST=0.0.0.0:11434` and set the URL in the configuration below.

---

### 3. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hcu-cml/OASIS_Challenge_CMLTrash.git
   cd OASIS_Challenge_CMLTrash
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your environment:**
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your keys (or enter them directly in the Web UI later):
   ```ini
   LLM_PROFILE=ollama
   OLLAMA_BASE_URL=http://localhost:11434/v1
   MODEL_VISION=gemma4
   MAPILLARY_TOKEN=MLY|your_mapillary_token_here
   HF_TOKEN=hf_your_huggingface_token_here
   ```
   *(When `HF_TOKEN` is provided, SAM3 model weights are automatically downloaded to `weights/sam3/` on the first live run).*

---

### 4. Running the Application

1. **Start the web application:**
   ```bash
   python -m app.main serve --port 5006
   ```

2. **Open the Dashboard:**
   Navigate to **http://localhost:5006** in your browser.

3. **Configure Interactively (Top-Right Button):**
   You don't need to manually edit `.env` if you prefer not to. Simply click the **Configure** button in the top-right corner of the dashboard to enter or reveal:
   - **Mapillary Token**
   - **HuggingFace Token (`HF_TOKEN`)**
   - **Ollama Base URL**
   
   Settings saved in the UI take effect immediately for new runs.

4. **Run a Health Check (Optional CLI):**
   You can also verify your environment, API tokens, and model endpoints from your terminal:
   ```bash
   python -m app.main doctor
   ```

---

### 5. Alternative: Run with Docker (Live Mode)

If you prefer to run the application in a Docker container while still using live Ollama inference, follow the steps below. **Ollama and the Gemma 4 model must be installed on the host machine first** — pulling a ~20 GB model inside a container is impractically slow.

#### Prerequisites

- **NVIDIA GPU** with up-to-date drivers
- **Docker** (with Compose v2) and [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- **Ollama installed on the host** with `gemma4` already pulled (see [Section 2](#2-setting-up-ollama--gemma-4) above)
- **Mapillary Token** and **HuggingFace Token** in your `.env` file (see [Section 3](#3-installation--setup))

#### Steps

1. **Verify Ollama is running on the host with gemma4 ready:**
   ```bash
   ollama list          # should show gemma4
   ollama serve         # if not already running as a system service
   ```

2. **Set your tokens in `.env`:**
   ```bash
   cp .env.example .env
   # Edit .env and fill in MAPILLARY_TOKEN and HF_TOKEN
   ```

3. **Build and start the app container:**
   ```bash
   docker compose up --build -d app
   ```
   > This starts **only** the `app` service (not the `ollama` service from the compose file). The app container connects to your host's Ollama instance.

4. **Open the dashboard** at **http://localhost:5050**.

5. **Follow logs:**
   ```bash
   docker compose logs -f app
   ```

6. **Stop the container:**
   ```bash
   docker compose down
   ```

> **Note:** By default, the app container reaches the host's Ollama via the Docker internal network. If you encounter connection issues, ensure Ollama is listening on all interfaces (`OLLAMA_HOST=0.0.0.0:11434 ollama serve`) and set `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1` in your `.env` file.

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and fill in your values:

```ini
LLM_PROFILE=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
MODEL_VISION=gemma4
MAPILLARY_TOKEN=
HF_TOKEN=
```

> **In-App Configuration**: You can also view, enter, or update `MAPILLARY_TOKEN`, `HF_TOKEN`, and `OLLAMA_BASE_URL` directly in the web app by clicking the **Configure** button in the top-right corner of the dashboard.

---

## How to Use the Web Dashboard

1. **Search or Draw an Area**:
   - Type a location name in the top search bar (e.g. `Hazaribagh, Dhaka`, `Altona, Hamburg`), or
   - Click **Draw box** to drag a custom bounding box directly on the map.
2. **Start Survey**: Click **Run Survey**. The agent will start executing the workflow graph.
3. **Live Progress Tracking**:
   - Follow each node's status in the left sidebar and open the **Workflow Graph** modal to inspect the LangGraph pipeline.
   - Watch the image screening counter in real-time.
4. **Human-in-the-Loop Review**:
   - If the vision model and segmentation model have disagreements, the review modal opens automatically.
   - Toggle between the sanitized base image and the green overlay, and click **Approve mask** or **Exclude**.
5. **View Detections & Export**:
   - Click any detection point on the map to preview the sanitized image and mask.
   - Click **Open Gallery** to inspect all approved findings.
   - Download the generated **CityGML 2.0** file directly from the summary card.

---

## Project Structure

```
├── app/
│   ├── agent/             # Agent logic, LangGraph workflow, LLM provider & nodes
│   │   ├── graph.py       # Linear workflow definition with SQLite checkpointer
│   │   ├── llm.py         # OpenAI-compatible provider with Ollama integration
│   │   ├── nodes.py       # Workflow execution nodes (AOI, screening, review, export)
│   │   └── schemas.py     # Pydantic data schemas
│   ├── citygml.py         # CityGML 2.0 XML generator
│   ├── geo.py             # Geospatial helpers & coordinate transformations
│   ├── main.py            # CLI entrypoint (serve, run, doctor)
│   ├── perception.py      # SAM3 segmentation & privacy redaction
│   ├── sources.py         # Mapillary & Nominatim API clients
│   └── web/               # Flask server and UI templates
├── fixtures/              # Pre-recorded offline demo data (cache, masks, LLM outputs)
├── Dockerfile.demo        # Dockerfile for the self-contained offline demo
├── requirements.txt       # Python dependencies
└── .env.example           # Example configuration template
```

---

## License & Acknowledgements

- Street-level imagery provided by **[Mapillary](https://www.mapillary.com/)**.
- Multimodal perception powered by **Google Gemma 4** and Meta AI's **SAM3**.
