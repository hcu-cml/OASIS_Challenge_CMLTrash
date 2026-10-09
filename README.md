# Roadside Trash Agent
<img width="1920" height="1080" alt="Screenshot (223)" src="https://github.com/user-attachments/assets/fc73842d-b169-4862-9a50-72984933da14" />

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

## Option 1: Offline Demo (Docker)

To try the agent without a GPU, API tokens, or Ollama, there is a prebuilt image with 6,024 cached images from Kaliganj, Gazipur (Dhaka Division), Bangladesh, along with the LLM screening output and SAM3 masks for them.

```bash
# 1. Run the demo container in the background (Docker assigns a free host port)
docker run --rm -d -P --name demo-app socialgood/trash-agent-demo:latest

# 2. Check the assigned port
docker port demo-app
```

That prints something like `5006/tcp -> 0.0.0.0:32768`. Open the mapped port, for example http://localhost:32768, and click Run Survey.

To stop the container:
```bash
docker stop demo-app
```

---

## Option 2: Live Mode

Live mode fetches current Mapillary imagery for any city or drawn box, screens it against your local Ollama instance, and runs SAM3 for real.

### 1. Prerequisites

- Python 3.10+ (3.12 or 3.13 recommended; 3.14 works)
- An NVIDIA GPU with current drivers. SAM3 falls back to CPU, but not fast enough for a full survey.
- A Mapillary client token, free from [Mapillary Developers](https://www.mapillary.com/dashboard/developers)
- A HuggingFace token (`HF_TOKEN`) with approved access to [`facebook/sam3`](https://huggingface.co/facebook/sam3), used for the segmentation and redaction weights
- Ollama, running with `gemma4:31b`

---

### 2. Setting Up Ollama & Gemma 4

1. Install [Ollama](https://ollama.com/).
2. Pull the vision model:
   ```bash
   ollama pull gemma4:31b
   ```
   Check the tag with `ollama list` and use exactly what it reports for `MODEL_VISION`. The value is passed to Ollama as-is, so a bare `gemma4` fails with a 404 when only `gemma4:31b` is installed.
3. Leave Ollama running, either via `ollama serve` or the system service. It serves an OpenAI-compatible API at `http://localhost:11434/v1`.

If Ollama runs on another machine on your LAN, start it with `OLLAMA_HOST=0.0.0.0:11434` and point `OLLAMA_BASE_URL` at that host.

---

### 3. Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/hcu-cml/OASIS_Challenge_CMLTrash.git
   cd OASIS_Challenge_CMLTrash
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This pulls the SAM3 stack (`torch`, `torchvision`, `transformers`, `opencv-python-headless`), so expect around 3 GB of CUDA wheels.

4. Copy the config template and fill in your tokens:
   ```bash
   cp .env.example .env
   ```
   See [Configuration](#configuration-env) for what each key does. You can also enter the tokens in the web UI instead.

---

### 4. Running the Application

1. Start the web app:
   ```bash
   python -m app.main serve --port 5006
   ```

2. Open http://localhost:5006.

3. If you would rather not edit `.env`, use the Configure button in the top-right of the dashboard. It takes the Mapillary token, `HF_TOKEN`, and Ollama base URL, and applies them to the next run.

4. To check your environment, tokens, and model endpoint from the terminal:
   ```bash
   python -m app.main doctor
   ```

> The server has no authentication. Anyone who can reach it can start runs that spend your Mapillary quota and GPU time, so keep it on localhost or behind a proxy.

---

### 5. Alternative: Run with Docker (Live Mode)

You can run the app in a container and still use live Ollama inference on the host. Pull `gemma4:31b` on the host first, as described in [Section 2](#2-setting-up-ollama--gemma-4) — fetching a model that size inside a container is not worth the wait.

Beyond the [prerequisites](#1-prerequisites) above, this needs Docker with Compose v2 and [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

1. Confirm Ollama is up on the host with the model ready:
   ```bash
   ollama list          # should show gemma4:31b
   ollama serve         # if not already running as a system service
   ```

2. Put your tokens in `.env` (see [Section 3](#3-installation--setup)).

3. Build and start the app container:
   ```bash
   docker compose up --build -d app
   ```
   Naming `app` starts only that service, not the bundled `ollama` one, so the container talks to the Ollama on your host.

4. Open http://localhost:5050.

5. Follow the logs:
   ```bash
   docker compose logs -f app
   ```

6. Stop it:
   ```bash
   docker compose down
   ```

#### Setting `OLLAMA_BASE_URL`

Inside a container, `localhost` means the container itself, so the value that works on the host does not work here. Pick the row that matches how you started things:

| How you run it | `OLLAMA_BASE_URL` in `.env` |
| --- | --- |
| Native Python (no Docker) | `http://localhost:11434/v1` |
| Only the `app` container, Ollama on the host | `http://host.docker.internal:11434/v1` |
| Both containers (`docker compose up`) | leave unset or comment it out |

If the container cannot reach Ollama it exits with `Ollama did not become reachable`. Check that Ollama listens on all interfaces (`OLLAMA_HOST=0.0.0.0:11434 ollama serve`) and that the value above matches your setup.

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and fill in your values:

| Key | What it does |
| --- | --- |
| `LLM_PROFILE` | Provider to use.`ollama` for live runs |
| `OLLAMA_BASE_URL` | Where Ollama listens. See the table above when running in Docker. |
| `MODEL_VISION` | Model tag for screening, exactly as `ollama list` reports it. |
| `MAPILLARY_TOKEN` | Imagery access. Needed for live runs, not for replay. |
| `HF_TOKEN` | Downloads the `facebook/sam3` weights to `weights/sam3/` on first live run. |

`MAPILLARY_TOKEN`, `HF_TOKEN`, and `OLLAMA_BASE_URL` can also be set from the Configure button in the dashboard, which takes effect on the next run.

### Optional: PostGIS time series

Runs are saved to PostGIS when `DATABASE_URL` is set. Docker Compose starts the `db` service and sets it for you from `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB`.

Detections are grouped into fixed square cells (`GRID_CELL_M`, default 250 m) that line up across runs, and bucketed by photo capture date. Each image is counted once, and only approved detections count.

```
GET /api/timeseries?cell_m=250&bucket=month&bbox=west,south,east,north
```

uses a rate rather than a raw count so results stay comparable when image coverage changes between runs.

---

## How to Use the Web Dashboard

1. Search for an area by name in the top bar, for example `Kaliganj, Gazipur` or `Mirpur, Dhaka`. Or click Draw box and drag a bounding box on the map.
2. Click Run Survey to start.
3. Each node reports its status in the left sidebar, with a live count of screened images. The Workflow Graph modal shows the LangGraph pipeline.
4. When the vision and segmentation models disagree, the review modal opens on its own. Toggle between the redacted image and the green mask overlay, then choose Approve mask or Exclude.
5. Click any detection on the map to see its redacted image and mask, or Open Gallery for all approved findings. The CityGML file downloads from the summary card.

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
│   ├── main.py            # CLI entrypoint (serve, run, doctor, record)
│   ├── perception.py      # SAM3 segmentation & privacy redaction
│   ├── sources.py         # Mapillary & Nominatim API clients
│   └── web/               # Flask server and UI templates
├── fixtures/golden/       # Reference export used to diff against new runs
├── Dockerfile             # App image for live mode
├── docker-compose.yml     # App + Ollama services, with GPU reservations
├── Makefile               # Shortcuts for install, doctor, serve, run, docker
├── requirements.txt       # Python dependencies
└── .env.example           # Example configuration template
```


---

## License & Acknowledgements

- Street-level imagery provided by [Mapillary](https://www.mapillary.com/).
- Perception uses Google's Gemma 4 and Meta AI's SAM3.
