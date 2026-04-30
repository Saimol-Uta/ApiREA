#!/usr/bin/env python3
"""
rae_server.py
Servidor FastAPI que recibe una lista de palabras, scrapea el DLE (RAE)
y devuelve un CSV listo para importar en Anki.

Uso:
    pip install fastapi uvicorn pyrae httpx
    uvicorn rae_server:app --host 0.0.0.0 --port 8765

Endpoints:
    POST /generar   — recibe palabras, inicia job en background
    GET  /estado    — consulta estado del job actual
    GET  /resultado — descarga el CSV cuando termina
"""

import asyncio
import csv
import io
import json
import time
import random
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

try:
    from pyrae import dle
    PYRAE_OK = True
except ImportError:
    PYRAE_OK = False
    print("AVISO: pyrae no instalado, instala con: pip install pyrae")

CACHE_FILE = Path("rae_cache.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

cache: dict = {}
job_state: dict = {
    "running": False,
    "total": 0,
    "done": 0,
    "errors": 0,
    "current_word": "",
    "output_file": None,
    "started_at": None,
    "finished_at": None,
}


def load_cache():
    global cache
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        print(f"Cache cargado: {len(cache)} palabras")


def save_cache():
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_cache()
    yield


app = FastAPI(title="RAE Anki Generator", lifespan=lifespan)


class WordEntry(BaseModel):
    word: str
    hint: str = ""


class GenerarRequest(BaseModel):
    entries: list[WordEntry]


def scrape_rae(word: str) -> dict:
    """
    Intenta obtener definición de la RAE via pyrae.
    Retorna dict con gram, defs o error.
    """
    if not PYRAE_OK:
        return {"error": "pyrae no instalado"}

    try:
        result = dle.search_by_word(word=word.lower().strip())
        if result is None or not result.definitions:
            return {"error": "no encontrada"}

        defs = []
        gram = ""

        for i, d in enumerate(result.definitions[:3]):
            definition_text = ""
            example_text = ""

            # pyrae devuelve objetos con atributo 'definition' y opcionalmente 'examples'
            if hasattr(d, "definition"):
                definition_text = str(d.definition).strip()
            elif hasattr(d, "text"):
                definition_text = str(d.text).strip()
            else:
                definition_text = str(d).strip()

            # Categoría gramatical viene en la primera definición normalmente
            if i == 0 and hasattr(d, "category"):
                gram = str(d.category).strip()

            # Ejemplos
            if hasattr(d, "examples") and d.examples:
                example_text = str(d.examples[0]).strip()

            if definition_text:
                defs.append({"def": definition_text, "ex": example_text})

        if not defs:
            return {"error": "sin definiciones"}

        return {"gram": gram, "defs": defs}

    except Exception as e:
        return {"error": str(e)}


def build_anki_back(gram: str, defs: list) -> str:
    parts = []
    if gram:
        parts.append(f"[{gram}]")
    for i, d in enumerate(defs, 1):
        parts.append(f"{i}. {d.get('def', '')}")
        if d.get("ex"):
            parts.append(f"   Ej: {d['ex']}")
    return "\n".join(parts)


def get_tag(gram: str) -> str:
    g = gram.lower() if gram else ""
    if "sustantivo" in g or "sust" in g:
        return "rae sustantivo"
    if "verbo" in g:
        return "rae verbo"
    if "adjetivo" in g or "adj" in g:
        return "rae adjetivo"
    if "adverbio" in g or "adv" in g:
        return "rae adverbio"
    return "rae"


def build_csv(results: list[dict]) -> str:
    output = io.StringIO()
    output.write("#separator:tab\n")
    output.write("#html:false\n")
    output.write("#notetype:Basic\n")
    output.write("#deck:Vocabulario RAE\n")
    writer = csv.writer(output, delimiter="\t", quoting=csv.QUOTE_ALL)
    for r in results:
        front = r.get("word", "")
        back = r.get("back", "")
        tags = r.get("tags", "rae")
        writer.writerow([front, back, tags])
    return output.getvalue()


async def process_words(entries: list[WordEntry]):
    global job_state, cache

    job_state["running"] = True
    job_state["total"] = len(entries)
    job_state["done"] = 0
    job_state["errors"] = 0
    job_state["output_file"] = None
    job_state["started_at"] = datetime.now().isoformat()
    job_state["finished_at"] = None

    results = []

    for entry in entries:
        word = entry.word.strip()
        job_state["current_word"] = word

        # Revisar caché primero
        if word.lower() in cache:
            cached = cache[word.lower()]
            results.append(cached)
            job_state["done"] += 1
            print(f"[CACHE] {word}")
            continue

        # Scraping con delay aleatorio para ser amable con el servidor
        delay = random.uniform(1.5, 3.0)
        await asyncio.sleep(delay)

        data = scrape_rae(word)

        if "error" in data:
            row = {
                "word": word,
                "back": f"[No encontrada en RAE: {data['error']}]",
                "tags": "rae error",
                "error": data["error"],
            }
            job_state["errors"] += 1
            print(f"[ERR]   {word}: {data['error']}")
        else:
            gram = data.get("gram", "")
            defs = data.get("defs", [])
            back = build_anki_back(gram, defs)
            row = {
                "word": word,
                "back": back,
                "tags": get_tag(gram),
                "gram": gram,
                "error": None,
            }
            # Guardar en caché
            cache[word.lower()] = row
            save_cache()
            print(f"[OK]    {word} ({gram})")

        results.append(row)
        job_state["done"] += 1

    # Generar CSV
    csv_content = build_csv(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"anki_rae_{timestamp}.txt"
    filepath = OUTPUT_DIR / filename
    filepath.write_text(csv_content, encoding="utf-8")

    job_state["output_file"] = filename
    job_state["running"] = False
    job_state["finished_at"] = datetime.now().isoformat()

    ok = len(results) - job_state["errors"]
    print(f"\nListo: {ok}/{len(results)} tarjetas en {filename}")


@app.post("/generar")
async def generar(request: GenerarRequest, background_tasks: BackgroundTasks):
    if job_state["running"]:
        raise HTTPException(status_code=409, detail="Ya hay un job corriendo")
    if not request.entries:
        raise HTTPException(status_code=400, detail="Lista vacía")

    background_tasks.add_task(process_words, request.entries)

    return {
        "status": "iniciado",
        "total": len(request.entries),
        "mensaje": f"Procesando {len(request.entries)} palabras en background. Consulta /estado"
    }


@app.get("/estado")
async def estado():
    s = job_state.copy()
    if s["total"] > 0:
        s["progreso_pct"] = round((s["done"] / s["total"]) * 100, 1)
        if s["running"] and s["done"] > 0:
            elapsed = (datetime.now() - datetime.fromisoformat(s["started_at"])).seconds
            rate = s["done"] / elapsed if elapsed > 0 else 0
            remaining = (s["total"] - s["done"]) / rate if rate > 0 else 0
            s["tiempo_restante_min"] = round(remaining / 60, 1)
    return s


@app.get("/resultado")
async def resultado():
    if job_state["running"]:
        raise HTTPException(status_code=202, detail="Job todavía en proceso")
    if not job_state["output_file"]:
        raise HTTPException(status_code=404, detail="No hay resultado disponible")

    filepath = OUTPUT_DIR / job_state["output_file"]
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    content = filepath.read_text(encoding="utf-8")
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={job_state['output_file']}"}
    )


@app.get("/cache/stats")
async def cache_stats():
    return {"palabras_en_cache": len(cache), "archivo": str(CACHE_FILE)}


@app.delete("/cache")
async def clear_cache():
    global cache
    cache = {}
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    return {"status": "cache limpiado"}


@app.get("/health")
async def health():
    return {"status": "ok", "pyrae": PYRAE_OK, "cache_size": len(cache)}
