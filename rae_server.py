#!/usr/bin/env python3
"""
rae_server.py
Servidor FastAPI que recibe una lista de palabras, scrapea el DLE (RAE)
y devuelve un CSV listo para importar en Anki.

Uso:
    pip install fastapi uvicorn httpx beautifulsoup4
    uvicorn rae_server:app --host 0.0.0.0 --port 8765

Endpoints:
    POST /generar   — recibe palabras, inicia job en background
    GET  /estado    — consulta estado del job actual
    GET  /resultado — descarga el CSV cuando termina
    GET  /debug/{palabra} — descarga HTML crudo para inspección
"""

import asyncio
import csv
import io
import json
import random
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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

# Headers que imitan navegador Linux real
RAE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
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


# ---------------------------------------------------------------------------
# SCRAPING — basado en estructura real del DLE (verificada con debug HTML)
#
# El DLE moderno usa esta estructura:
#
#   <ol class="c-definitions">
#     <li id="..." class="j">                    ← acepción principal
#       <div class="c-definitions__item" role="definition">
#         <div>
#           <abbr class="d" title="nombre femenino">f.</abbr>
#           Pasividad, desinterés, falta de voluntad.
#           <span class="h">ejemplo en cursiva</span>   ← ejemplo (si existe)
#         </div>
#       </div>
#     </li>
#     <li class="j2"> ... </li>                  ← acepciones secundarias
#   </ol>
#
# La categoría gramatical está en <abbr class="d" title="nombre femenino">
# Los ejemplos están en <span class="h"> (color morado en la web)
# Los sinónimos están en <span class="sin"> (no los queremos en la definición)
# ---------------------------------------------------------------------------

def scrape_rae(word: str) -> dict:
    url = f"https://dle.rae.es/{word.lower().strip()}"

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=RAE_HEADERS)

        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}"}

        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # --- 1. Buscar el <ol class="c-definitions"> principal ---
        ol = soup.find("ol", class_="c-definitions")
        if not ol:
            _save_debug(word, html)
            return {"error": "sin lista de definiciones (HTML guardado para debug)"}

        defs = []
        gram = ""

        # --- 2. Iterar los <li class="j"> / <li class="j1"> / <li class="j2"> ---
        # Clases de acepciones: j, j1, j2, j3 (principales), m (locuciones)
        li_items = ol.find_all("li", class_=re.compile(r'^[jm]\d*$'))

        # Fallback: cualquier <li> con role="definition" o div.c-definitions__item
        if not li_items:
            li_items = ol.find_all("li")

        for i, li in enumerate(li_items[:3]):
            # Buscar el div de contenido de la definición
            def_div = li.find("div", attrs={"role": "definition"})
            if not def_div:
                def_div = li.find("div", class_="c-definitions__item")
            if not def_div:
                def_div = li  # fallback: usar el li directamente

            # Extraer categoría gramatical del primer ítem
            # <abbr class="d" title="nombre femenino">f.</abbr>
            if i == 0:
                abbr_gram = def_div.find("abbr", class_="d")
                if abbr_gram:
                    gram = abbr_gram.get("title", abbr_gram.get_text(strip=True))

            # Clonar para no mutilar el soup original
            work = BeautifulSoup(str(def_div), "html.parser")

            # Extraer ejemplo antes de limpiar
            # <span class="h"> = cursiva morada = ejemplo
            ejemplo = ""
            for ex_span in work.find_all("span", class_="h"):
                ejemplo = ex_span.get_text(strip=True)
                ex_span.decompose()
                break

            # Quitar elementos que ensucian el texto:
            # - <span class="n_acep">: número de acepción ("1.")
            # - <div class="c-definitions__item-footer">: sinónimos/antónimos
            # - <abbr class="sin-header-inline">: etiquetas Sin./Ant.
            for noise in work.find_all(class_=re.compile(
                r'n_acep|c-definitions__item-footer|c-word-list|sin-header-inline|sin\b'
            )):
                noise.decompose()

            # También quitar <abbr> de categoría gramatical del texto (ya la guardamos)
            for abbr in work.find_all("abbr", class_="d"):
                abbr.decompose()

            # Texto limpio
            texto = work.get_text(separator=" ", strip=True)
            texto = re.sub(r'\s+', ' ', texto)
            texto = re.sub(r'^\d+\.\s*', '', texto)  # quitar "1. " residual
            texto = texto.replace(" ,", ",").replace(" .", ".").strip()

            if texto and len(texto) > 3:
                defs.append({"def": texto, "ex": ejemplo})

        if not defs:
            _save_debug(word, html)
            return {"error": "definiciones vacías tras limpieza (HTML guardado para debug)"}

        return {"gram": gram, "defs": defs}

    except Exception as e:
        return {"error": str(e)}


def _save_debug(word: str, html: str):
    debug_path = OUTPUT_DIR / f"debug_{word.lower()}.html"
    debug_path.write_text(html, encoding="utf-8")
    print(f"[DEBUG] HTML guardado en {debug_path}")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

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
    if "sustantivo" in g or "sust" in g or re.search(r'\bf\b|\bm\b', g):
        return "rae sustantivo"
    if "verbo" in g or re.search(r'\btr\b|\bintr\b|\bprnl\b', g):
        return "rae verbo"
    if "adjetivo" in g or re.search(r'\badj\b', g):
        return "rae adjetivo"
    if "adverbio" in g or re.search(r'\badv\b', g):
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
        writer.writerow([r.get("word", ""), r.get("back", ""), r.get("tags", "rae")])
    return output.getvalue()


# ---------------------------------------------------------------------------
# PROCESAMIENTO EN BACKGROUND
# ---------------------------------------------------------------------------

async def process_words(entries: list[WordEntry]):
    global job_state, cache

    job_state.update({
        "running": True,
        "total": len(entries),
        "done": 0,
        "errors": 0,
        "output_file": None,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
    })

    results = []

    for entry in entries:
        word = entry.word.strip()
        job_state["current_word"] = word

        if word.lower() in cache:
            results.append(cache[word.lower()])
            job_state["done"] += 1
            print(f"[CACHE] {word}")
            continue

        await asyncio.sleep(random.uniform(1.5, 3.0))

        data = scrape_rae(word)

        if "error" in data:
            row = {
                "word": word,
                "back": f"[No encontrada: {data['error']}]",
                "tags": "rae error",
                "error": data["error"],
            }
            job_state["errors"] += 1
            print(f"[ERR]   {word}: {data['error']}")
        else:
            gram = data.get("gram", "")
            defs = data.get("defs", [])
            row = {
                "word": word,
                "back": build_anki_back(gram, defs),
                "tags": get_tag(gram),
                "gram": gram,
                "error": None,
            }
            cache[word.lower()] = row
            save_cache()
            print(f"[OK]    {word} ({gram})")

        results.append(row)
        job_state["done"] += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"anki_rae_{timestamp}.txt"
    (OUTPUT_DIR / filename).write_text(build_csv(results), encoding="utf-8")

    job_state["output_file"] = filename
    job_state["running"] = False
    job_state["finished_at"] = datetime.now().isoformat()

    ok = len(results) - job_state["errors"]
    print(f"\nListo: {ok}/{len(results)} tarjetas → {filename}")


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/generar")
async def generar(request: GenerarRequest, background_tasks: BackgroundTasks):
    if job_state["running"]:
        raise HTTPException(409, "Ya hay un job corriendo")
    if not request.entries:
        raise HTTPException(400, "Lista vacía")
    background_tasks.add_task(process_words, request.entries)
    return {"status": "iniciado", "total": len(request.entries)}


@app.get("/estado")
async def estado():
    s = job_state.copy()
    if s["total"] > 0:
        s["progreso_pct"] = round((s["done"] / s["total"]) * 100, 1)
        if s["running"] and s["done"] > 0 and s["started_at"]:
            elapsed = (datetime.now() - datetime.fromisoformat(s["started_at"])).seconds
            rate = s["done"] / max(elapsed, 1)
            s["tiempo_restante_min"] = round((s["total"] - s["done"]) / rate / 60, 1)
    return s


@app.get("/resultado")
async def resultado():
    if job_state["running"]:
        raise HTTPException(202, "Job todavía en proceso")
    if not job_state["output_file"]:
        raise HTTPException(404, "Sin resultado disponible")
    filepath = OUTPUT_DIR / job_state["output_file"]
    if not filepath.exists():
        raise HTTPException(404, "Archivo no encontrado en disco")
    content = filepath.read_text(encoding="utf-8")
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={job_state['output_file']}"},
    )


@app.get("/debug/{word}")
async def debug_html(word: str):
    debug_path = OUTPUT_DIR / f"debug_{word.lower()}.html"
    if not debug_path.exists():
        raise HTTPException(404, f"No hay debug guardado para '{word}'")
    content = debug_path.read_text(encoding="utf-8")
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename=debug_{word}.html"},
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
    return {"status": "ok", "cache_size": len(cache)}