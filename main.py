import os
import re
import json
import uuid
import time
import logging
import pathlib
from typing import Any, Dict, List, Optional

import requests
import sympy as sp
from supabase import create_client
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qwen_agent.agents import Assistant
from qwen_agent.tools import BaseTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen-fismat")

# =========================
# Configuración general
# =========================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MAIN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
ROUTER_MODEL = os.getenv("QWEN_ROUTER_MODEL", "qwen-turbo")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
COLLECTION = os.getenv("QDRANT_COLLECTION", "qwen_fismat_rag")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if not DASHSCOPE_API_KEY:
    logger.warning("No se encontró DASHSCOPE_API_KEY.")

FALLBACK_MESSAGES: List[Dict[str, Any]] = []
FALLBACK_PROFILES: Dict[str, Dict[str, Any]] = {}
FALLBACK_EVENTS: List[Dict[str, Any]] = []

# =========================
# Clientes externos
# =========================

supabase = None
try:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase inicializado correctamente.")
except Exception as e:
    logger.error(f"Error inicializando Supabase: {e}")

qdrant = None
try:
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    if QDRANT_URL and QDRANT_API_KEY:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("Qdrant inicializado correctamente.")
except Exception as e:
    logger.error(f"Error inicializando Qdrant: {e}")

# =========================
# Utilidades
# =========================

def parse_params(params: Any) -> Dict[str, Any]:
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        try:
            return json.loads(params)
        except Exception:
            return {"raw": params}
    return {}


def hf_embed(texts: List[str]) -> List[List[float]]:
    if not HF_TOKEN:
        logger.warning("HF_TOKEN no configurado. RAG sin embeddings.")
        return []
    url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json={"inputs": texts}, timeout=120)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "embeddings" in data:
                    return data["embeddings"]
                return []
            if response.status_code == 503:
                time.sleep(8 + attempt * 5)
                continue
            return []
        except Exception as e:
            logger.error(f"Excepción en embeddings: {e}")
            time.sleep(5)
    return []


def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) <= max_chars:
            current += ("\n\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars])
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks


def ensure_collection(vector_size: int):
    if not qdrant:
        return
    try:
        collections = qdrant.get_collections().collections
        names = [c.name for c in collections]
    except Exception:
        names = []
    if COLLECTION not in names:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Colección {COLLECTION} creada con tamaño {vector_size}.")


def ingest_corpus():
    if not qdrant:
        logger.warning("Qdrant no configurado. Se omite ingesta RAG.")
        return
    corpus_dir = pathlib.Path("data/corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    points: List[PointStruct] = []
    for path in corpus_dir.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            if not chunks:
                continue
            vectors = hf_embed(chunks)
            if len(vectors) != len(chunks):
                continue
            if vectors:
                ensure_collection(len(vectors[0]))
                for chunk, vector in zip(chunks, vectors):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.name}:{chunk[:120]}"))
                    points.append(PointStruct(id=point_id, vector=vector, payload={"text": chunk, "source": path.name}))
        except Exception as e:
            logger.error(f"Error ingiriendo {path}: {e}")
    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points)
        logger.info(f"Se subieron {len(points)} puntos a Qdrant.")


def rag_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    if not qdrant:
        return {"note": "RAG no configurado."}
    vectors = hf_embed([query])
    if not vectors:
        return {"note": "Embeddings no disponibles."}
    try:
        results = qdrant.search(collection_name=COLLECTION, query_vector=vectors[0], limit=top_k)
        return {"results": [{"text": r.payload.get("text", ""), "source": r.payload.get("source", ""), "score": float(r.score)} for r in results]}
    except Exception as e:
        return {"error": str(e)}

# =========================
# Herramientas matemáticas (SymPy)
# =========================

def symbolic_solve(expression: str, symbols: str) -> Dict[str, Any]:
    try:
        sym = sp.symbols(symbols)
        expr = sp.sympify(expression)
        sols = sp.solve(expr, sym)
        if isinstance(sols, dict):
            return {"solutions": [f"{k}={v}" for k, v in sols.items()]}
        if isinstance(sols, list):
            return {"solutions": [str(s) for s in sols]}
        return {"solutions": [str(sols)]}
    except Exception as e:
        return {"error": str(e)}


def symbolic_integrate(expression: str, variable: str, lower=None, upper=None) -> Dict[str, Any]:
    try:
        x = sp.symbols(variable)
        expr = sp.sympify(expression)
        if lower is not None and upper is not None:
            result = sp.integrate(expr, (x, sp.sympify(lower), sp.sympify(upper)))
        else:
            result = sp.integrate(expr, x)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


def symbolic_diff(expression: str, variable: str, order: int = 1) -> Dict[str, Any]:
    try:
        x = sp.symbols(variable)
        expr = sp.sympify(expression)
        result = sp.diff(expr, x, int(order))
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


def numeric_evaluate(expression: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        expr = sp.sympify(expression)
        subs = {}
        if isinstance(variables, dict):
            for k, v in variables.items():
                subs[sp.symbols(str(k))] = sp.sympify(v)
        return {"result": str(expr.evalf(subs=subs))}
    except Exception as e:
        return {"error": str(e)}

# =========================
# DETECTOR AUTOMÁTICO DE MATEMÁTICAS
# =========================

def detect_math_intent(message: str) -> Optional[Dict[str, Any]]:
    """
    Detecta automáticamente si el mensaje contiene una solicitud matemática
    y devuelve la herramienta + parámetros a ejecutar.
    """
    msg_lower = message.lower()

    # Detectar ecuaciones: "resuelve", "soluciona", "= 0", "x^2"
    equation_patterns = [
        r'resuelve\s+(?:la\s+)?ecuaci[oó]n\s*(.+)',
        r'solucion[aá]\s+(?:la\s+)?ecuaci[oó]n\s*(.+)',
        r'(?:resuelve|calcula|halla)\s+(.+?)\s*=\s*(.+)',
    ]
    for pattern in equation_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            expr_text = match.group(0)
            # Intentar extraer la expresión
            eq_match = re.search(r'(.+?)\s*=\s*(.+)', expr_text)
            if eq_match:
                lhs = eq_match.group(1).strip()
                rhs = eq_match.group(2).strip()
                # Limpiar palabras como "resuelve", "la ecuación"
                lhs = re.sub(r'(?:resuelve|soluciona|calcula|halla|la|ecuación|ecuacion)\s*', '', lhs).strip()
                expression = f"{lhs} - ({rhs})"
                # Detectar variable
                var_match = re.search(r'[a-wyz]', expression)
                variable = var_match.group(0) if var_match else 'x'
                return {"tool": "symbolic_solve", "params": {"expression": expression, "symbols": variable}}

    # Detectar integrales: "integral de", "integrar", "∫"
    integral_patterns = [
        r'integral\s+(?:de\s+)?(.+?)(?:\s+(?:dx|dt|dy|dz|respecto|con))',
        r'integra[r]\s+(.+?)(?:\s+(?:dx|dt|dy|dz|respecto|con))',
        r'∫\s*(.+?)\s*d([a-z])',
        r'(?:calcula|resuelve|halla)\s+(?:la\s+)?integral\s+(?:de\s+)?(.+)',
    ]
    for pattern in integral_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            expr = match.group(1).strip()
            # Detectar variable de integración
            var_match = re.search(r'd([a-z])', msg_lower)
            variable = var_match.group(1) if var_match else 'x'
            return {"tool": "symbolic_integrate", "params": {"expression": expr, "variable": variable}}

    # Detectar derivadas: "derivada de", "derivar", "d/dx"
    derivative_patterns = [
        r'derivada\s+(?:de\s+)?(.+?)(?:\s+(?:respecto|con|en))',
        r'deriva[r]\s+(.+?)(?:\s+(?:respecto|con|en))',
        r'd/d([a-z])\s*(.+)',
        r'(?:calcula|resuelve|halla)\s+(?:la\s+)?derivada\s+(?:de\s+)?(.+)',
    ]
    for pattern in derivative_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            expr = match.group(1).strip() if match.lastindex and match.group(1) else match.group(0)
            var_match = re.search(r'd/d([a-z])', msg_lower) or re.search(r'respecto\s+(?:a\s+)?([a-z])', msg_lower)
            variable = var_match.group(1) if var_match else 'x'
            return {"tool": "symbolic_diff", "params": {"expression": expr, "variable": variable, "order": 1}}

    # Detectar evaluación numérica: "calcula el valor", "evalúa"
    eval_patterns = [
        r'(?:calcula|eval[uú]a|halla)\s+(?:el\s+valor\s+(?:de\s+|numérico\s+de\s+))?(.+)',
    ]
    for pattern in eval_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            expr = match.group(1).strip()
            # Solo si parece una expresión matemática
            if re.search(r'[\d\+\-\*\/\^\(\)]', expr) and len(expr) < 100:
                return {"tool": "numeric_evaluate", "params": {"expression": expr}}

    return None


def execute_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """Ejecuta la herramienta y devuelve el resultado como string."""
    if tool_name == "symbolic_solve":
        result = symbolic_solve(params.get("expression", ""), params.get("symbols", "x"))
    elif tool_name == "symbolic_integrate":
        result = symbolic_integrate(params.get("expression", ""), params.get("variable", "x"), params.get("lower"), params.get("upper"))
    elif tool_name == "symbolic_diff":
        result = symbolic_diff(params.get("expression", ""), params.get("variable", "x"), int(params.get("order", 1)))
    elif tool_name == "numeric_evaluate":
        result = numeric_evaluate(params.get("expression", ""), params.get("variables"))
    elif tool_name == "rag_search":
        result = rag_search(params.get("query", ""), int(params.get("top_k", 3)))
    else:
        result = {"error": f"Herramienta desconocida: {tool_name}"}
    return json.dumps(result, ensure_ascii=False)

# =========================
# Memoria persistente
# =========================

def save_message(session_id: str, role: str, content: str):
    if supabase:
        try:
            supabase.table("messages").insert({"session_id": session_id, "role": role, "content": content}).execute()
            return
        except Exception as e:
            logger.error(f"Error guardando mensaje: {e}")
    FALLBACK_MESSAGES.append({"session_id": session_id, "role": role, "content": content})


def recent_messages(session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
    if supabase:
        try:
            res = supabase.table("messages").select("role,content").eq("session_id", session_id).order("created_at", desc=True).limit(limit).execute()
            return list(reversed(res.data or []))
        except Exception as e:
            logger.error(f"Error leyendo mensajes: {e}")
    return [{"role": m["role"], "content": m["content"]} for m in FALLBACK_MESSAGES if m["session_id"] == session_id][-limit:]


def get_profile(session_id: str) -> Dict[str, Any]:
    if supabase:
        try:
            res = supabase.table("student_profile").select("*").eq("session_id", session_id).limit(1).execute()
            if res.data:
                return res.data[0]
        except Exception as e:
            logger.error(f"Error obteniendo perfil: {e}")
    return FALLBACK_PROFILES.get(session_id, {})


def upsert_profile(session_id: str, patch: Dict[str, Any]):
    if supabase:
        try:
            existing = get_profile(session_id)
            preferences = existing.get("preferences", {}) if isinstance(existing, dict) else {}
            if not isinstance(preferences, dict):
                preferences = {}
            preferences.update(patch)
            if existing:
                supabase.table("student_profile").update({"preferences": preferences}).eq("session_id", session_id).execute()
            else:
                supabase.table("student_profile").insert({"session_id": session_id, "preferences": preferences}).execute()
            return
        except Exception as e:
            logger.error(f"Error actualizando perfil: {e}")
    profile = FALLBACK_PROFILES.get(session_id, {})
    profile.update(patch)
    FALLBACK_PROFILES[session_id] = profile


def save_event(session_id: str, event_type: str, payload: Dict[str, Any]):
    if supabase:
        try:
            supabase.table("learning_events").insert({"session_id": session_id, "event_type": event_type, "payload": payload}).execute()
            return
        except Exception as e:
            logger.error(f"Error guardando evento: {e}")
    FALLBACK_EVENTS.append({"session_id": session_id, "event_type": event_type, "payload": payload})

# =========================
# Prompts de agentes
# =========================

MATH_CONTEXT_TEMPLATE = """
RESULTADO DE CÁLCULO SIMBÓLICO (ejecutado con SymPy):
Herramienta usada: {tool_name}
Parámetros: {params}
Resultado: {result}

Usa este resultado verificado para explicar la solución al estudiante de forma pedagógica.
NO digas que no puedes ejecutar código. El código YA fue ejecutado y el resultado está arriba.
Explica el procedimiento paso a paso usando este resultado.
"""

BASE_PROMPT = """
Eres un tutor experto de física y matemáticas universitarias.
IMPORTANTE: SIEMPRE completa tu respuesta entera. NUNCA la dejes a medias.
IMPORTANTE: NUNCA digas que no puedes ejecutar código o que no tienes herramientas. El sistema ejecuta cálculos automáticamente.
IMPORTANTE: Si se te proporciona un resultado de cálculo, úsalo para explicar la solución.
Usa formato Markdown. Usa LaTeX entre $...$ para fórmulas en línea y $$...$$ para ecuaciones de bloque.
Sé pedagógico, claro y completo.
"""

PROMPTS: Dict[str, str] = {
    "orchestrator": "Eres un orquestador. Devuelve JSON: {\"agent\": \"math\", \"confidence\": 0.9}. Agentes válidos: tutor, math, physics, solver, verifier, exercise, evaluator, librarian. No añadas texto fuera del JSON.",
    "tutor": BASE_PROMPT + "\nEres un tutor socrático. Guía con preguntas y explicaciones. No des la solución completa si el estudiante necesita aprender.",
    "planner": BASE_PROMPT + "\nEres un planificador curricular. Crea planes de estudio realistas y progresivos.",
    "math": BASE_PROMPT + "\nEres experto en matemáticas universitarias: cálculo, álgebra lineal, ecuaciones diferenciales, probabilidad.",
    "physics": BASE_PROMPT + "\nEres experto en física universitaria: mecánica, electromagnetismo, termodinámica, ondas.",
    "solver": BASE_PROMPT + "\nEres un resolvedor paso a paso. Descompón en pasos pequeños y pedagógicos.",
    "verifier": BASE_PROMPT + "\nEres un verificador. Comprueba resultados y explica errores.",
    "critic": BASE_PROMPT + "\nEres un crítico de errores conceptuales. Identifica errores con empatía.",
    "exercise": BASE_PROMPT + "\nGenera ejercicios adaptados al nivel del estudiante con solución y rúbrica.",
    "evaluator": BASE_PROMPT + "\nEvalúa respuestas con rúbricas. Valora el procedimiento.",
    "lab": BASE_PROMPT + "\nEres un agente de laboratorio computacional. Propón experimentos numéricos.",
    "librarian": BASE_PROMPT + "\nEres un bibliotecario académico. Busca y cita fuentes.",
    "memory": BASE_PROMPT + "\nResume el progreso y perfil del estudiante.",
    "metacognition": BASE_PROMPT + "\nAyuda al estudiante a reflexionar sobre su aprendizaje.",
    "security": BASE_PROMPT + "\nDetecta solicitudes inseguras.",
    "analytics": BASE_PROMPT + "\nAnaliza el progreso y detecta patrones.",
    "multilevel": BASE_PROMPT + "\nExplica en tres niveles: intuitivo, formal y aplicado.",
    "recovery": BASE_PROMPT + "\nSi hay incertidumbre, pide aclaración.",
}

# =========================
# Construcción de agentes
# =========================

QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

llm_main = {
    "model": MAIN_MODEL,
    "model_server": QWEN_API_BASE,
    "api_key": DASHSCOPE_API_KEY,
    "generate_cfg": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.9}
}

llm_router = {
    "model": ROUTER_MODEL,
    "model_server": QWEN_API_BASE,
    "api_key": DASHSCOPE_API_KEY,
    "generate_cfg": {"max_tokens": 512, "temperature": 0.1, "top_p": 0.9}
}


def build_agent(name: str, description: str, prompt: str, llm_cfg: Dict[str, Any]):
    try:
        return Assistant(
            name=name,
            description=description,
            llm=llm_cfg,
            system_message=prompt,
            function_list=[]
        )
    except Exception as e:
        logger.error(f"No se pudo construir el agente {name}: {e}")
        return None


AGENTS: Dict[str, Any] = {}
for agent_key, prompt in PROMPTS.items():
    if agent_key == "orchestrator":
        continue
    AGENTS[agent_key] = build_agent(
        name=agent_key,
        description=f"Agente {agent_key}",
        prompt=prompt,
        llm_cfg=llm_main
    )

ROUTER = build_agent(
    name="orchestrator",
    description="Orquestador",
    prompt=PROMPTS["orchestrator"],
    llm_cfg=llm_router
)

ALLOWED_AGENTS = [k for k in PROMPTS.keys() if k != "orchestrator"]

# =========================
# Ejecución de agentes
# =========================

def extract_text(event: Any) -> str:
    try:
        if isinstance(event, list) and event:
            last = event[-1]
            if isinstance(last, dict):
                return str(last.get("content") or last)
            return str(last)
        if isinstance(event, dict):
            return str(event.get("content") or event)
        return str(event)
    except Exception:
        return str(event)


def run_agent(agent: Any, messages: List[Dict[str, str]]) -> str:
    if agent is None:
        return "El agente no está disponible."
    final = ""
    try:
        for event in agent.run(messages=messages):
            text = extract_text(event)
            if text:
                final = text
    except Exception as e:
        logger.error(f"Error ejecutando agente: {e}")
        return f"Error del agente: {e}"
    return final.strip() or "El agente no devolvió respuesta."


def classify(message: str, context: str) -> str:
    if ROUTER is None:
        return "tutor"
    content = f"Contexto:\n{context}\n\nMensaje: {message}\nDevuelve JSON."
    raw = run_agent(ROUTER, [{"role": "user", "content": content}])
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            agent = data.get("agent", "tutor")
            if agent in ALLOWED_AGENTS:
                return agent
        except Exception:
            pass
    return "tutor"


def build_context(session_id: str) -> str:
    profile = get_profile(session_id)
    msgs = recent_messages(session_id, limit=6)
    lines = ["Contexto del estudiante:", json.dumps(profile, ensure_ascii=False), "", "Historial reciente:"]
    for m in msgs:
        lines.append(f"{m.get('role', 'user')}: {str(m.get('content', ''))[:200]}")
    return "\n".join(lines)

# =========================
# Orquestador principal
# =========================

MODE_MAP = {
    "Auto": "", "Tutoría": "tutor", "Plan": "planner", "Matemáticas": "math",
    "Física": "physics", "Resolver": "solver", "Verificar": "verifier",
    "Crítico": "critic", "Ejercicios": "exercise", "Evaluación": "evaluator",
    "Laboratorio": "lab", "Documentos": "librarian", "Memoria": "memory",
    "Metacognición": "metacognition", "Seguridad": "security",
    "Analítica": "analytics", "Multinivel": "multilevel", "Recuperación": "recovery",
}


class Orchestrator:
    def handle(self, session_id: str, message: str, mode: str) -> str:
        save_message(session_id, "user", message)
        context = build_context(session_id)

        agent_key = MODE_MAP.get(mode, "")
        if not agent_key:
            agent_key = classify(message, context)

        agent = AGENTS.get(agent_key, AGENTS.get("tutor"))

        # DETECCIÓN AUTOMÁTICA DE MATEMÁTICAS
        math_intent = detect_math_intent(message)
        tool_result_text = ""

        if math_intent:
            tool_name = math_intent["tool"]
            tool_params = math_intent["params"]
            tool_result = execute_tool(tool_name, tool_params)
            tool_result_text = MATH_CONTEXT_TEMPLATE.format(
                tool_name=tool_name,
                params=json.dumps(tool_params, ensure_ascii=False),
                result=tool_result
            )
            logger.info(f"Herramienta ejecutada automáticamente: {tool_name}")

        # Construir mensajes para el agente
        system_content = context
        if tool_result_text:
            system_content += "\n\n" + tool_result_text

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": message}
        ]

        answer = run_agent(agent, messages)

        save_message(session_id, "assistant", answer)
        save_event(session_id, "agent_response", {"agent": agent_key, "mode": mode, "tool_used": math_intent["tool"] if math_intent else None})

        return answer

# =========================
# Arranque
# =========================

orchestrator = Orchestrator()

# =========================
# API FastAPI
# =========================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="QwenFisMat Tutor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    mode: str = "Auto"
    message: str


@app.on_event("startup")
def startup_event():
    if os.getenv("RUN_INGEST", "1") == "1":
        try:
            ingest_corpus()
        except Exception as e:
            logger.error(f"No se pudo ingestar el corpus: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "service": "QwenFisMat Tutor"}


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    if not req.message.strip():
        return JSONResponse(status_code=400, content={"error": "El mensaje no puede estar vacío."})
    try:
        answer = orchestrator.handle(session_id, req.message, req.mode)
    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        answer = f"Error del sistema: {e}"
    return {"session_id": session_id, "answer": answer}


static_dir = pathlib.Path("static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Backend activo. Falta static/index.html."}
