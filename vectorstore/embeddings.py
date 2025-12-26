import os
from typing import Any, List, Optional


def get_embedding(text: str, *, task_type: str) -> Optional[List[float]]:
    api_key = (os.getenv('GEMINI_API_KEY', '') or '').strip()
    if not api_key:
        return None

    model_name = (os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001') or 'gemini-embedding-001').strip()

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    tt = (task_type or '').strip()
    if tt and tt.upper() == tt:
        tt_norm = tt
    else:
        tt_norm = tt.upper()

    out: Any = client.models.embed_content(
        model=model_name,
        contents=(text or '').strip(),
        config=types.EmbedContentConfig(task_type=tt_norm) if tt_norm else None,
    )

    try:
        embs = getattr(out, 'embeddings', None)
        if embs and isinstance(embs, list) and len(embs) > 0:
            vals = getattr(embs[0], 'values', None)
            if isinstance(vals, list):
                return [float(x) for x in vals]
    except Exception:
        pass

    return None
