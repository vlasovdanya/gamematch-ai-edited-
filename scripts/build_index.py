import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gamematch.catalog import Catalog  # noqa: E402

t0 = time.time()
c = Catalog()
print(f"Игр: {len(c.games)} · эмбеддер: {c.embedder.name} · индекс: {c.index.backend} · {time.time() - t0:.1f} с")
