import hashlib
import json
from pathlib import Path

def cache_path_for(
        cache_dir: str | Path,
        key_data: dict[str, object],
) -> Path:
    serialized_key = json.dumps(
        key_data,
        ensure_ascii=False,
        sort_keys=True,
    )

    digest = hashlib.sha256(
        serialized_key.encode('utf-8')
    ).hexdigest()

    return Path(cache_dir) / f"{digest}.json"