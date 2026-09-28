import hashlib
from collections.abc import Iterable

from groundtruth.schemas import Document


def documents_fingerprint(documents: Iterable[Document]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.id):
        digest.update(document.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
