"""Persist trained smoke artifacts before evaluation; never promote implicitly."""
import hashlib
import os
from pathlib import Path
import torch


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix+'.pending')
    with temporary.open('xb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    restored = torch.load(temporary, map_location='cpu', weights_only=False)
    for key, value in payload.get('state_dict', {}).items():
        assert torch.equal(restored['state_dict'][key], value.cpu()), key
    temporary.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.sha256').write_text(f'{digest}  {path.name}\n')
    return digest
