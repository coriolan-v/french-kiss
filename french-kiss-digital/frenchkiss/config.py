"""Versioned digital model parameters, shared by physics and training."""
import json
from pathlib import Path

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'simulation-config.json').read_text())
if CONFIG['schema']!=1:
    raise ValueError('Unsupported simulation configuration version')
