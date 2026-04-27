import importlib
import importlib.util
import re
import sys
from pathlib import Path
from src.engine import Engine
from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import itertools


_CPYTHON_TAG_RE = re.compile(r"\.cpython-(\d+)(?:-|$)")


def _format_cpython_tag(tag):
    if len(tag) < 2:
        return f"CPython {tag}"
    if tag.startswith("3") and len(tag) >= 2:
        return f"CPython 3.{tag[1:]}"
    return f"CPython {tag[:-1]}.{tag[-1]}"


def _compiled_module_version_mismatch_message(path):
    package_name, _, module_name = path.rpartition(".")
    if not package_name:
        return None

    package_spec = importlib.util.find_spec(package_name)
    if package_spec is None or not package_spec.submodule_search_locations:
        return None

    candidates = []
    for location in package_spec.submodule_search_locations:
        candidates.extend(sorted(Path(location).glob(f"{module_name}.cpython-*.so")))

    if not candidates:
        return None

    candidate_tags = sorted(
        {
            match.group(1)
            for candidate in candidates
            for match in [_CPYTHON_TAG_RE.search(candidate.name)]
            if match is not None
        }
    )
    current_tag = f"{sys.version_info.major}{sys.version_info.minor}"
    if current_tag in candidate_tags:
        return None

    formatted_versions = ", ".join(_format_cpython_tag(tag) for tag in candidate_tags)
    binaries = ", ".join(candidate.name for candidate in candidates)
    return (
        f"{path} is only available as compiled extension(s) [{binaries}] for {formatted_versions}; "
        f"current interpreter is CPython {sys.version.split()[0]}. "
        "Run this baseline test with a matching Python interpreter."
    )

def load_players(config, verbose=False):
    assert "players" in config, "Config must have a 'players' key"
    assert isinstance(config["players"], list), "Players must be a list"

    if verbose:
        print("--- Importing Players ---")
    imported_players = []

    for i, player_conf in enumerate(config["players"]):
        try:
            path = player_conf["path"]
            cls_name = player_conf["class"]
            
            # 1. Import module
            module = importlib.import_module(path)
            
            # 2. Get Class
            cls = getattr(module, cls_name)
            
            # 3. Append class
            imported_players.append(cls)
            
            if verbose:
                print(f"Imported {cls.__name__} from {path}")
            
        except Exception as e:
            mismatch_message = _compiled_module_version_mismatch_message(path)
            if mismatch_message is not None:
                e = ImportError(mismatch_message)
            print(f"Failed to import {cls_name} from {path} for player {i}:  {e}")
            raise e
    
    return imported_players

def _normalize_player_entries(entries, is_baseline):
    normalized = []
    for p in entries:
        if isinstance(p, list):
            item = {
                "path": p[0],
                "class": p[1],
            }
            if len(p) > 2:
                item["args"] = p[2]
            if len(p) > 3:
                item["label"] = p[3]
        elif isinstance(p, dict):
            item = dict(p)
        else:
            raise ValueError(f"Invalid player config: {p}")
        item["is_baseline"] = is_baseline
        normalized.append(item)
    return normalized


def _preprocess_player_config(config):
    import copy
    config = copy.deepcopy(config)
    players = _normalize_player_entries(config.get("players", []), is_baseline=False)
    baselines = _normalize_player_entries(config.get("baselines", []), is_baseline=True)
    merged_players = players + baselines
    for i, p in enumerate(merged_players):
        p["player_id"] = i
    config["players"] = merged_players
    config["baselines"] = baselines
    return config
