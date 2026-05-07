#!/usr/bin/env python3
"""ToolACE dataset registration and preprocessing for MAD-OPD.

The ToolACE corpus (Team-ACE/ToolACE on HuggingFace) stores each example as a
single multi-turn tool-use conversation.  For OPAD (Paper §4.4) we supervise
one assistant turn at a time, so this preprocessor splits every example into
per-turn sub-samples that share a ``_toolace_group_id``.  The
``ToolACEGroupedBatchSampler`` in ``mad_opd.trainers.mad_opd_trainer`` keeps
sub-samples of the same group contiguous inside a batch so the student sees
the whole trajectory in order.

The dataset file path is configured at runtime via the environment variable
``TOOLACE_DATASET_PATH`` (local ``.jsonl``) or via ``--dataset`` on the
``ms-swift`` CLI after calling ``register_toolace()`` once.

ToolACE raw format::

    {"system": "...",
     "conversations": [
        {"from": "user",      "value": "..."},
        {"from": "assistant", "value": "[FuncCall(param=value)]"},
        {"from": "tool",      "value": "[{\\"name\\": ..., \\"results\\": {...}}]"},
        {"from": "assistant", "value": "Here are the results..."},
        ...]}

After preprocessing each sub-sample is a standard chat::

    {"messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "[FuncCall(param=value)]"}],
     "tool_responses":        "[...]",
     "tool_call_indices":     "[...]",
     "_toolace_group_id":     "42",
     "_toolace_turn_index":   0,
     "_toolace_total_turns":  4}
"""
import hashlib
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Union

from swift.dataset import DatasetMeta, ResponsePreprocessor, register_dataset
from swift.utils import get_logger

logger = get_logger()

_toolace_global_group_counter = 0


class ToolACEPreprocessor(ResponsePreprocessor):
    """Split a multi-turn ToolACE example into per-turn sub-samples."""

    def preprocess(self, row: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        global _toolace_global_group_counter

        messages = []

        system_content = row.get('system', '')
        if system_content:
            messages.append({'role': 'system', 'content': system_content})

        tool_responses: List[str] = []
        tool_call_indices: List[int] = []

        conversations = row.get('conversations', [])
        for conversation in conversations:
            role_from = conversation.get('from', '')
            value = conversation.get('value', '')

            if role_from == 'user':
                messages.append({'role': 'user', 'content': value})
            elif role_from == 'assistant':
                current_index = len(messages)
                messages.append({'role': 'assistant', 'content': value})
                # Heuristic: a leading `[` + `(` indicates a tool-call turn.
                stripped_value = value.strip()
                if stripped_value.startswith('[') and '(' in stripped_value:
                    tool_call_indices.append(current_index)
            elif role_from == 'tool':
                messages.append({'role': 'tool', 'content': value})
                tool_responses.append(value)

        tool_responses_json = json.dumps(tool_responses, ensure_ascii=False)
        tool_call_indices_json = json.dumps(tool_call_indices)

        assistant_indices = [
            idx for idx, msg in enumerate(messages)
            if msg.get('role') == 'assistant'
        ]

        if len(assistant_indices) <= 1:
            result = {
                'messages': messages,
                'tool_responses': tool_responses_json,
                'tool_call_indices': tool_call_indices_json,
                '_toolace_group_id': str(_toolace_global_group_counter),
                '_toolace_turn_index': 0,
                '_toolace_total_turns': max(len(assistant_indices), 1),
            }
            _toolace_global_group_counter += 1
            return result

        group_id = str(_toolace_global_group_counter)
        _toolace_global_group_counter += 1
        total_turns = len(assistant_indices)

        sub_samples = []
        for turn_index, assistant_idx in enumerate(assistant_indices):
            sub_messages = deepcopy(messages[:assistant_idx + 1])
            sub_samples.append({
                'messages': sub_messages,
                'tool_responses': tool_responses_json,
                'tool_call_indices': tool_call_indices_json,
                '_toolace_group_id': group_id,
                '_toolace_turn_index': turn_index,
                '_toolace_total_turns': total_turns,
            })

        return sub_samples


def _resolve_dataset_path() -> str:
    """Resolve the ToolACE path from env / conventional locations.

    Accepts both ``.jsonl`` (one JSON object per line) and ``.json`` (a single
    JSON array) — the latter is the native HuggingFace download format and is
    what ``Team-ACE/ToolACE`` ships as.  ms-swift's dataset loader handles both.

    Priority:
    1. ``$TOOLACE_DATASET_PATH``                              (explicit override)
    2. ``./data_cache/toolace/data.jsonl``                     (download-script default)
    3. ``./data_cache/toolace/data.json``                      (HF raw array default)
    4. Sibling ``toolace_data/`` directory, if present         (legacy layout)
    """
    candidates = [
        os.environ.get('TOOLACE_DATASET_PATH', ''),
        './data_cache/toolace/data.jsonl',
        './data_cache/toolace/data.json',
        os.path.join(os.path.dirname(__file__), 'toolace_data', 'data.jsonl'),
        os.path.join(os.path.dirname(__file__), 'toolace_data', 'data.json'),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ''


def register_toolace(dataset_path: str = '') -> None:
    """Register the ToolACE dataset with ms-swift.

    Args:
        dataset_path: optional explicit path to the ToolACE ``.jsonl``.  If
            empty, resolves via ``_resolve_dataset_path``.
    """
    path = dataset_path or _resolve_dataset_path()
    register_dataset(DatasetMeta(
        ms_dataset_id=None,
        hf_dataset_id='Team-ACE/ToolACE',
        dataset_name='toolace',
        preprocess_func=ToolACEPreprocessor(),
        dataset_path=path or None,
        split=['train'],
    ))


# Register on import so `--custom_register_path .../toolace_dataset.py` works.
register_toolace()


# Extend standard_keys so ms-swift does not strip our per-turn metadata.
from swift.dataset.preprocessor.core import RowPreprocessor

_toolace_extra_keys = [
    '_toolace_group_id',
    '_toolace_turn_index',
    '_toolace_total_turns',
    'tool_responses',
    'tool_call_indices',
]
for _key in _toolace_extra_keys:
    if _key not in RowPreprocessor.standard_keys:
        RowPreprocessor.standard_keys.append(_key)

logger.info('[mad_opd] ToolACE dataset registered; use `--dataset toolace`.')
