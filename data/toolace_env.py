#!/usr/bin/env python3
import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch

from swift.utils import get_logger

logger = get_logger()


class ToolACEStaticEnv:

    def __init__(self):
        self.tool_responses: List[str] = []
        self.current_step: int = 0
        self.system_prompt: str = ''
        self.initial_messages: List[Dict[str, str]] = []

    def reset(self, data_item: Dict[str, Any]) -> List[Dict[str, str]]:
        messages = data_item.get('messages', [])

        tool_responses_str = data_item.get('tool_responses', '[]')
        if isinstance(tool_responses_str, str):
            self.tool_responses = json.loads(tool_responses_str)
        else:
            self.tool_responses = list(tool_responses_str)

        self.current_step = 0

        self.initial_messages = []
        for message in messages:
            role = message.get('role', '')
            if role == 'system':
                self.initial_messages.append(deepcopy(message))
            elif role == 'user':
                self.initial_messages.append(deepcopy(message))
                break

        return deepcopy(self.initial_messages)

    def step(self, assistant_response: str) -> Tuple[Optional[str], bool]:
        if self.current_step < len(self.tool_responses):
            tool_response = self.tool_responses[self.current_step]
            self.current_step += 1
            done = self.current_step >= len(self.tool_responses)
            return tool_response, done
        else:
            return None, True

    @property
    def total_steps(self) -> int:
        return len(self.tool_responses)

    @property
    def remaining_steps(self) -> int:
        return max(0, len(self.tool_responses) - self.current_step)


class ToolACEMultiTurnManager:

    def __init__(self, max_turns: int = 10):
        """Args:
            max_turns:"""
        self.env = ToolACEStaticEnv()
        self.max_turns = max_turns

    def generate_multi_turn_sequence(
        self,
        data_item: Dict[str, Any],
        model,
        tokenizer,
        generation_config,
        template,
        device: torch.device,
        max_completion_length: int = 2048,
    ) -> Dict[str, Any]:
        initial_messages = self.env.reset(data_item)
        current_messages = deepcopy(initial_messages)

        all_token_ids = []
        all_loss_mask = []

        current_turn = 0

        while current_turn < self.max_turns:
            current_turn += 1

            prompt_encoded = self._encode_messages(
                current_messages, tokenizer, template, device, prompt_only=True
            )
            if prompt_encoded is None:
                break

            prompt_input_ids = prompt_encoded['input_ids']
            prompt_length = prompt_input_ids.shape[1]

            if current_turn == 1:
                all_token_ids.extend(prompt_input_ids[0].tolist())
                all_loss_mask.extend([0] * prompt_length)

            with torch.no_grad():
                model_inputs = {k: v for k, v in prompt_encoded.items() if k != 'labels'}
                model_inputs.pop('position_ids', None)
                model_inputs.pop('text_position_ids', None)

                generated_outputs = model.generate(
                    **model_inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )
                generated_tokens = generated_outputs.sequences

                if not getattr(template, 'skip_prompt', False):
                    new_token_ids = generated_tokens[0][prompt_length:].tolist()
                else:
                    new_token_ids = generated_tokens[0].tolist()

            assistant_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

            all_token_ids.extend(new_token_ids)
            all_loss_mask.extend([1] * len(new_token_ids))

            current_messages.append({'role': 'assistant', 'content': assistant_text})

            tool_response, done = self.env.step(assistant_text)

            if tool_response is not None:
                current_messages.append({'role': 'tool', 'content': tool_response})

                tool_token_ids = tokenizer.encode(tool_response, add_special_tokens=False)
                all_token_ids.extend(tool_token_ids)
                all_loss_mask.extend([0] * len(tool_token_ids))

                if not done:
                    next_user_msg = self._get_next_user_message(data_item, current_turn)
                    if next_user_msg:
                        current_messages.append({'role': 'user', 'content': next_user_msg})
                        user_token_ids = tokenizer.encode(next_user_msg, add_special_tokens=False)
                        all_token_ids.extend(user_token_ids)
                        all_loss_mask.extend([0] * len(user_token_ids))
                    else:
                        break
                else:
                    break
            else:
                break

        input_ids = torch.tensor([all_token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        loss_mask_tensor = torch.tensor([all_loss_mask], dtype=torch.long, device=device)
        labels[loss_mask_tensor == 0] = -100

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = -100
            attention_mask[input_ids == pad_token_id] = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'messages': current_messages,
        }

    def _get_next_user_message(self, data_item: Dict[str, Any], current_turn: int) -> Optional[str]:
        messages = data_item.get('messages', [])
        user_messages = [m for m in messages if m.get('role') == 'user']

        next_user_index = current_turn
        if next_user_index < len(user_messages):
            return user_messages[next_user_index].get('content', '')
        return None

    def _encode_messages(
        self,
        messages: List[Dict[str, str]],
        tokenizer,
        template,
        device: torch.device,
        prompt_only: bool = True,
    ) -> Optional[Dict[str, torch.Tensor]]:
        try:
            if prompt_only:
                encode_messages = deepcopy(messages)
                if encode_messages and encode_messages[-1].get('role') == 'assistant':
                    encode_messages[-1]['content'] = None

            data = {'messages': encode_messages}
            encoded = template.encode(data, return_length=True)

            from swift.utils import to_device
            batch = template.data_collator([encoded])
            batch = to_device(batch, device)
            return batch
        except Exception as error:
            logger.warning(f'Failed to encode messages: {error}')
            return None

def split_multi_turn_to_single_turns(data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = data_item.get('messages', [])
    if not messages:
        return [data_item]

    sub_samples = []

    for turn_idx, message in enumerate(messages):
        role = message.get('role', '')
        if role != 'assistant':
            continue

        sub_messages = deepcopy(messages[:turn_idx + 1])

        sub_item = deepcopy(data_item)
        sub_item['messages'] = sub_messages

        sub_samples.append(sub_item)

    if not sub_samples:
        return [data_item]

    return sub_samples


def build_toolace_off_policy_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    tool_role_token_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    if tool_role_token_ids is None:
        tool_markers = ['<|im_start|>tool\n', '<|start_header_id|>tool<|end_header_id|>\n']
        tool_end_markers = ['<|im_end|>', '<|eot_id|>']

        for marker in tool_markers:
            encoded = tokenizer.encode(marker, add_special_tokens=False)
            if encoded:
                tool_role_token_ids = encoded
                break

        if tool_role_token_ids is None:
            logger.warning_once(
                'Cannot detect tool role token markers. '
                'Tool responses will not be masked in loss computation.'
            )
            return labels

    tool_end_token_ids = None
    for marker in ['<|im_end|>', '<|eot_id|>']:
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if encoded:
            tool_end_token_ids = encoded
            break

    if tool_end_token_ids is None:
        return labels

    marker_length = len(tool_role_token_ids)
    end_length = len(tool_end_token_ids)
    marker_tensor = torch.tensor(tool_role_token_ids, device=input_ids.device, dtype=input_ids.dtype)
    end_tensor = torch.tensor(tool_end_token_ids, device=input_ids.device, dtype=input_ids.dtype)

    modified_labels = labels.clone()
    batch_size, seq_len = input_ids.shape

    for batch_idx in range(batch_size):
        sequence = input_ids[batch_idx]
        in_tool_response = False
        position = 0

        while position < seq_len:
            if not in_tool_response:
                if position + marker_length <= seq_len:
                    candidate = sequence[position:position + marker_length]
                    if torch.equal(candidate, marker_tensor):
                        in_tool_response = True
                        modified_labels[batch_idx, position:position + marker_length] = -100
                        position += marker_length
                        continue
                position += 1
            else:
                modified_labels[batch_idx, position] = -100

                if position + end_length <= seq_len:
                    candidate = sequence[position:position + end_length]
                    if torch.equal(candidate, end_tensor):
                        modified_labels[batch_idx, position:position + end_length] = -100
                        position += end_length
                        in_tool_response = False
                        continue
                position += 1

    return modified_labels
