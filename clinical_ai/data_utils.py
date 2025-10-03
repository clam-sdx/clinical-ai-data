import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# --- Helper to extract first A–Z from model output safely ---
_letter_pat = re.compile(r"[A-Z]")

# -------------------------------
# 1) Minimal "user context" editor
# -------------------------------

def edit_user_context(store: Dict[str, str], key: str, new_context: str) -> None:
    """
    Mutate a simple in-memory store to set/replace the user context string.
    Example: edit_user_context(context_store, key='default', new_context='...').
    """
    store[key] = new_context


# ------------------------------------------------------
# 2) Chat templating -> input_ids/attention_mask builder
# ------------------------------------------------------

def render_chat_inputs(
    tokenizer,
    user_instruction: str,
    user_context: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Build a chat-formatted prompt from (user_instruction, user_context).
    Returns input_ids, attention_mask on `device`, and the raw prompt `text`.
    """
    messages = [
        {"role": "user", "content": user_instruction},
        {"role": "user", "content": user_context},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # ensure assistant turn begins next
        )
        enc = tokenizer(text, return_tensors="pt")
    else:
        text = f"<|user|>\n{user_instruction}\n<|user|>\n{user_context}\n<|assistant|>\n"
        enc = tokenizer(text, return_tensors="pt")

    # Move tensors to device and attach the raw text
    enc = {k: v.to(device) for k, v in enc.items()}
    enc["text"] = text  # <-- include the actual prompt text
    return enc


def _extract_first_letter(text: str) -> str:
    if not text:
        return ""
    m = _letter_pat.search(text.upper())
    return m.group(0) if m else ""


def predict_letters_batched(prompts, gen, tokenizer, pad_id, batch_size: int = 32):
    predictions = []
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            # HF text-generation returns a list (per input) of length num_return_sequences
            # so shape is: List[List[{"generated_text": str}]]
            outputs = gen(
                batch,
                max_new_tokens=1,
                do_sample=False,
                num_return_sequences=1,
                return_full_text=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Normalize to list-of-dicts regardless of single/multi input behavior
            # When passing a list, `outputs` is a list of lists; when a single str, it's a list
            # Here `batch` is always a list, so `outputs[j][0]["generated_text"]` is correct.
            for out in outputs:
                gen_text = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
                predictions.append(_extract_first_letter(gen_text))
    return predictions


# ------------------------------------------------------------------
# ------------------------------------------------------------------

def encode_target_tokens(tokenizer, target: str) -> List[int]:
    """
    Encode the target string WITHOUT adding special tokens.
    NOTE: For many LLM tokenizers, a leading space can matter:
      target='A0' vs target=' A0' may tokenize differently.
    Adjust the target string upstream if you need a leading space.
    """
    return tokenizer.encode(target, add_special_tokens=False)


@torch.no_grad()
def next_token_string_probability(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target: str,
) -> List[float]:
    """
    Compute P(first |target| generated tokens == target | prompt) per example.
    Uses the chain rule:
        Π_t p(target_t | prompt + target_<t)
    by incrementally feeding the target tokens and using past_key_values.

    Args:
        model: causal LM (HF Transformers) with .generate/.forward
        tokenizer: matching tokenizer
        input_ids: (batch, seq_len) prompt ids
        attention_mask: (batch, seq_len) mask
        target: the exact string to score (can be multi-token)

    Returns:
        probs: list of floats (length = batch) with the sequence probability.
    """
    model.eval()
    device = input_ids.device
    target_ids = encode_target_tokens(tokenizer, target)
    if len(target_ids) == 0:
        # Empty target: by definition probability 1.0
        return [1.0] * input_ids.size(0)

    batch_size = input_ids.size(0)
    # We'll compute per-sample to keep it simple & robust across models.
    probs: List[float] = []

    for b in range(batch_size):
        ids = input_ids[b:b+1]
        mask = attention_mask[b:b+1]

        # Initial forward pass to get logits for the NEXT token
        outputs = model(input_ids=ids, attention_mask=mask, use_cache=True)
        past = outputs.past_key_values  # speeds up incremental decoding

        seq_prob = 1.0
        for t_idx, tgt_id in enumerate(target_ids):
            # logits for the next position are in outputs.logits[:, -1, :]
            logits_next = outputs.logits[:, -1, :]  # (1, vocab)
            p_next = F.softmax(logits_next, dim=-1)[0, tgt_id].item()
            seq_prob *= float(p_next)

            # teacher-force the target token for the next step
            next_token = torch.tensor([[tgt_id]], device=device)
            outputs = model(input_ids=next_token, use_cache=True, past_key_values=past)
            past = outputs.past_key_values

        probs.append(seq_prob)

    return probs

@torch.no_grad()
def batch_target_prefix_probability(
    model,
    tokenizer,
    user_instruction: str,
    user_contexts: List[str],
    target: str,
    device: Optional[torch.device] = None,
) -> Tuple[List[float], float]:
    """
    Build prompts using `render_chat_inputs(user_instruction, user_context)` for each context,
    then compute P(first generated tokens == `target`) per item and the batch average.

    Args:
        model: HF causal LM
        tokenizer: matching tokenizer
        user_instruction: shared instruction string placed in a 'user' message
        user_contexts: list of user-context strings (one per example)
        target: exact string to score (watch leading spaces per tokenizer)
        device: torch device; defaults to model's device

    Returns:
        (probs, avg_prob): list of per-example probabilities and their average
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Ensure a pad token exists for batching
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))

    # Encode each prompt using the provided renderer
    encoded = [
        render_chat_inputs(
            tokenizer=tokenizer,
            user_instruction=user_instruction,
            user_context=ctx,
            device=device,
        )
        for ctx in user_contexts
    ]

    # Pad into a single batch
    input_id_list = [e["input_ids"][0] for e in encoded]          # (seq,)
    mask_list     = [e["attention_mask"][0] for e in encoded]     # (seq,)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        mask_list,
        batch_first=True,
        padding_value=0,
    )

    # Compute probabilities
    probs = next_token_string_probability(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target=target,
    )

    avg = float(sum(probs) / max(1, len(probs)))
    return probs, avg

