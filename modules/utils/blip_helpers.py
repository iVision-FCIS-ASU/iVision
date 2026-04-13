import numpy as np
import numpy.typing as npt
import onnxruntime as ort


def log_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def apply_repetition_penalty(logits, tokens, penalty):
    PAD_TOKEN_ID = 0

    for t in set(tokens[0].tolist()):
        if t == PAD_TOKEN_ID:
            continue
        if logits[0, t] < 0:
            logits[0, t] *= penalty
        else:
            logits[0, t] /= penalty
    return logits


def get_banned_tokens(tokens, ngram_size):
    tokens_list = tokens[0].tolist()

    if len(tokens_list) < ngram_size:
        return []

    ngrams = {}
    for i in range(len(tokens_list) - ngram_size + 1):
        prefix = tuple(tokens_list[i:i + ngram_size - 1])
        nxt = tokens_list[i + ngram_size - 1]
        ngrams.setdefault(prefix, []).append(nxt)

    current_prefix = tuple(tokens_list[-(ngram_size - 1):])
    return ngrams.get(current_prefix, [])


def normalize_score(score, length, alpha=1.0):
    return score / (((5 + length) / 6) ** alpha)


def beam_search_onnx(
    text_decoder_onnx: ort.InferenceSession,
    encoder_hidden_states: npt.NDArray,
    prompt_input_ids: npt.NDArray,
    prompt_attention_mask: npt.NDArray,
    bos_token_id: int,
    eos_token_id: int,
    max_length=50,
    min_length=20,
    beam_size=8,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,
    length_penalty=1.0,
    early_stopping=True,
    num_return_sequences=2,
):
    input_ids = prompt_input_ids.copy()
    input_ids[:, 0] = bos_token_id
    input_ids = input_ids[:, :-1]
    attention_mask = prompt_attention_mask.copy()
    attention_mask = attention_mask[:, :-1]
    
    beams = [
        (input_ids, attention_mask, 0.0)
    ] + [
        (input_ids, attention_mask, -1e9)
        for _ in range(beam_size - 1)
    ]

    completed = []

    for step in range(max_length):
        vocab_size = None

        all_scores = []
        all_tokens = []
        all_masks = []

        for beam_idx, (tokens, mask, score) in enumerate(beams):

            if tokens[0, -1] == eos_token_id:
                # do NOT expand finished beams
                all_scores.append(np.full((1, vocab_size), -1e9) if vocab_size else None)
                all_tokens.append(tokens)
                all_masks.append(mask)
                continue

            logits = text_decoder_onnx.run(
                None,
                {
                    "input_ids": tokens.astype(np.int64),
                    "attention_mask": mask.astype(np.int64),
                    "encoder_hidden_states": encoder_hidden_states,
                }
            )[0]

            next_logits = logits[:, -1, :]

            if vocab_size is None:
                vocab_size = next_logits.shape[-1]

            next_logits = apply_repetition_penalty(
                next_logits.copy(), tokens, repetition_penalty
            )

            banned = get_banned_tokens(tokens, no_repeat_ngram_size)
            for t in banned:
                next_logits[0, t] = -1e9

            if step < min_length:
                next_logits[0, eos_token_id] = -1e9

            log_probs = log_softmax(next_logits.astype(np.float32), axis=-1)

            total_scores = score + log_probs[0]  # (vocab,)

            all_scores.append(total_scores)
            all_tokens.append(tokens)
            all_masks.append(mask)

        # stack → (beam_size, vocab)
        all_scores = np.stack(all_scores, axis=0)

        # 🔥 global top-k
        beams_to_keep = 2 * beam_size
        flat_scores = all_scores.reshape(-1)

        topk = np.argpartition(flat_scores, -beams_to_keep)[-beams_to_keep:]
        topk = topk[np.argsort(flat_scores[topk])[::-1]]

        beam_indices = topk // vocab_size
        token_indices = topk % vocab_size


        new_beams = []
        new_completed = []

        for b_idx, t_idx in zip(beam_indices, token_indices):
            tokens = all_tokens[b_idx]
            mask = all_masks[b_idx]
            score = all_scores[b_idx, t_idx]

            new_tokens = np.concatenate([tokens, [[t_idx]]], axis=1)
            new_mask = np.concatenate([mask, [[1]]], axis=1)

            if t_idx == eos_token_id:
                new_completed.append((new_tokens, new_mask, float(score)))
            else:
                new_beams.append((new_tokens, new_mask, float(score)))


        # add new completed
        completed.extend(new_completed)

        # merge BOTH pools
        merged = completed + new_beams

        # sort globally
        merged = sorted(
            merged,
            key=lambda x: normalize_score(
                x[2],
                x[0].shape[1],
                length_penalty
            ),
            reverse=True
        )

        # rebuild beams + completed
        beams = []
        completed = []

        for tokens, mask, score in merged:
            if tokens[0, -1] == eos_token_id:
                completed.append((tokens, mask, score))
            else:
                if len(beams) < beam_size:
                    beams.append((tokens, mask, score))


        if early_stopping and len(completed) >= beam_size:
            break

    candidates = completed + beams

    sorted_beams = sorted(
        candidates,
        key=lambda x: normalize_score(x[2], x[0].shape[1], length_penalty),
        reverse=True
    )

    return [b[0] for b in sorted_beams[:num_return_sequences]]
