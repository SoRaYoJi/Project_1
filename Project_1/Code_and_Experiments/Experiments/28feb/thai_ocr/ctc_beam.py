# thai_ocr/ctc_beam.py
import torch

def ctc_beam_search(log_probs: torch.Tensor, tokenizer, beam_width: int = 10):
    """
    log_probs: [T, C]  (log softmax)
    returns: best decoded string
    """
    blank = tokenizer.blank_id

    beams = [(tuple(), 0.0)]  # (prefix, score)
    for t in range(log_probs.size(0)):
        new_beams = {}
        topk = torch.topk(log_probs[t], k=min(beam_width, log_probs.size(1)))
        for c, lp in zip(topk.indices.tolist(), topk.values.tolist()):
            for prefix, score in beams:
                if c == blank:
                    # stay
                    key = prefix
                    new_beams[key] = max(new_beams.get(key, -1e18), score + lp)
                else:
                    # append with CTC rule: collapse repeats later via tokenizer.decode_greedy
                    key = prefix + (c,)
                    new_beams[key] = max(new_beams.get(key, -1e18), score + lp)

        # keep best beams
        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(k, v) for k, v in beams]

    best_seq = beams[0][0] if beams else tuple()
    # reuse tokenizer greedy collapse to remove repeats/blanks
    return tokenizer.decode_greedy(list(best_seq))