"""Fixed-step character deletion (the paper's Step rule), as used by the character-level pipeline.

delete_chars_uniform deletes every floor(1/d)-th character of the text (spaces, newlines and punctuation
included), where d is the target deletion rate. When a single step size cannot hit d within the tolerance,
the deletions of two adjacent step sizes are mixed, and a final adjustment brings the deleted count into
[d - tolerance, d + tolerance] of the text length. The rule is deterministic: no random numbers are drawn.
"""


def delete_chars_uniform(text, target_deletion_rate, tolerance=0.02):
    """Delete characters at (mixed) fixed steps.

    Returns (corrupted_text, original_text, actual_deletion_rate).
    """
    if not isinstance(text, str):
        return text, text, 0.0

    text_len = len(text)
    if text_len == 0:
        return text, text, 0.0

    if target_deletion_rate <= 0:
        return text, text, 0.0

    if target_deletion_rate >= 1:
        return "", text, 1.0

    # target number of deleted characters
    target_delete_count = int(text_len * target_deletion_rate)

    # tolerance band
    min_delete = int(text_len * max(0, target_deletion_rate - tolerance))
    max_delete = int(text_len * min(1, target_deletion_rate + tolerance))
    max_delete = min(max_delete, text_len - 1)  # keep at least one character

    # two adjacent step sizes
    step_small = max(2, int(1 / target_deletion_rate))  # deletes slightly more
    step_large = step_small + 1                         # deletes slightly less

    # deletions produced by each step size alone
    count_small = len(range(step_small - 1, text_len, step_small))
    count_large = len(range(step_large - 1, text_len, step_large))

    # a single step size already falls inside the band
    if min_delete <= count_small <= max_delete:
        delete_indices = set(range(step_small - 1, text_len, step_small))
        corrupted_text = ''.join([ch for i, ch in enumerate(text) if i not in delete_indices])
        actual_rate = len(delete_indices) / text_len
        return corrupted_text, text, actual_rate

    if min_delete <= count_large <= max_delete:
        delete_indices = set(range(step_large - 1, text_len, step_large))
        corrupted_text = ''.join([ch for i, ch in enumerate(text) if i not in delete_indices])
        actual_rate = len(delete_indices) / text_len
        return corrupted_text, text, actual_rate

    # mix: start from step_large and add positions of step_small (or remove positions) as needed
    indices_large = set(range(step_large - 1, text_len, step_large))
    indices_small = set(range(step_small - 1, text_len, step_small))
    extra_needed = target_delete_count - len(indices_large)

    if extra_needed > 0:
        extra_candidates = sorted(list(indices_small - indices_large))
        if extra_candidates:
            if extra_needed >= len(extra_candidates):
                extra_indices = set(extra_candidates)
            else:
                step_extra = len(extra_candidates) / extra_needed
                extra_indices = set()
                for i in range(extra_needed):
                    idx = int(i * step_extra)
                    if idx < len(extra_candidates):
                        extra_indices.add(extra_candidates[idx])
            delete_indices = indices_large | extra_indices
        else:
            remaining = [i for i in range(text_len) if i not in indices_large]
            if remaining and extra_needed <= len(remaining):
                step_extra = len(remaining) / extra_needed
                extra_indices = set()
                for i in range(extra_needed):
                    idx = int(i * step_extra)
                    if idx < len(remaining):
                        extra_indices.add(remaining[idx])
                delete_indices = indices_large | extra_indices
            else:
                delete_indices = indices_large
    else:
        remove_needed = len(indices_large) - target_delete_count
        if remove_needed > 0:
            indices_list = sorted(list(indices_large))
            step_remove = len(indices_list) / remove_needed
            remove_indices = set()
            for i in range(remove_needed):
                idx = int(i * step_remove)
                if idx < len(indices_list):
                    remove_indices.add(indices_list[idx])
            delete_indices = indices_large - remove_indices
        else:
            delete_indices = indices_large

    # final adjustment into the tolerance band
    delete_count = len(delete_indices)
    if delete_count < min_delete:
        remaining = sorted([i for i in range(text_len) if i not in delete_indices])
        needed = min_delete - delete_count
        step_add = max(1, len(remaining) // needed) if needed > 0 else 1
        for i in range(0, len(remaining), step_add):
            if len(delete_indices) >= min_delete:
                break
            delete_indices.add(remaining[i])

    elif delete_count > max_delete:
        indices_list = sorted(list(delete_indices))
        needed = delete_count - max_delete
        step_remove = max(1, len(indices_list) // needed) if needed > 0 else 1
        for i in range(0, len(indices_list), step_remove):
            if len(delete_indices) <= max_delete:
                break
            delete_indices.discard(indices_list[i])

    corrupted_text = ''.join([ch for i, ch in enumerate(text) if i not in delete_indices])
    actual_rate = len(delete_indices) / text_len
    return corrupted_text, text, actual_rate
