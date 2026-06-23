"""
Optional fine-tuned reader model for SmartPaper AI.

Loads the Phi-3 Mini + QLoRA adapter trained on QASPER (Day 4),
pushed to HuggingFace Hub as a LoRA-only checkpoint. This is NOT used
by default - it's a toggle (use_finetuned) that can be switched on in
the demo to show the fine-tuning skill alongside the main Groq-backed
agent.

Loading happens lazily on first use, since Unsloth + a 4-bit model
load is slow (several seconds) and most requests won't need it.
"""

import config

_model = None
_tokenizer = None


def _load():
    """Lazily load the fine-tuned model on first call."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    from unsloth import FastLanguageModel

    print(f"[finetuned_reader] Loading {config.FINETUNED_MODEL_REPO} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.FINETUNED_MODEL_REPO,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    _model, _tokenizer = model, tokenizer
    print("[finetuned_reader] Loaded.")
    return _model, _tokenizer


def generate_answer(question: str, context: str, max_new_tokens: int = 150) -> str:
    """
    Generate an answer using the fine-tuned Phi-3 reader, given a
    question and a context string (e.g. retrieved text/table chunks
    joined together).
    """
    model, tokenizer = _load()

    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=["Question:"],
        tokenizer=tokenizer,
    )

    input_length = inputs["input_ids"].shape[1]
    generated = outputs[0][input_length:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    answer = answer.split("Question")[0].strip()
    return answer