import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.rotation_utils import random_orthogonal_matrix, hadamard_matrix, apply_rotation, is_power_of_two
import wandb
from src.wandb_config import WANDB_ENTITY, WANDB_PROJECT, WANDB_MODE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def rotated_forward(model, input_ids, R):
    """
    Apply rotation at embedding output (simple approximation of R1 idea)
    """
    with torch.no_grad():
        inputs_embeds = model.transformer.wte(input_ids)

        # rotate
        rotated = apply_rotation(inputs_embeds, R)

        # inverse rotate
        restored = apply_rotation(rotated, R.T)

        outputs = model(inputs_embeds=restored)

    return outputs.logits


def compare_outputs(logits1, logits2):
    diff = (logits1 - logits2).abs().max().item()
    rel = (logits1 - logits2).norm() / logits1.norm()
    return diff, rel.item()


if __name__ == "__main__":
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        mode=WANDB_MODE,
        name="baseline-forward"
    )

    model, tokenizer = load_model()

    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    dim = model.config.hidden_size

    # try different matrices
    R_rand = random_orthogonal_matrix(dim, device=DEVICE)
    
    use_hadamard = is_power_of_two(dim)
    if use_hadamard:
        R_had = hadamard_matrix(dim, device=DEVICE)
    else:
        R_had = None
        print(f"Skipping Hadamard: hidden size {dim} is not a power of 2")

        with torch.no_grad():
            baseline_logits = model(**inputs).logits

    logits_rand = rotated_forward(model, inputs["input_ids"], R_rand)
    diff_rand, rel_rand = compare_outputs(baseline_logits, logits_rand)

    wandb.log({
        "rand_max_diff": diff_rand,
        "rand_rel_error": rel_rand,
    })
    print("Random rotation diff:", diff_rand, rel_rand)

    if R_had is not None:
        logits_had = rotated_forward(model, inputs["input_ids"], R_had)
        diff_had, rel_had = compare_outputs(baseline_logits, logits_had)

        wandb.log({
            "had_max_diff": diff_had,
            "had_rel_error": rel_had,
        })
        print("Hadamard rotation diff:", diff_had, rel_had)
    
    run.finish()