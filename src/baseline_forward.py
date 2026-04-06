import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from src.wandb_config import WANDB_ENTITY, WANDB_PROJECT, WANDB_MODE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def get_activations(model, input_ids):
    activations = {}
    handles = []

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                for i, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        activations[f"{name}[{i}]"] = item.detach().cpu()
                        break
        return hook

    for name, module in model.named_modules():
        if "mlp.c_fc" in name:
            handles.append(module.register_forward_hook(hook_fn(name)))
            print(f"Hooked module: {name}")
            break

    with torch.no_grad():
        outputs = model(input_ids)

    for h in handles:
        h.remove()

    return outputs, activations


def run_baseline(model, tokenizer, text="Hello world"):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    outputs, activations = get_activations(model, inputs["input_ids"])

    logits = outputs.logits

    return {
        "logits": logits,
        "activations": activations,
    }


def log_to_wandb(results):
    wandb.log({
        "logits_norm": results["logits"].norm().item(),
    })


if __name__ == "__main__":
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        mode=WANDB_MODE,
        name="baseline-forward"
    )

    model, tokenizer = load_model()

    results = run_baseline(model, tokenizer)

    log_to_wandb(results)

    print("Baseline run complete.")
    run.finish()