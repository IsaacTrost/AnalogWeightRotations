from src.hardware_configs import supported_hardware_presets, build_rpu_config

print("presets:", supported_hardware_presets())
for preset in ["ideal_analog", "ir_drop_only", "weight_noise_only", "quant_only", "full_stack"]:
    cfg = build_rpu_config(preset)
    print(preset, type(cfg).__name__)