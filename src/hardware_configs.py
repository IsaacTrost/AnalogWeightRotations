from __future__ import annotations

import copy
from typing import Callable, Dict, Optional


def _require_aihwkit():
    """
    Import AIHWKit lazily so float-only code paths still work without AIHWKit.
    """
    from aihwkit.simulator.configs import (
        InferenceRPUConfig,
        TorchInferenceRPUConfigIRDropT,
    )
    from aihwkit.simulator.parameters.enums import (
        WeightNoiseType,
        BoundManagementType,
        NoiseManagementType,
    )
    from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
    from aihwkit.inference.converter.conductance import SinglePairConductanceConverter

    return {
        "InferenceRPUConfig": InferenceRPUConfig,
        "TorchInferenceRPUConfigIRDropT": TorchInferenceRPUConfigIRDropT,
        "WeightNoiseType": WeightNoiseType,
        "BoundManagementType": BoundManagementType,
        "NoiseManagementType": NoiseManagementType,
        "PCMLikeNoiseModel": PCMLikeNoiseModel,
        "GlobalDriftCompensation": GlobalDriftCompensation,
        "SinglePairConductanceConverter": SinglePairConductanceConverter,
    }


def build_ideal_analog_config():
    """
    Ideal-ish AnalogLinear baseline.

    Uses AIHWKit AnalogLinear, but disables the explicit hardware losses we
    isolate in the other presets.
    """
    aihw = _require_aihwkit()

    cfg = aihw["InferenceRPUConfig"]()
    cfg.forward.ir_drop = 0.0
    cfg.forward.w_noise = 0.0
    cfg.forward.w_noise_type = aihw["WeightNoiseType"].NONE
    cfg.forward.inp_noise = 0.0
    cfg.forward.out_noise = 0.0
    cfg.forward.inp_res = -1.0
    cfg.forward.out_res = -1.0
    cfg.forward.out_bound = -1.0
    cfg.forward.bound_management = aihw["BoundManagementType"].NONE
    cfg.forward.noise_management = aihw["NoiseManagementType"].NONE
    return cfg
    


def build_ir_drop_only_config():
    """
    IR drop only.

    Matches cfg_irdrop_only() from 1layertests/explore_rotations.py.
    """
    aihw = _require_aihwkit()

    cfg = aihw["InferenceRPUConfig"]()
    cfg.forward.ir_drop = 1.0
    cfg.forward.w_noise = 0.0
    cfg.forward.w_noise_type = aihw["WeightNoiseType"].NONE
    cfg.forward.inp_noise = 0.0
    cfg.forward.out_noise = 0.0
    cfg.forward.inp_res = -1.0
    cfg.forward.out_res = -1.0
    cfg.forward.out_bound = -1.0
    cfg.forward.bound_management = aihw["BoundManagementType"].NONE
    cfg.forward.noise_management = aihw["NoiseManagementType"].NONE
    return cfg


def build_weight_noise_only_config():
    """
    Additive constant weight noise only.

    Matches cfg_w_noise_only() from 1layertests/explore_rotations.py.
    """
    aihw = _require_aihwkit()

    cfg = aihw["InferenceRPUConfig"]()
    cfg.forward.ir_drop = 0.0
    cfg.forward.w_noise = 0.02
    cfg.forward.w_noise_type = aihw["WeightNoiseType"].ADDITIVE_CONSTANT
    cfg.forward.inp_noise = 0.0
    cfg.forward.out_noise = 0.0
    cfg.forward.inp_res = -1.0
    cfg.forward.out_res = -1.0
    cfg.forward.out_bound = -1.0
    cfg.forward.bound_management = aihw["BoundManagementType"].NONE
    cfg.forward.noise_management = aihw["NoiseManagementType"].NONE
    return cfg


def build_quant_only_config():
    """
    8-bit DAC + ADC quantization only.

    Matches cfg_inp_quant() from 1layertests/explore_rotations.py.
    """
    aihw = _require_aihwkit()

    cfg = aihw["InferenceRPUConfig"]()
    cfg.forward.ir_drop = 0.0
    cfg.forward.w_noise = 0.0
    cfg.forward.w_noise_type = aihw["WeightNoiseType"].NONE
    cfg.forward.inp_noise = 0.0
    cfg.forward.out_noise = 0.0
    cfg.forward.inp_res = 2**8 - 2
    cfg.forward.out_res = 2**8 - 2
    cfg.forward.bound_management = aihw["BoundManagementType"].NONE
    cfg.forward.noise_management = aihw["NoiseManagementType"].NONE
    return cfg


def build_full_pcm_config():
    """
    Realistic PCM-style full stack.

    Includes PCM-like programming/read noise, IR drop, 10-bit DAC/ADC,
    and global drift compensation.

    Matches cfg_full_pcm() from 1layertests/explore_rotations.py.
    """
    aihw = _require_aihwkit()

    cfg = aihw["InferenceRPUConfig"]()
    cfg.noise_model = aihw["PCMLikeNoiseModel"](
        g_max=25.0,
        prog_noise_scale=1.0,
        read_noise_scale=1.0,
        drift_scale=0.0,
        g_converter=aihw["SinglePairConductanceConverter"](
            g_min=0.1,
            g_max=25.0,
        ),
    )
    cfg.forward.ir_drop = 0.5
    cfg.forward.w_noise = 0.0
    cfg.forward.inp_noise = 0.0
    cfg.forward.out_noise = 0.0
    cfg.forward.inp_res = 2**10 - 2
    cfg.forward.out_res = 2**10 - 2
    cfg.forward.bound_management = aihw["BoundManagementType"].NONE
    cfg.forward.noise_management = aihw["NoiseManagementType"].NONE
    cfg.drift_compensation = aihw["GlobalDriftCompensation"]()
    return cfg


def build_advanced_ir_drop_config():
    """
    Advanced time-dependent IR-drop config.

    Matches cfg_adv_irdrop() from 1layertests/explore_rotations.py.

    Warning:
        This can be extremely slow / memory-heavy for large layers.
    """
    aihw = _require_aihwkit()

    cfg = aihw["TorchInferenceRPUConfigIRDropT"]()
    cfg.forward.ir_drop = 1.0
    cfg.forward.ir_drop_segments = 4
    cfg.forward.ir_drop_v_read = 0.4
    cfg.forward.w_noise = 0.0
    cfg.forward.w_noise_type = aihw["WeightNoiseType"].NONE
    cfg.forward.inp_noise = 0.0
    cfg.forward.out_noise = 0.0
    cfg.forward.inp_res = 2**10 - 2
    cfg.forward.out_res = -1.0
    cfg.forward.out_bound = -1.0
    cfg.forward.bound_management = aihw["BoundManagementType"].NONE
    cfg.forward.noise_management = aihw["NoiseManagementType"].NONE
    return cfg


HARDWARE_PRESET_BUILDERS: Dict[str, Callable[[], object]] = {
    # Preferred new names
    "ideal_analog": build_ideal_analog_config,
    "ir_drop_only": build_ir_drop_only_config,
    "weight_noise_only": build_weight_noise_only_config,
    "quant_only": build_quant_only_config,
    "full_stack": build_full_pcm_config,
    "advanced_ir_drop": build_advanced_ir_drop_config,

    # Backward-compatible names from 1layertests/explore_rotations.py
    "irdrop_only": build_ir_drop_only_config,
    "w_noise_only": build_weight_noise_only_config,
    "inp_quant": build_quant_only_config,
    "full_pcm": build_full_pcm_config,
    "adv_irdrop": build_advanced_ir_drop_config,
}


HARDWARE_PRESETS_REQUIRING_PROGRAMMING = {
    "full_stack",
    "full_pcm",
}


def supported_hardware_presets() -> tuple[str, ...]:
    """Return the names accepted by build_rpu_config()."""
    return tuple(HARDWARE_PRESET_BUILDERS.keys())


def requires_program_analog_weights(hardware_preset: str) -> bool:
    """
    Return whether a preset needs layer.program_analog_weights() after loading weights.
    """
    return hardware_preset in HARDWARE_PRESETS_REQUIRING_PROGRAMMING


def build_rpu_config(
    hardware_preset: str = "ideal_analog",
    *,
    overrides: Optional[dict] = None,
):
    """
    Build a fresh AIHWKit RPU config for a named hardware preset.

    Always returns a deep copy so each AnalogLinear gets its own config object.
    """
    if hardware_preset not in HARDWARE_PRESET_BUILDERS:
        raise ValueError(
            f"Unknown hardware preset '{hardware_preset}'. "
            f"Supported presets: {', '.join(supported_hardware_presets())}"
        )

    config = HARDWARE_PRESET_BUILDERS[hardware_preset]()

    if overrides:
        _apply_overrides(config, overrides)

    return copy.deepcopy(config)


def _apply_overrides(config, overrides: dict) -> None:
    """
    Apply simple dotted-path overrides.

    Example:
        overrides = {
            "forward.inp_res": 256,
            "forward.out_res": 256,
        }
    """
    for dotted_key, value in overrides.items():
        target = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)