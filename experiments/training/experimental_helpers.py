# Historical helper implementations extracted unchanged from rocket_ppa.py and train.py.

def get_backbone_hidden_size(backbone):
    """Resolve the last hidden-state width across common HF/PEFT config layouts."""
    configs = [getattr(backbone, "config", None)]
    if configs[0] is not None:
        configs.extend([
            getattr(configs[0], "text_config", None),
            getattr(configs[0], "base_config", None),
        ])
    base_model = getattr(backbone, "base_model", None)
    if base_model is not None:
        configs.append(getattr(base_model, "config", None))

    for config in configs:
        if config is None:
            continue
        for attr in ("hidden_size", "d_model", "n_embd"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)

    embeddings = backbone.get_input_embeddings()
    if hasattr(embeddings, "embedding_dim"):
        return int(embeddings.embedding_dim)
    if hasattr(embeddings, "weight"):
        return int(embeddings.weight.shape[-1])

    config_type = type(getattr(backbone, "config", None)).__name__
    raise AttributeError(f"Could not infer hidden size from backbone config {config_type}")

def resolve_lora_target_modules(model, target_names):
    """Target inner Linear modules when a model wraps projections."""
    module_names = {name for name, _ in model.named_modules()}
    resolved = []
    for target_name in target_names:
        wrapped_linear_name = f"{target_name}.linear"
        if any(name.endswith(wrapped_linear_name) for name in module_names):
            resolved.append(wrapped_linear_name)
        else:
            resolved.append(target_name)
    return resolved

def clock_filter(data, clock: float):
    if clock is None:
        return True
    clock = float(clock)
    if data['clock'] is None:
        return False
    synthesis_clock = float(data['clock'])
    if synthesis_clock == clock:
        return True
    return False
