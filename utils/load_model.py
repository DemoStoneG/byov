import torch

def load_ckpt(backbone, ckpt_name):
    print(f'Loading pre-trained model: {ckpt_name}')
    ckpt = torch.load(
        ckpt_name,
        map_location=lambda storage, loc: storage,
    )

    if "state_dict" in ckpt:
        key = "state_dict"
        source_state = ckpt[key]
    elif "model_state" in ckpt:
        key = "model_state"
        source_state = ckpt[key]
    else:
        key = "raw_state_dict"
        source_state = ckpt
    target_keys = set(backbone.state_dict().keys())
    state_dict = {}
    prefixes = ('encoder.', 'module.encoder.', 'model.encoder.', 'module.')

    for source_key, value in source_state.items():
        candidates = [source_key]
        candidates.extend(source_key[len(prefix):] for prefix in prefixes if source_key.startswith(prefix))
        matched_key = next((candidate for candidate in candidates if candidate in target_keys), None)
        if matched_key is not None:
            state_dict[matched_key] = value

    if not state_dict:
        raise RuntimeError(
            f'Checkpoint {ckpt_name} contains no parameters matching the BYOV encoder. '
            f'Example checkpoint keys: {list(source_state)[:5]}'
        )

    missing_keys, unexpected_keys = backbone.load_state_dict(
        state_dict, strict=False
    )

    print('missing', missing_keys)
    print('unexpected', unexpected_keys)
    return {
        'checkpoint_key': key,
        'source_parameter_count': len(source_state),
        'loaded_parameter_count': len(state_dict),
        'missing_keys': list(missing_keys),
        'unexpected_keys': list(unexpected_keys),
    }
