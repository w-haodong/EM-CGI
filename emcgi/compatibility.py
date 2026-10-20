"""Architecture schema validation for core model state dictionaries."""
from collections import OrderedDict
from copy import deepcopy


SCHEMA = 'v132_silc_joint_residual_v2'
METHOD = 'SILC'
LEGACY_IDENTITY = (SCHEMA, METHOD)
PREFIXES = {}


def normalize_config(config):
    """Validate the model architecture identifier."""
    result = deepcopy(config)
    identity = (result.get('schema'), result.get('method'))
    if identity not in ((SCHEMA, METHOD), LEGACY_IDENTITY):
        raise ValueError(f"Unsupported V132 checkpoint schema: {result.get('schema')}")
    result['schema'] = SCHEMA
    result['method'] = METHOD
    return result


def _rename_key(key):
    prefix, separator, rest = key.partition('.')
    return PREFIXES.get(prefix, prefix) + separator + rest


def normalize_model_state(state):
    """Keep tensor objects and registration order; translate only module prefixes."""
    result = OrderedDict()
    for key, tensor in state.items():
        renamed = _rename_key(key)
        if renamed in result:
            raise ValueError(f'Ambiguous legacy/current checkpoint key: {renamed}')
        result[renamed] = tensor
    if hasattr(state, '_metadata'):
        result._metadata = OrderedDict((_rename_key(key), value)
                                       for key, value in state._metadata.items())
    return result
