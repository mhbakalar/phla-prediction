import torch.nn as nn

# Monkey patch transformer model
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

def patch_attention(modules):
    attn_outputs = []

    def patch(m):
        forward_orig = m.forward

        def wrap(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False

            return forward_orig(*args, **kwargs)

        m.forward = wrap

    for module in modules:
        if isinstance(module, nn.MultiheadAttention):
            save_output = SaveOutput()
            patch(module)
            module.register_forward_hook(save_output)
            attn_outputs.append(save_output)

    return attn_outputs