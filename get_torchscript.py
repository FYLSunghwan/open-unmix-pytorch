import torch
import openunmix
import numpy as np
import torch.utils.mobile_optimizer as mobile_optimizer


def main():
    # Get OpenUnmix Model
    separator = openunmix.umxhq()
    model = separator.target_models["vocals"]
    model.eval()

    # Dummy Input Test
    dummy_input = torch.FloatTensor(np.zeros([427, 1, 2, 2049]))
    model(dummy_input)

    # Convert Model to TorchScript
    traced = torch.jit.trace(model, dummy_input)
    scripted = torch.jit.script(traced)
    opt_model = mobile_optimizer.optimize_for_mobile(scripted)
    opt_model.save("model.pt")
    
    logger.info("[+] Successfully got openunmix torchscript model.")
    return 0

if __name__ == "__main__":
    exit(main())
