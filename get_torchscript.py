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
    scripted = torch.jit.script(model)
    opt_model = mobile_optimizer.optimize_for_mobile(scripted)
    opt_model.save("model.pt")
    
    return 0

if __name__ == "__main__":
    exit(main())
