import torch
from utils import logger
from thop import profile
from model.model import MyModel
from fvcore.nn import FlopCountAnalysis


def count_flop(path_to_load):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(path_to_load, weights_only=False).to(device)

    model.eval()
    lh = torch.randn(1, 204, 21, 2).to(device=device, dtype=torch.float32)
    rh = torch.randn(1, 204, 21, 2).to(device=device, dtype=torch.float32)
    bd = torch.randn(1, 204, 12, 2).to(device=device, dtype=torch.float32)

    flops = FlopCountAnalysis(model, (lh, rh, bd))
    # print("FLOPs: ", flops.total(), "GFLOPs")
    # with torch.no_grad():
    #     out = model(lh, rh, bd)
    macs, params = profile(model, inputs=(lh, rh, bd))
    print("FLOPs:", macs * 2 / 1e9)
    print("params:", params)

    return flops.total() / 1e9


if __name__ == "__main__":
    module = MyModel(
        num_enc_layers=1,
        num_dec_layers=1,
        pat_dec=0,
    )

    print(f'GFLOPs: {count_flop(f"out-checkpoints/LSA64/checkpoint_v_0.pth")}')
    print(f"Params: {sum(p.numel() for p in module.parameters() if p.requires_grad)}")
