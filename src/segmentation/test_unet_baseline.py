import torch

from src.segmentation.unet_baseline import UNetBaseline


def main() -> None:
    model = UNetBaseline(in_channels=3, out_channels=1, base_channels=16)
    model.eval()
    dummy = torch.randn(2, 3, 256, 256)
    output = model(dummy)

    assert output.shape == (2, 1, 256, 256), output.shape
    assert output.dtype == torch.float32, output.dtype
    assert torch.isfinite(output).all(), "Model output contains non-finite values."
    assert output.device.type == "cpu", output.device

    print("UNET FORWARD TEST: PASS")


if __name__ == "__main__":
    main()
