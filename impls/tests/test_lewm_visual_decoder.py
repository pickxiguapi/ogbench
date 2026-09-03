import torch

from lewm_visual_decoder import CLSDecoder


def test_lewm_visual_decoder_shape_and_range():
    model = CLSDecoder(
        cls_dim=16, hidden_dim=32, depth=1, heads=2, dim_head=8,
        mlp_dim=64, image_size=32, patch_size=8,
    )
    output = model(torch.randn(2, 16))
    assert output.shape == (2, 3, 32, 32)
    assert output.min() >= -1
    assert output.max() <= 1
