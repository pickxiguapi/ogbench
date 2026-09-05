import torch

from lewm_visual_decoder import CLSDecoder, ConvDecoder


def test_lewm_visual_decoder_shape_and_range():
    model = CLSDecoder(
        cls_dim=16, hidden_dim=32, depth=1, heads=2, dim_head=8,
        mlp_dim=64, image_size=32, patch_size=8,
    )
    output = model(torch.randn(2, 16))
    assert output.shape == (2, 3, 32, 32)
    assert output.min() >= -1
    assert output.max() <= 1


def test_lewm_conv_visual_decoder_shape_and_finite_values():
    model = ConvDecoder(
        cls_dim=16,
        base_channels=64,
        min_channels=16,
        start_size=4,
        image_size=32,
        num_res_blocks=2,
    )
    output = model(torch.randn(2, 16))
    assert output.shape == (2, 3, 32, 32)
    assert torch.isfinite(output).all()
