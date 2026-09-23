"""Board attention blocks and their wiring into the DDRNet context branch.

`attention.py` was verified bit-for-bit against a KataGo port before that port
was removed; the equivalence tests live on the archive/pre-prune branch.
"""

import unittest

import torch

from vgo_training.attention import (
    BoardFeedForward,
    BoardSelfAttention,
    BoardTransformerBlock,
    rope_tables,
)

CHANNELS, HEADS, SIZE = 32, 4, 6


def board(batch: int = 3, channels: int = CHANNELS, size: int = SIZE) -> torch.Tensor:
    torch.manual_seed(5)
    return torch.randn(batch, channels, size, size)


class RopeTests(unittest.TestCase):
    def test_rotation_preserves_length(self) -> None:
        # A rotation changes direction, never magnitude. If this fails the
        # pairing in `rotate` is wrong.
        cos, sin = rope_tables(16, 4, 4, theta=100.0)
        from vgo_training.attention import rotate

        x = torch.randn(2, 16, 3, 16)
        rotated = rotate(x, cos, sin)
        torch.testing.assert_close(
            x.norm(dim=-1), rotated.norm(dim=-1), rtol=1e-5, atol=1e-5
        )

    def test_rejects_a_theta_too_small_for_the_board(self) -> None:
        with self.assertRaises(ValueError):
            BoardSelfAttention(CHANNELS, HEADS, 64, 64, rope_theta=100.0)

    def test_rejects_a_head_dim_not_divisible_by_four(self) -> None:
        with self.assertRaises(ValueError):
            rope_tables(6, 4, 4)


class MaskTests(unittest.TestCase):
    def test_masked_cells_cannot_influence_the_output(self) -> None:
        # The point of the mask: whatever sits on off-board cells must not reach
        # the live ones. Corrupt them and the live outputs should not move.
        mine = BoardSelfAttention(CHANNELS, HEADS, SIZE, SIZE, qk_norm=True).eval()
        x = board()
        mask = torch.ones(x.shape[0], 1, SIZE, SIZE)
        mask[:, :, SIZE - 2 :, :] = 0.0

        polluted = x.clone()
        polluted[:, :, SIZE - 2 :, :] = 1e3
        with torch.no_grad():
            base = mine(x, mask)[:, :, : SIZE - 2, :]
            after = mine(polluted, mask)[:, :, : SIZE - 2, :]
        torch.testing.assert_close(base, after, rtol=1e-4, atol=1e-4)


class BlockTests(unittest.TestCase):
    def test_block_is_residual(self) -> None:
        # Both halves return residuals, so a block with zeroed output
        # projections has to be the identity.
        block = BoardTransformerBlock(CHANNELS, HEADS, SIZE, SIZE).eval()
        with torch.no_grad():
            block.attention.to_out.weight.zero_()
            block.feed_forward.down.weight.zero_()
            x = board()
            torch.testing.assert_close(block(x), x)

    def test_block_preserves_shape(self) -> None:
        block = BoardTransformerBlock(CHANNELS, HEADS, SIZE, SIZE).eval()
        x = board()
        with torch.no_grad():
            self.assertEqual(block(x).shape, x.shape)

    def test_gradients_reach_every_parameter(self) -> None:
        block = BoardTransformerBlock(CHANNELS, HEADS, SIZE, SIZE)
        block(board()).sum().backward()
        missing = [n for n, p in block.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])




class ContextAttentionWiringTests(unittest.TestCase):
    """Attention in the DDRNet context branch, behind `context_attention_blocks`."""

    def build(self, attention_blocks: int, **kwargs):
        from vgo_training.model import build_model

        return build_model(
            channels=5, width=32, blocks=8, policy_resolution=32,
            norm_groups=8, context_attention_blocks=attention_blocks,
            raster_resolution=64, **kwargs,
        )

    def test_zero_blocks_leaves_the_model_untouched(self) -> None:
        # The flag must be inert by default, or every existing checkpoint
        # silently becomes incompatible.
        from vgo_training.model import build_model

        torch.manual_seed(0)
        plain = build_model(
            channels=5, width=32, blocks=8,
            policy_resolution=32, norm_groups=8,
        )
        torch.manual_seed(0)
        flagged = self.build(0)
        self.assertEqual(set(plain.state_dict()), set(flagged.state_dict()))
        x = torch.randn(2, 5, 64, 64)
        plain.eval()
        flagged.eval()
        with torch.no_grad():
            for a, b in zip(plain(x), flagged(x)):
                self.assertTrue(torch.equal(a, b))

    def test_attention_blocks_replace_residual_blocks(self) -> None:
        from vgo_training.model import ResidualBlock

        with_attention = self.build(1)
        stage = with_attention.context_stage2
        self.assertEqual(len(stage.attention), 1)
        # One residual block gave way to it rather than being added alongside.
        plain = self.build(0)
        self.assertEqual(
            len([m for m in stage.body if isinstance(m, ResidualBlock)]),
            len([m for m in plain.context_stage2.body if isinstance(m, ResidualBlock)]) - 1,
        )

    def test_forward_shapes_are_unchanged(self) -> None:
        model = self.build(1).eval()
        x = torch.randn(2, 5, 64, 64)
        with torch.no_grad():
            policy, values = model(x)
        self.assertEqual(policy.shape, (2, 32 * 32 + 1))
        self.assertEqual(values.shape, (2,))

    def test_gradients_reach_the_attention_blocks(self) -> None:
        model = self.build(1)
        model(torch.randn(2, 5, 64, 64))[0].sum().backward()
        missing = [
            name
            for name, param in model.context_stage2.attention.named_parameters()
            if param.grad is None
        ]
        self.assertEqual(missing, [])

    def test_requires_the_raster_resolution(self) -> None:
        from vgo_training.model import build_model

        with self.assertRaises(ValueError):
            build_model(
                channels=5, width=32, blocks=8, policy_resolution=32,
                norm_groups=8, context_attention_blocks=1,
            )

    def test_rejects_replacing_more_blocks_than_exist(self) -> None:
        with self.assertRaises(ValueError):
            self.build(99)


if __name__ == "__main__":
    unittest.main()
