"""The rewritten blocks must compute exactly what the KataGo port computes.

`attention.py` is a from-scratch reading, so "it looks right" is not evidence.
These tests copy weights across and demand bit-level agreement -- if the two
ever diverge, the readable version is wrong and this says so.
"""

import unittest

import torch

from vgo_training.attention import (
    BoardFeedForward,
    BoardSelfAttention,
    BoardTransformerBlock,
    rope_tables,
)
from vgo_training.katago_transformer import (
    TransformerAttentionBlock,
    TransformerFFNBlock,
    precompute_freqs_cos_sin_2d,
)

CHANNELS, HEADS, SIZE = 32, 4, 6


def board(batch: int = 3, channels: int = CHANNELS, size: int = SIZE) -> torch.Tensor:
    torch.manual_seed(5)
    return torch.randn(batch, channels, size, size)


class RopeTests(unittest.TestCase):
    def test_tables_match_the_katago_port(self) -> None:
        mine_cos, mine_sin = rope_tables(CHANNELS // HEADS, SIZE, SIZE, theta=100.0)
        theirs_cos, theirs_sin = precompute_freqs_cos_sin_2d(
            CHANNELS // HEADS, SIZE, theta=100.0
        )
        self.assertTrue(torch.equal(mine_cos, theirs_cos))
        self.assertTrue(torch.equal(mine_sin, theirs_sin))

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


class AttentionEquivalenceTests(unittest.TestCase):
    def build(self, qk_norm: bool):
        mine = BoardSelfAttention(CHANNELS, HEADS, SIZE, SIZE, qk_norm=qk_norm).eval()
        theirs = TransformerAttentionBlock(
            "attn", CHANNELS,
            {"transformer_heads": HEADS, "attention_qk_norm": qk_norm},
            pos_len=SIZE,
        ).eval()
        theirs.load_state_dict(
            {
                "norm1.weight": mine.norm.weight,
                "q_proj.weight": mine.to_q.weight,
                "k_proj.weight": mine.to_k.weight,
                "v_proj.weight": mine.to_v.weight,
                "out_proj.weight": mine.to_out.weight,
                **(
                    {
                        "q_norm.weight": mine.q_norm.weight,
                        "k_norm.weight": mine.k_norm.weight,
                    }
                    if qk_norm
                    else {}
                ),
            },
            strict=False,
        )
        return mine, theirs

    def test_matches_without_qk_norm(self) -> None:
        mine, theirs = self.build(qk_norm=False)
        x = board()
        with torch.no_grad():
            torch.testing.assert_close(mine(x), theirs(x), rtol=1e-5, atol=1e-6)

    def test_matches_with_qk_norm(self) -> None:
        mine, theirs = self.build(qk_norm=True)
        x = board()
        with torch.no_grad():
            torch.testing.assert_close(mine(x), theirs(x), rtol=1e-5, atol=1e-6)

    def test_matches_under_a_board_mask(self) -> None:
        mine, theirs = self.build(qk_norm=True)
        x = board()
        mask = torch.ones(x.shape[0], 1, SIZE, SIZE)
        mask[:, :, SIZE - 2 :, :] = 0.0
        with torch.no_grad():
            torch.testing.assert_close(
                mine(x, mask), theirs(x, mask), rtol=1e-5, atol=1e-6
            )

    def test_masked_cells_cannot_influence_the_output(self) -> None:
        # The point of the mask: whatever sits on off-board cells must not reach
        # the live ones. Corrupt them and the live outputs should not move.
        mine, _ = self.build(qk_norm=True)
        x = board()
        mask = torch.ones(x.shape[0], 1, SIZE, SIZE)
        mask[:, :, SIZE - 2 :, :] = 0.0

        polluted = x.clone()
        polluted[:, :, SIZE - 2 :, :] = 1e3
        with torch.no_grad():
            base = mine(x, mask)[:, :, : SIZE - 2, :]
            after = mine(polluted, mask)[:, :, : SIZE - 2, :]
        torch.testing.assert_close(base, after, rtol=1e-4, atol=1e-4)


class FeedForwardEquivalenceTests(unittest.TestCase):
    def build(self, swiglu: bool, depthwise: bool):
        hidden = 2 * CHANNELS
        mine = BoardFeedForward(
            CHANNELS, hidden, swiglu=swiglu, depthwise=depthwise
        ).eval()
        theirs = TransformerFFNBlock(
            "ffn", CHANNELS,
            {
                "transformer_ffn_channels": hidden,
                "transformer_ffn_depthwise_conv": depthwise,
            },
            use_swiglu=swiglu,
        ).eval()
        state = {
            "norm.weight": mine.norm.weight,
            "ffn_linear1.weight": mine.up.weight,
            "ffn_linear2.weight": mine.down.weight,
        }
        if swiglu:
            state["ffn_linear_gate.weight"] = mine.gate.weight
        if depthwise:
            state["ffn_dwconv.weight"] = mine.conv.weight
        theirs.load_state_dict(state, strict=False)
        return mine, theirs

    def test_matches_with_swiglu(self) -> None:
        mine, theirs = self.build(swiglu=True, depthwise=False)
        x = board()
        with torch.no_grad():
            torch.testing.assert_close(mine(x), theirs(x), rtol=1e-5, atol=1e-6)

    def test_matches_without_swiglu(self) -> None:
        mine, theirs = self.build(swiglu=False, depthwise=False)
        x = board()
        with torch.no_grad():
            torch.testing.assert_close(mine(x), theirs(x), rtol=1e-5, atol=1e-6)

    def test_matches_with_depthwise_conv(self) -> None:
        mine, theirs = self.build(swiglu=True, depthwise=True)
        x = board()
        mask = torch.ones(x.shape[0], 1, SIZE, SIZE)
        with torch.no_grad():
            torch.testing.assert_close(
                mine(x, mask), theirs(x, mask), rtol=1e-5, atol=1e-6
            )


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
            "ddrnet", channels=5, width=32, blocks=8, policy_resolution=32,
            norm_groups=8, context_attention_blocks=attention_blocks,
            raster_resolution=64, **kwargs,
        )

    def test_zero_blocks_leaves_the_model_untouched(self) -> None:
        # The flag must be inert by default, or every existing checkpoint
        # silently becomes incompatible.
        from vgo_training.model import build_model

        torch.manual_seed(0)
        plain = build_model(
            "ddrnet", channels=5, width=32, blocks=8,
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
                "ddrnet", channels=5, width=32, blocks=8, policy_resolution=32,
                norm_groups=8, context_attention_blocks=1,
            )

    def test_rejects_replacing_more_blocks_than_exist(self) -> None:
        with self.assertRaises(ValueError):
            self.build(99)


if __name__ == "__main__":
    unittest.main()
