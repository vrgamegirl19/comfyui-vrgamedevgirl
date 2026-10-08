import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "minimax/nodes.py"
INIT_SOURCE = ROOT / "__init__.py"


def comfy_root():
    return next((parent for parent in ROOT.parents if (parent / "comfy").is_dir()), None)


class MiniMaxH3FastVAEDecodeTests(unittest.TestCase):
    def test_node_is_registered(self):
        self.assertIn('".minimax.nodes"', INIT_SOURCE.read_text(encoding="utf-8"))
        source = NODE_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"H3FastVAEDecode": H3FastVAEDecode', source)

    @unittest.skipUnless(torch is not None and comfy_root() is not None, "Requires ComfyUI's Python environment.")
    def test_batched_tiles_match_stock_spatial_blending(self):
        root = comfy_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        spec = importlib.util.spec_from_file_location("vrgdg_h3_fast_vae", ROOT / "minimax/vae_decode.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class TileFixture:
            vae_ratio = 1
            tile_size = 8
            tile_overlap_min = 2
            split_tiles = MiniMaxH3VideoVAE.split_tiles
            blend = MiniMaxH3VideoVAE.blend
            if hasattr(MiniMaxH3VideoVAE, "_decode_tile_row"):
                _decode_tile_row = MiniMaxH3VideoVAE._decode_tile_row

            def _decode_pixels(self, z):
                z = z.contiguous()
                return z + z.mean(dim=(-3, -2, -1), keepdim=True)

        fixture = TileFixture()
        for height, width in ((4, 4), (8, 19), (19, 8), (19, 23), (24, 24)):
            latent = torch.randn(2, 3, 7, height, width)
            stock = MiniMaxH3VideoVAE.tiled_decode(fixture, latent)
            for batch_size in (1, 2, 4, 16):
                actual = module.tiled_decode_batched(fixture, latent, batch_size)
                torch.testing.assert_close(actual, stock, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
