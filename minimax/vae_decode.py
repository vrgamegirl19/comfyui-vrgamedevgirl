"""Batched MiniMax video VAE decoding with stock-compatible spatial blending."""

from typing import Any

import comfy.model_management as mm
import torch


def tiled_decode_batched(model: Any, z: torch.Tensor, tile_batch_size: int) -> torch.Tensor:
    """Batch spatial decoding while preserving the installed stock VAE seam policy."""
    height, width = z.shape[-2] * model.vae_ratio, z.shape[-1] * model.vae_ratio
    y_idx, y_len, y_overlap = model.split_tiles(height)
    x_idx, x_len, x_overlap = model.split_tiles(width)
    tiles = [(i, j, yp // model.vae_ratio, yl // model.vae_ratio,
              xp // model.vae_ratio, xl // model.vae_ratio)
             for i, (yp, yl) in enumerate(zip(y_idx, y_len))
             for j, (xp, xl) in enumerate(zip(x_idx, x_len))]
    canvas = None
    blended_tails = hasattr(model, "_decode_tile_row")
    strip = None
    new_strip = None
    row_tails = []
    new_tails = []
    left_tail = None
    out_y = 0
    out_x = 0
    start = 0
    while start < len(tiles):
        mm.throw_exception_if_processing_interrupted()
        end = start + 1
        shape = (tiles[start][3], tiles[start][5])
        while end < min(start + tile_batch_size, len(tiles)):
            if (tiles[end][3], tiles[end][5]) != shape:
                break
            end += 1
        batch = torch.cat([z[..., yp:yp + yl, xp:xp + xl]
                           for _, _, yp, yl, xp, xl in tiles[start:end]], dim=0)
        decoded = model._decode_pixels(batch)
        for k, (i, j, _, _, _, _) in enumerate(tiles[start:end]):
            tile = decoded[k * z.shape[0]:(k + 1) * z.shape[0]]
            if not blended_tails and i < len(y_idx) - 1:
                new_tails.append(tile[..., -y_overlap[i]:, :].clone())
            next_left_tail = (
                tile[..., :, -x_overlap[j]:].clone()
                if not blended_tails and j < len(x_idx) - 1 else None
            )
            if i > 0:
                above = strip[..., :, x_idx[j]:x_idx[j] + x_len[j]] if blended_tails else row_tails[j]
                tile = model.blend(above, tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = model.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
            if blended_tails:
                left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
            else:
                left_tail = next_left_tail
            if j < len(x_idx) - 1:
                tile = tile[..., :, :-x_overlap[j]]
            if i < len(y_idx) - 1:
                if blended_tails:
                    if new_strip is None:
                        new_strip = torch.empty(
                            *tile.shape[:-2], y_overlap[i], width, dtype=tile.dtype, device=tile.device,
                        )
                    new_strip[..., :, out_x:out_x + tile.shape[-1]] = tile[..., -y_overlap[i]:, :]
                tile = tile[..., :-y_overlap[i], :]
            if canvas is None:
                canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
            canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
            out_x += tile.shape[-1]
            if j == len(x_idx) - 1:
                strip = new_strip
                new_strip = None
                row_tails = new_tails
                new_tails = []
                out_y += tile.shape[-2]
                out_x = 0
        del tile, decoded, batch
        start = end
    return canvas
