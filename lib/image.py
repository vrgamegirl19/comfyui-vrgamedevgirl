import base64

import PIL
import numpy as np
from PIL import Image
from torch import Tensor


def tensor2pil(image: Tensor) -> PIL.Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2base64(image: PIL.Image.Image) -> str:
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
