import base64
import io
from PIL import Image

def base64_image_encoder(image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')