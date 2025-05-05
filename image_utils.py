from pathlib import Path
from io import BytesIO
import base64, cv2, numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).with_suffix('').parent
PRESETS = ROOT / "data" / "presets"

def pil_to_base64(img:Image.Image)->str:
    buf = BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()
def base64_to_pil(b64:str)->Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64)))

# filters
def gray(img):  return ImageOps.grayscale(img)
def edge(img):
    cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.Canny(cv,50,150))
def neon(img):  # a creative flair filter
    edged = edge(img); cv = cv2.applyColorMap(np.array(edged), cv2.COLORMAP_HOT)
    return Image.fromarray(cv)

FILTERS = {"gray": gray, "edge": edge, "neon": neon}

def list_presets(): return [p.name for p in PRESETS.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".gif"}]
def load_preset(name:str): return Image.open(PRESETS/name)
