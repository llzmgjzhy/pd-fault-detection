from .patchtst import PatchTST
from .swinPatchtst import SwinPatchTST

model_factory = {
    "patchtst": PatchTST,
    "swinPatchtst": SwinPatchTST,
    # Add other models here as needed
}
