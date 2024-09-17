from typing import Any, Tuple
import torchvision.datasets

class CelebA(torchvision.datasets.CelebA):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        return img/255, {attr_name : value for attr_name, value in zip(
            self.attr_names, label
        )}
