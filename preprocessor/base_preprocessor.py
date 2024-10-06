import torchvision.transforms as T
from utils.config import Config

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

class Transform():
    def __init__(self):
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((416, 416)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    def __call__(self, image):
        return self.transform(image)


class ColonPreprocessor():
    """For train dataset standard transformation."""
    def __init__(self):
        self.pre_size = 256
        self.image_size = 224
        self.interpolation = T.InterpolationMode.BILINEAR
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.Resize(self.pre_size,
                                interpolation=self.interpolation),
            T.CenterCrop(self.image_size),
            T.RandomHorizontalFlip(),
            T.RandomCrop(self.image_size, padding=4),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def setup(self):
        pass

    def __call__(self, image):
        return self.transform(image)

