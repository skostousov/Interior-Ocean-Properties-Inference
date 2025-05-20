from torchvision import transforms, utils
import torch

class RescaledRotationTransform(object):
    def __init__(self, degree_range = 180, scaling_interval = (1, 2)):
        self.degrees = degree_range
        self.scale = scaling_interval
        self.transform = transforms.RandomAffine(self.degrees, scale=self.scale)
    def __call__(self, image, label):
        img_features = image.shape[0]
        label_features = label.shape[0]
        concatonated = torch.cat([image, label], axis=0)
        output = self.transform(concatonated)
        transformed_image, transformed_label = output[:img_features], output[img_features:label_features+img_features]
        return transformed_image, transformed_label
    
class ToTensor(object):
    def __call__(self, image, label):
        return torch.from_numpy(image), torch.from_numpy(label)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
        if torch.is_tensor(image):
            image = image.numpy()
        if torch.is_tensor(label):
            label = label.numpy()
        return image, label