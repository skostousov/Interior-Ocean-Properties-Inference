from torchvision import transforms, utils
import torch

class RescaledRotationTransform(object):
    def __init__(self, degree_range = (-180, 180), scaling_interval = (0, 2)):
        self.degrees = degree_range
        self.scale = scaling_interval
        self.transform = transforms.RandomAffine(self.degrees, scale=self.scale)
    def __call__(self, image, label):
        img_features = image.shape[1]
        label_features = label.shape[1]
        concatonated = torch.cat([image, label], axis=1)
        output = self.transform(concatonated)
        transformed_image, transformed_output = output[:,:img_features], output[:, label_features+img_features]
        return transformed_image, transformed_output
    
class ToTensor(object):
    def __call__(self, image, label):
        return torch.from_numpy(image), torch.from_numpy(label)
