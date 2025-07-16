from torchvision import transforms, utils
import torch

class RescaledRotationTransform(object):
    def __init__(self, degree_range = 180, scaling_interval = (1, 2)):
        self.degrees = degree_range
        self.scale = scaling_interval
        self.transform = transforms.RandomAffine(self.degrees, scale=self.scale)
    def __call__(self, image, label=None):
        img_features = image.shape[0]
        if label is not None:
            label_features = label.shape[0]
            concatenated = torch.cat([image, label], axis=0)
            output = self.transform(concatenated)
            transformed_image, transformed_label = output[:img_features], output[img_features:label_features+img_features]
            return transformed_image, transformed_label
        else:
            return self.transform(image)

class GANTransform():
    def __init__(self, size = 50):
        self.random_crop = transforms.RandomResizedCrop(size)
        self.random_h_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.random_v_flip = transforms.RandomVerticalFlip(p=0.5)
        self.compose = transforms.Compose([
            self.random_crop,
            self.random_h_flip,
            self.random_v_flip,
        ])
    def __call__(self, image, label=None):
        img_features = image.shape[0]
        if label is not None:
            label_features = label.shape[0]
            concatenated = torch.cat([image, label], axis=0)
            output = self.compose(concatenated)
            transformed_image, transformed_label = output[:img_features], output[img_features:label_features+img_features]
            return transformed_image, transformed_label
        else:
            return self.compose(image)

    
class ToTensor(object):
    def __call__(self, image, label=None):
        if label is not None:
            return torch.from_numpy(image), torch.from_numpy(label)
        return torch.from_numpy(image)

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
    

if __name__ == "__main__":
    tensor = torch.randn(6, 21, 21)
    transform = RescaledRotationTransform()
    transformed_tensor = transform(tensor)
    print(transformed_tensor.shape)