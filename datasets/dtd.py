import numpy as np
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive


class DTD(VisionDataset):
    """
        Flowers Datasets
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    splits = ('train', 'val', 'trainval1', 'test1', "trainval_20", "test_5")
    #splits = ('train', 'val', 'trainval', 'test')

    #img_folder = os.path.join('fgvc', 'data', 'images')
    

    def __init__(self, root, train='trainval', transform=None,
                 target_transform=None, download=False):
        print(root)
        super(DTD, self).__init__(root, transform=transform, target_transform=target_transform)
        #split = 'trainval_20' if train else 'test_5'
        #split = 'trainval' if train else 'test'
        split = train

        self.root = root
        
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.split = split
        print(root)
        print(self.root)

        self.img_folder = "images"
        self.label_folder = "labels"
        self.classes_file = os.path.join(self.root, self.label_folder, '%s.txt' % (self.split))

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        #print(image_ids)
        samples = self.make_dataset(image_ids, targets, classes)
        #print(samples)
        print(classes)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        #image_ids[i]print(self.samples[index])
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(self.root) and \
               os.path.exists(self.classes_file)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        #print(self.classes_file)
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split('/')
                image_ids.append(split_line[1].rstrip())
                targets.append(split_line[0])

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets, classes):
        assert (len(image_ids) == len(targets))
        images = []
        #print(images_ids[i])
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder, classes[targets[i]], image_ids[i]), targets[i])
            images.append(item)
        return images


if __name__ == '__main__':
    train_dataset = Aircraft('/mnt/ssd/dtd', train=True, download=False)
    test_dataset = Aircraft('/mnt/ssd/dtd', train=False, download=False)
