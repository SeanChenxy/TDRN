import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ConvertFromInts_pair(object):
    def __call__(self, image, boxes=None, labels=None):
        return [image[0].astype(np.float32), image[1].astype(np.float32)], boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

class pairRandomSampleCrop(object):

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image[0].shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                # print('No crop')
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image1 = image[0]
                current_image2 = image[1]

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes[0], rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image1 = current_image1[rect[1]:rect[3], rect[0]:rect[2], :]
                current_image2 = current_image2[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers1 = (boxes[0][:, :2] + boxes[0][:, 2:]) / 2.0
                centers2 = (boxes[1][:, :2] + boxes[1][:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1_1 = (rect[0] < centers1[:, 0]) * (rect[1] < centers1[:, 1])
                m1_2 = (rect[0] < centers2[:, 0]) * (rect[1] < centers2[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2_1 = (rect[2] > centers1[:, 0]) * (rect[3] > centers1[:, 1])
                m2_2 = (rect[2] > centers2[:, 0]) * (rect[3] > centers2[:, 1])

                # mask in that both m1 and m2 are true
                mask1 = m1_1 * m2_1
                mask2 = m1_2 * m2_2
                mask = mask1 * mask2
                # have any valid boxes? try again if not
                # validate = mask1.any() and mask2.any()
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes1 = boxes[0][mask, :].copy()
                current_boxes2 = boxes[1][mask, :].copy()

                # take only matching gt labels
                current_labels1 = labels[0][mask]
                current_labels2 = labels[1][mask]

                # should we use the box left and top corner or the crop's
                current_boxes1[:, :2] = np.maximum(current_boxes1[:, :2], rect[:2])
                current_boxes2[:, :2] = np.maximum(current_boxes2[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes1[:, :2] -= rect[:2]
                current_boxes2[:, :2] -= rect[:2]

                current_boxes1[:, 2:] = np.minimum(current_boxes1[:, 2:], rect[2:])
                current_boxes2[:, 2:] = np.minimum(current_boxes2[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes1[:, 2:] -= rect[:2]
                current_boxes2[:, 2:] -= rect[:2]

                return [current_image1, current_image2], [current_boxes1, current_boxes2], [current_labels1, current_labels2]


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels

class pairExpand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image[0].shape

        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image1 = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image[0].dtype)
        expand_image2 = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image[1].dtype)

        expand_image1[:, :, :] = self.mean
        expand_image2[:, :, :] = self.mean

        expand_image1[int(top):int(top + height),
                     int(left):int(left + width)] = image[0]
        expand_image2[int(top):int(top + height),
                    int(left):int(left + width)] = image[1]

        image[0] = expand_image1
        image[1] = expand_image2

        boxes = boxes.copy()
        boxes[0][:, :2] += (int(left), int(top))
        boxes[0][:, 2:] += (int(left), int(top))
        boxes[1][:, :2] += (int(left), int(top))
        boxes[1][:, 2:] += (int(left), int(top))

        return image, boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

class pairRandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image[0].shape
        if random.randint(2):
            image[0] = image[0][:, ::-1]
            image[1] = image[1][:, ::-1]
            boxes = boxes.copy()
            boxes[0][:, 0::2] = width - boxes[0][:, 2::-2]
            boxes[1][:, 0::2] = width - boxes[1][:, 2::-2]
        return image, boxes, classes

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class pairPhotometricDistort(object):
    def __init__(self):
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes, labels):
        im1, im2 = image[0].copy(), image[1].copy()
        if random.randint(2):
            delta = random.uniform(-32, 32)
            im1 += delta
            im2 += delta
        if random.randint(2):
            if random.randint(2):
                alpha = random.uniform(0.5, 1.5)
                im1 *= alpha
                im2 *= alpha
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
            if random.randint(2):
                RadSat = random.uniform(0.5, 1.5)
                im1[:, :, 1] *= RadSat
                im2[:, :, 1] *= RadSat
            if random.randint(2):
                RadHue = random.uniform(-18, 18)
                im1[:, :, 0] += RadHue
                im1[:, :, 0][im1[:, :, 0] > 360.0] -= 360.0
                im1[:, :, 0][im1[:, :, 0] < 0.0] += 360.0
                im2[:, :, 0] += RadHue
                im2[:, :, 0][im2[:, :, 0] > 360.0] -= 360.0
                im2[:, :, 0][im2[:, :, 0] < 0.0] += 360.0
            im1 = cv2.cvtColor(im1, cv2.COLOR_HSV2BGR)
            im2 = cv2.cvtColor(im2, cv2.COLOR_HSV2BGR)
        else:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
            if random.randint(2):
                RadSat = random.uniform(0.5, 1.5)
                im1[:, :, 1] *= RadSat
                im2[:, :, 1] *= RadSat
            if random.randint(2):
                RadHue = random.uniform(-18, 18)
                im1[:, :, 0] += RadHue
                im1[:, :, 0][im1[:, :, 0] > 360.0] -= 360.0
                im1[:, :, 0][im1[:, :, 0] < 0.0] += 360.0
                im2[:, :, 0] += RadHue
                im2[:, :, 0][im2[:, :, 0] > 360.0] -= 360.0
                im2[:, :, 0][im2[:, :, 0] < 0.0] += 360.0
            im1 = cv2.cvtColor(im1, cv2.COLOR_HSV2BGR)
            im2 = cv2.cvtColor(im2, cv2.COLOR_HSV2BGR)
            if random.randint(2):
                alpha = random.uniform(0.5, 1.5)
                im1 *= alpha
                im2 *= alpha

        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            im1 = shuffle(im1)
            im2 = shuffle(im2)
        return [im1, im2], boxes, labels

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

class pairSSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # ConvertFromInts(),
            # ToAbsoluteCoords(),
            pairPhotometricDistort(),
            pairExpand(self.mean),
            pairRandomSampleCrop(),
            pairRandomMirror(),
            # ToPercentCoords(),
            # Resize(self.size),
            # SubtractMeans(self.mean)
        ])

    def __call__(self, img_pair, boxes_pair, labels_pair):
        for i in range(2):
            img_pair[i] = img_pair[i].astype(np.float32)
            height, width, channels = img_pair[i].shape
            boxes_pair[i][:, 0] *= width
            boxes_pair[i][:, 2] *= width
            boxes_pair[i][:, 1] *= height
            boxes_pair[i][:, 3] *= height
        img_pair, boxes_pair, labels_pair = self.augment(img_pair, boxes_pair, labels_pair)
        ########## display ############
        # img_draw, img_t_draw = img_pair[0].copy(), img_pair[1].copy()
        # img_draw = img_draw.astype(np.uint8)
        # img_t_draw = img_t_draw.astype(np.uint8)
        # height, width, channels = img_draw.shape
        # box, box_t = boxes_pair
        # lab, lab_t = labels_pair
        # print(lab, lab_t, img_draw.shape, img_t_draw.shape)
        # for bb, bbt in zip(box, box_t):
        #     x_min, y_min, x_max, y_max = bb
        #     x_min_t, y_min_t, x_max_t, y_max_t = bbt
        #     img_draw = cv2.rectangle(img_draw, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0,0),2)
        #     img_t_draw = cv2.rectangle(img_t_draw, (int(x_min_t),int(y_min_t)), (int(x_max_t),int(y_max_t)), (255,0,0),2)
        # img_c = np.concatenate((img_draw, img_t_draw), 1)
        # img_c = cv2.line(img_c, (width,0), (width, height), (0,0,255), 2)
        # cv2.imshow('test', cv2.resize(img_c, (width, int(height/2))))
        # cv2.waitKey(0)
        ###############################
        for i in range(2):
            height, width, channels = img_pair[i].shape
            boxes_pair[i][:, 0] /= width
            boxes_pair[i][:, 2] /= width
            boxes_pair[i][:, 1] /= height
            boxes_pair[i][:, 3] /= height
            img_pair[i] = cv2.resize(img_pair[i], (self.size, self.size))
            img_pair[i] = img_pair[i].astype(np.float32)
            img_pair[i] -= self.mean
        return img_pair, boxes_pair, labels_pair

class seqSSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.pre_augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            #Expand(self.mean),
            #RandomSampleCrop(),
            # RandomMirror(),
            # ToPercentCoords(),
            # Resize(self.size),
            # SubtractMeans(self.mean)
        ])
        self.post_augment = Compose([
            # ConvertFromInts(),
            # ToAbsoluteCoords(),
            # PhotometricDistort(),
            # Expand(self.mean),
            # RandomSampleCrop(),
            # RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def mirror(self, image, boxes, labels):
        _, width, _ = image.shape
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels

    def expand(self, image, boxes, labels, ratio):
        height, width, depth = image.shape
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels

    def __call__(self, img, boxes, labels, mirror=False, expand=0):
        img, boxes, labels = self.pre_augment(img, boxes, labels)
        if expand:
            img, boxes, labels = self.expand(img, boxes, labels, ratio=expand)
        if mirror:
            img, boxes, labels = self.mirror(img, boxes, labels)
        return self.post_augment(img, boxes, labels)