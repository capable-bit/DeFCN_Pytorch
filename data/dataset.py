from torch.utils.data import Dataset
import os
from utils.box_utils import BoxMode,Boxes
import copy
from PIL import Image
import numpy as np
import torch
from data.instances import Instances


class ResizeTransform:
    def __init__(self,img, annotations=None):

        self.short_edge_length = (640, 672, 704, 736, 768, 800)
        self.max_size = 1333
        self.img = img
        self.annotations = annotations

    def apply(self):
        img = self.img
        annotations = self.annotations

        h, w =img.shape[:2]
        size = np.random.choice(self.short_edge_length)
        scale = size * 1.0 / min(h, w)

        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize((neww, newh), 2)
        img = np.asarray(pil_image)
        #  --------------annotation-------------- #
        for annotation in annotations:
            # 将定位框 XYWH->XYXY
            bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
            idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
            coords = np.asarray([bbox]).reshape(-1, 4)[:, idxs].reshape(-1, 2)

            coords[:, 0] = coords[:, 0] * (neww * 1.0 / w)
            coords[:, 1] = coords[:, 1] * (newh * 1.0 / h)
            coords = coords.reshape((-1, 4, 2))

            minxy = coords.min(axis=1)
            maxxy = coords.max(axis=1)
            trans_boxes = np.concatenate((minxy, maxxy), axis=1)[0]

            annotation["bbox"] = trans_boxes
            annotation["bbox_mode"] = BoxMode.XYXY_ABS

        return img,annotations
class FlipTransform:
    def __init__(self,img, annotations=None):
        self.img = img
        self.annotations = annotations

    def apply(self):
        img = self.img
        annotations = self.annotations

        h, w = img.shape[:2]

        do = np.random.uniform(0, 1.0, []) < 0.5

        if do:
            tensor = torch.from_numpy(np.ascontiguousarray(img).copy())
            tensor = tensor.flip((-2))
            img = tensor.numpy()

            #  --------------annotation-------------- #
            for annotation in annotations:
                bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)

                idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
                coords = np.asarray([bbox]).reshape(-1, 4)[:, idxs].reshape(-1, 2)

                # self.apply_coords(coords).reshape((-1, 4, 2))
                coords[:, 0] = w - coords[:, 0]
                coords = coords.reshape((-1, 4, 2))

                minxy = coords.min(axis=1)
                maxxy = coords.max(axis=1)
                trans_boxes = np.concatenate((minxy, maxxy), axis=1)[0]

                annotation["bbox"] = trans_boxes
                annotation["bbox_mode"] = BoxMode.XYXY_ABS

            return img, annotations
        else:
            return img,annotations

class COCODataset(Dataset):
    def __init__(self,ann_file,img_folder,transforms):
        super().__init__()
        self.dataset_dicts = self._load_annotations(ann_file,img_folder)  # 依靠COCO数据集api构建自定义的数据list
        self.dataset_dicts = self._filter_annotations()  # 过滤掉iscrowd=1 的数据
        self._set_group_flag()
        self.transforms = transforms

    # 依靠COCO数据集api构建自定义的数据list
    def _load_annotations(self,ann_file,img_folder):
        from pycocotools.coco import COCO
        coco_api = COCO(ann_file)   # 加载coco数据
        cat_ids = sorted(coco_api.getCatIds())  # 获取所有类别ID
        id_map = {v: i for i, v in enumerate(cat_ids)}  # 类别id映射到0~80

        img_ids = sorted(coco_api.imgs.keys())  # 获取所有图片id
        imgs = coco_api.loadImgs(img_ids)       # 加载图片

        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]   # 获取每张图片下所有目标的信息归在一起

        imgs_anns = list(zip(imgs, anns))       # 将图片与图片对应下所有目标信息组合一起

        dataset_dicts = []
        ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]  # ann的key有那些

        for (img_dict, anno_dict_list) in imgs_anns:    # 遍历每张图片以及每个目标
            record = {}
            record["file_name"] = os.path.join(img_folder,img_dict["file_name"])  # 获取图片路径
            record["height"] = img_dict["height"]       # 图片宽高
            record["width"] = img_dict["width"]
            record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list: # 遍历每个目标
                obj = {key: anno[key] for key in ann_keys if key in anno}
                obj["category_id"] = id_map[obj["category_id"]]
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    # 过滤掉iscrowd=1 的数据
    def _filter_annotations(self):
        def valid(anns):
            for ann in anns:
                if ann.get("iscrowd", 0) == 0:
                    return True
            return False

        return [x for x in self.dataset_dicts if valid(x["annotations"])]

    def _set_group_flag(self):
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)
        if "width" in self.dataset_dicts[0] and "height" in self.dataset_dicts[0]:
            for i in range(len(self)):
                dataset_dict = self.dataset_dicts[i]
                if dataset_dict['width'] / dataset_dict['height'] > 1:
                    self.aspect_ratios[i] = 1

    def __getitem__(self, index):
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        # 读取图片
        image = self._read_image(dataset_dict["file_name"], format='RGB')

        annotations = dataset_dict.pop("annotations")
        annotations = [
            ann for ann in annotations if ann.get("iscrowd", 0) == 0]

        # ---transform------- #
        # Resize + horizontal Flip:
        for tfm in self.transforms:
            image, annotations = tfm(image, annotations).apply()

        instances = self._annotations_to_instances(annotations, image)

        dataset_dict["instances"] = self.filter_empty_instances(instances)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        return dataset_dict

    # 读取图片，转为'RGB'格式
    def _read_image(self, file_path, format=None):
        with open(file_path, "rb") as f:
            image = Image.open(f)
            image = image.convert(format)
            image = np.asarray(image)
            image = image[:, :, ::-1]
            return image

    def _annotations_to_instances(self, annos, image):
        image_size = image.shape[:2]  # h, w

        boxes = [
            BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
            for obj in annos
        ]
        target = Instances(image_size)

        boxes = target.gt_boxes = Boxes(boxes)
        boxes.clip(image_size)

        classes = [obj["category_id"] for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        r = []
        r.append(target.gt_boxes.nonempty())

        # can also filter visible keypoints
        if not r:
            return target
        m = r[0]
        for x in r[1:]:
            m = m & x
        return target[m]

    def __len__(self):
        return len(self.dataset_dicts)

    def filter_empty_instances(self, instances, by_box=True, by_mask=True):
        """
        Filter out empty instances in an `Instances` object.

        Args:
            instances (Instances):
            by_box (bool): whether to filter out instances with empty boxes
            by_mask (bool): whether to filter out instances with empty masks

        Returns:
            Instances: the filtered instances.
        """
        assert by_box or by_mask
        r = []
        if by_box:
            r.append(instances.gt_boxes.nonempty())
        if instances.has("gt_masks") and by_mask:
            r.append(instances.gt_masks.nonempty())

        # TODO: can also filter visible keypoints

        if not r:
            return instances
        m = r[0]
        for x in r[1:]:
            m = m & x
        return instances[m]