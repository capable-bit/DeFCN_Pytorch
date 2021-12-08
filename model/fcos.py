from utils.model_utils import *
from utils.common import permute_to_N_HWA_K, permute_all_cls_and_box_to_N_HWA_K_and_concat, cat
from utils.box_utils import pairwise_iou,Boxes
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch

class FCOS(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = cfg.MODEL.DEVICE
        PIXEL_MEAN =  cfg.MODEL.PIXEL_MEAN

        self.backbone = cfg.build_backbone(
            cfg, input_shape=3)

        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        self.head = cfg.build_fcoshead(cfg,feature_shapes)
        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)

        BBOX_REG_WEIGHTS = cfg.MODEL.FCOS.BBOX_REG_WEIGHTS
        self.shift2box_transform = Shift2BoxTransform(
            weights=BBOX_REG_WEIGHTS)
        self.poto_alpha =  cfg.MODEL.POTO.ALPHA
        self.center_sampling_radius = cfg.MODEL.POTO.CENTER_SAMPLING_RADIUS
        self.poto_aux_topk = cfg.MODEL.POTO.AUX_TOPK


        PIXEL_STD  = cfg.MODEL.PIXEL_STD

        pixel_mean = torch.Tensor(PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.num_classes =      cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides =      cfg.MODEL.FCOS.FPN_STRIDES
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type =    cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.reg_weight =       cfg.MODEL.FCOS.REG_WEIGHT

        self.to(self.device)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        gt_instances = [
            x["instances"].to(self.device) for x in batched_inputs
        ]

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_filter = self.head(features)
        shifts = self.shift_generator(features)

        if self.training:
            gt_classes, gt_shifts_reg_deltas = self.get_ground_truth(
                shifts, gt_instances, box_cls, box_delta,
                box_filter)  # # gt_classes为所有特征点对应的类别以及anchor(只有最佳匹配的anchor才为真实目标类别，其余anchor为背景类别。只有最佳匹配的anchor才有定位参数其余的都为0  )

            losses = self.losses(gt_classes, gt_shifts_reg_deltas, box_cls,
                                 box_delta, box_filter)

            # 与上面的gt_classes不同，
            # 上面是以真实目标为出发点，在真实目标对应的所有预测特征点中找出最佳匹配的一个,一个真实目标与一个预测框对应，这里是考虑全局匹配
            # 这里的是以预测特征点为出发点，在预测特征所有对应的真实目标中找出一个最有把握的真实目标与之对应，一个预测点与一个最有把握的真实目标对应,当然这里无需考虑全局匹配，但也要对一些得分过低的预测框进行筛选，比如说有些预测框预测的全部真实目标的得分均很低那么这样的就不能要
            gt_classes = self.get_aux_ground_truth(
                shifts, gt_instances, box_cls, box_delta)
            aux_losses = self.aux_losses(gt_classes,
                                         box_cls)  # 同样的求出辅助函数中的cls，gt_classes中预测框与真实目标得分较高的为前景，预测框与所有真实目标得分过低的为背景
            losses.update(aux_losses)
            return losses


    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]   # 归一化
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets, box_cls, box_delta, box_filter):

        gt_classes = []
        gt_shifts_deltas = []
        # 一张图片对应的不同尺度特征图上所有特征点的80个类别分类得分、anchor定位参数、anchor生成的filter 均各自融合操作
        box_cls = torch.cat([permute_to_N_HWA_K(x, self.num_classes) for x in box_cls], dim=1)
        box_delta = torch.cat([permute_to_N_HWA_K(x, 4) for x in box_delta], dim=1)
        box_filter = torch.cat([permute_to_N_HWA_K(x, 1) for x in box_filter], dim=1)
        box_cls = box_cls.sigmoid_() * box_filter.sigmoid_()

        num_fg = 0
        num_gt = 0

        for shifts_per_image, targets_per_image, box_cls_per_image, box_delta_per_image in zip(
                shifts, targets, box_cls, box_delta):
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            prob = box_cls_per_image[:, targets_per_image.gt_classes].t()  # 取先验
            boxes = self.shift2box_transform.apply_deltas(
                box_delta_per_image, shifts_over_all_feature_maps
            )  # 根据对应特征图生成的anchor shift 来对预测bbox进行调整
            iou = pairwise_iou(gt_boxes, Boxes(boxes))
            quality = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha

            deltas = self.shift2box_transform.get_deltas(  # 真实目标gt_box 调整
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            if self.center_sampling_radius > 0:  # 中心采样，将那些离真实目标较远的特征点给较低的分
                centers = gt_boxes.get_centers()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                        torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            quality[~is_in_boxes] = -1

            gt_idxs, shift_idxs = linear_sum_assignment(quality.cpu().numpy(), maximize=True)  # 二分匹配，找出真实目标对应的最佳匹配预测框

            num_fg += len(shift_idxs)
            num_gt += len(targets_per_image)

            gt_classes_i = shifts_over_all_feature_maps.new_full(
                (len(shifts_over_all_feature_maps),), self.num_classes, dtype=torch.long
            )
            gt_shifts_reg_deltas_i = shifts_over_all_feature_maps.new_zeros(
                len(shifts_over_all_feature_maps), 4
            )
            if len(targets_per_image) > 0:
                # ground truth classes
                gt_classes_i[shift_idxs] = targets_per_image.gt_classes[gt_idxs]
                # ground truth box regression
                gt_shifts_reg_deltas_i[shift_idxs] = self.shift2box_transform.get_deltas(
                    shifts_over_all_feature_maps[shift_idxs], gt_boxes[gt_idxs].tensor
                )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)

        # get_event_storage().put_scalar("num_fg_per_gt", num_fg / num_gt)

        return torch.stack(gt_classes), torch.stack(gt_shifts_deltas)

    def losses(self, gt_classes, gt_shifts_deltas, pred_class_logits,
               pred_shift_deltas, pred_filtering):


        pred_class_logits, pred_shift_deltas, pred_filtering = \
            permute_all_cls_and_box_to_N_HWA_K_and_concat(
                pred_class_logits, pred_shift_deltas, pred_filtering,
                self.num_classes
            )  # Shapes: (N x R, K) and (N x R, 4), respectively.  # pred_class_logits代表网络预测出的所有特征点针对80个类别的分类得分，pred_shift_delta:代表所有特征点预测的定位得分 pred_filtering:anchor进入3DMF后生成的用来增强的数据
                #  将所有尺度特征图上，所有特征点的得分拼接在一起：(2,132*96+66*58+33*24+17*12+9*6,80)
        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        pred_class_logits = pred_class_logits.sigmoid() * pred_filtering.sigmoid()
        focal_loss_jit = torch.jit.script(focal_loss)  # type: torch.jit.ScriptModule

        # logits loss
        loss_cls = focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="sum",
        ) / max(1.0, num_foreground) * self.reg_weight

        return {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
        }

    def get_aux_ground_truth(self, shifts, targets, box_cls, box_delta):
        gt_classes = []

        box_cls = torch.cat([permute_to_N_HWA_K(x, self.num_classes) for x in box_cls], dim=1)
        box_delta = torch.cat([permute_to_N_HWA_K(x, 4) for x in box_delta], dim=1)
        box_cls = box_cls.sigmoid_()

        num_fg = 0
        num_gt = 0

        for shifts_per_image, targets_per_image, box_cls_per_image, box_delta_per_image in zip(
                shifts, targets, box_cls, box_delta):
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            prob = box_cls_per_image[:, targets_per_image.gt_classes].t()
            boxes = self.shift2box_transform.apply_deltas(
                box_delta_per_image, shifts_over_all_feature_maps
            )
            iou = pairwise_iou(gt_boxes, Boxes(boxes))
            quality = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha

            candidate_idxs = []
            st, ed = 0, 0
            for shifts_i in shifts_per_image: # 分别取出每个特征尺度上quality得分前9的预测点，是分别
                ed += len(shifts_i)
                _, topk_idxs = quality[:, st:ed].topk(self.poto_aux_topk, dim=1) # 取所有得分前9的,每行取前九
                candidate_idxs.append(st + topk_idxs)
                st = ed
            candidate_idxs = torch.cat(candidate_idxs, dim=1) # 5个尺度特征图，分别取每个尺度上得分为前9的特征点，即5*9=45个候选特征点

            is_in_boxes = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1)
            ).min(dim=-1).values > 0

            candidate_qualities = quality.gather(1, candidate_idxs) # 取出这45个候选特征点的得分
            quality_thr = candidate_qualities.mean(dim=1, keepdim=True) + \
                          candidate_qualities.std(dim=1, keepdim=True) # 求均值和方差
            is_foreground = torch.zeros_like(is_in_boxes).scatter_(1, candidate_idxs, True) # 45个候选特征点上的 index=True

            is_foreground &= quality >= quality_thr  # 当且仅当 45个候选特征点的quality大于平均quality_thr时才能算是前景框

            quality[~is_in_boxes] = -1  # 特征点不在is_in_boxes内的 设置较低的分-1
            quality[~is_foreground] = -1 # 不是前景框的也设置为-1

            # if there are still more than one objects for a position, 对于一个预测点来说，可能对应上多个真实目标，选取对应得分最大的最优把握的
            # we choose the one with maximum quality
            positions_max_quality, gt_matched_idxs = quality.max(dim=0) # 输出每一列最大值，每一列代表一个特征点，行数代表该特征点对n个真实目标，取每一列最大值代表该特征点最有把握的是哪个真实目标

            num_fg += (positions_max_quality != -1).sum().item()
            num_gt += len(targets_per_image)

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs] # 取出预测特征点对应的真实目标的类别，加入预测点对应index=0的真实目标，则取出该真实目标的label作为预测框的类别
                # Shifts with quality -1 are treated as background.
                gt_classes_i[positions_max_quality == -1] = self.num_classes # 其余的全当作背景处理
            else:
                gt_classes_i = torch.zeros_like(
                    gt_matched_idxs) + self.num_classes

            gt_classes.append(gt_classes_i)

        # get_event_storage().put_scalar("num_fg_per_gt_aux", num_fg / num_gt)

        return torch.stack(gt_classes)

    def aux_losses(self, gt_classes, pred_class_logits):
        pred_class_logits = cat([
            permute_to_N_HWA_K(x, self.num_classes) for x in pred_class_logits
        ], dim=1).view(-1, self.num_classes)

        gt_classes = gt_classes.flatten()

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())
        sigmoid_focal_loss_jit = torch.jit.script(
            sigmoid_focal_loss
        )  # type: torch.jit.ScriptModule

        # logits loss
        loss_cls_aux = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        return {"loss_cls_aux": loss_cls_aux}