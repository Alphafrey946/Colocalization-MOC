import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import HEADS, build_loss
from .anchor_free_head_pc import AnchorFreeHead_pc

INF = 1e8


@HEADS.register_module()
class FCOSHead_pc(AnchorFreeHead_pc):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 moc_on_reg = False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_moc=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.moc_on_reg = moc_on_reg
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_moc = build_loss(loss_moc)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.fcos_moc = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_centerness, std=0.01)
        normal_init(self.fcos_moc, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                centernesses (list[Tensor]): Centerss for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions, centerness, and moc
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        if self.moc_on_reg:
            moc = self.fcos_moc(reg_feat)
        else:
            moc = self.fcos_moc(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness, moc

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses','moc'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             mocs,
             gt_bboxes,
             gt_labels,
             img_metas,
             imgs,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            mocs (listp[Temspr]): Coefficient for each scale level, each is 
                a 4D-tensor, the channel numer is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            imgs (list[Tensor]): images in each level.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_moc = [
            moc.permute(0, 2, 3, 1).reshape(-1)
            for moc in mocs
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_mocs = torch.cat(flatten_moc)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        num_points = [center.size(0) for center in all_level_points]
        #coefficient
        moc_result_list = self.convertlevel2img(flatten_bbox_targets,flatten_labels,flatten_bbox_preds,flatten_points,flatten_mocs,num_points,num_imgs)
        flatten_bbox_targets_reshape = moc_result_list[0]
        flatten_labels_targets_reshape= moc_result_list[1] 
        flatten_bbox_preds_reshape = moc_result_list[2]
        flatten_points_reshape = moc_result_list[3]
        flatten_conv_reshape= moc_result_list[4]
        
        
        #print(num_points,labels,bg_class_ind)
        pos_moc = flatten_mocs[pos_inds]
        if num_pos > 0:
            

            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)

            assert len(flatten_bbox_targets_reshape) == len(flatten_labels_targets_reshape) == len(flatten_bbox_preds_reshape)
            bbox_preds_moc,bbox_targets_moc,conv_moc= self.compute_bbox_per_image(flatten_bbox_targets_reshape,
                                                                          flatten_labels_targets_reshape,
                                                                          flatten_bbox_preds_reshape,flatten_points_reshape,
                                                                          flatten_conv_reshape,bg_class_ind)
            moc_result,conv_mocs,loss_conv_moc_for_clcs = self.moc_overlap(
                bbox_preds_moc,
                bbox_targets_moc,
                conv_moc,
                imgs)
            #print(bbox_preds_moc)
            loss_moc = self.loss_moc(
                conv_mocs.to(pos_centerness.device),
                moc_result.to(pos_centerness.device))
            # centerness weighted iou loss
            #print(moc_result.sum().to(pos_centerness.device),pos_centerness_targets-moc_result.to(pos_centerness_targets.device))
            #for nonzero_index in range(len(moc_result)):
             #   if moc_result[nonzero_index]==0:
            #        moc_result[nonzero_index] =moc_result[nonzero_index]+0.000001
            if moc_result is not None and not torch.any(moc_result > 0.):
                loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds,
                    weight=pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())
                
            else:
                loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds,
                    #weight=pos_centerness_targets,
                    #avg_factor=pos_centerness_targets.sum())
                    weight=moc_result.to(pos_centerness_targets.device),
                    avg_factor=moc_result.sum().to(pos_centerness_targets.device))
                
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_moc = pos_moc.sum()

        #loss_centerness = loss_centerness -loss_centerness+0.000001
        #loss_moc = loss_moc-loss_moc
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_moc = loss_moc,
            loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses','moc'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   mocs,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        #print(mocs,img_metas,cfg)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            moc_pred_list = [
                mocs[i][img_id].detach() for i in range(num_levels)
            ]
            new_centerness_pred_list = [(g+h)/1.5 for g, h in zip(centerness_pred_list, moc_pred_list )]
            #sqrt root the moc_pred_list ,
            #new_centerness_pred_list = torch.sqrt(new_centerness_pred_list)
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 #new_centerness_pred_list,
						 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            centerness = torch.sqrt(centerness)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level.
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.background_label  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
    def compute_bbox_per_image(self,flatten_bbox_targets_reshape,flatten_labels_targets_reshape,
                               flatten_bbox_preds_reshape,flatten_points_reshape,flatten_conv_reshape,bg_class_ind):
        #select bbox  per image level based on labels, and decode distance bbox
        bbox_targets_moc = []
        labels_targets_moc = []
        bbox_preds_moc = []
        conv_moc = []
        for bbox_targets, labels,bbox_preds,points,conv in zip(flatten_bbox_targets_reshape,
                                                   flatten_labels_targets_reshape,
                                                          flatten_bbox_preds_reshape,flatten_points_reshape,flatten_conv_reshape):
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().reshape(-1)
            #print(pos_inds)
            pos_bbox_preds =bbox_preds[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_points = points[pos_inds]
            pos_conv = conv[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            bbox_targets_moc.append(pos_decoded_target_preds)
            bbox_preds_moc.append(pos_decoded_bbox_preds)
            conv_moc.append(pos_conv)
            #print(pos_decoded_target_preds[:5,3]-pos_decoded_target_preds[:5,1],pos_decoded_target_preds[:5])
        return bbox_preds_moc,bbox_targets_moc,conv_moc

    def moc_overlap(self,pred, target,conv_moc,imgs,eps=1e-6):
        """MOC.

        Computing the MOC loss between a set of predicted bboxes and target bboxes.
        The loss is calculated as negative log of MOC.

        Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            
        Return:
            Tensor: Loss tensor.
        """
        loss_value = []
        loss_conv_moc = []
        loss_conv_moc_for_clcs = []
        #print(len(conv_moc))
        assert len(imgs)!=0
        #print(imgs.shape)
        for pred_moc, gt_moc, img, conv in zip(pred, target,imgs,conv_moc):
            _,sizex,sizey = img.shape
            #print(img.shape)
            thred = 0.5
            #newimg = torch.zeros((sizex,sizey)).to(img.device)
            #newimg_tgt = torch.zeros((sizex,sizey)).to(img.device)
            #newimg3 = torch.ones((sizex,sizey)).to(img.device)
            #img[0] =torch.add(img[0]*58.395, 123.675)
            #img[1] =torch.add(img[1]*57.12, 116.28)
            #img[2] =torch.add(img[2]*57.375,103.53)
            #ious = bbox_overlaps(pred_moc, gt_moc, is_aligned=True).clamp(min=eps)
            
            assert len(pred_moc) == len(gt_moc) ==len(conv)
            loss_conv_moc.append(conv)
            #print(conv[ious>=thred])
            #ordered, indices = torch.sort(ious,descending=True)
            #print(ordered, indices)
            for single_pred, single_gt_moc in zip(pred_moc.int(), gt_moc.int()):
                #newimg[single_pred[0]:single_pred[2],single_pred[1]:single_pred[3]] = 1
                #newimg_tgt[single_gt_moc[0].int():single_gt_moc[2],single_gt_moc[1]:single_gt_moc[3]] = 1
                #print(single_pred,single_gt_moc)
                #mask = newimg3*((newimg+newimg_tgt)==2).float()
                #overlapping_region = img*newimg
                #img0 =torch.add(overlapping_region[0]*58.395, 123.675)
                #img1 =torch.add(overlapping_region[1]*57.12, 116.28)
                #img2 =torch.add(overlapping_region[2]*57.375,103.53)
                #img0 = overlapping_region[0]
                #img1 = overlapping_region[1]
                #img2 = overlapping_region[2]
                sizex = single_pred[2]-single_pred[0]
                sizey = single_pred[3]-single_pred[1]
                #print(single_pred,sizex,sizey)
                if sizex >=1 and sizey >=1:
                    img0 = img[0,single_pred[0]:single_pred[2],single_pred[1]:single_pred[3]]
                    #mean_0 = torch.mean(img0.contiguous().view(-1))
                    img2 = img[2,single_pred[0]:single_pred[2],single_pred[1]:single_pred[3]]
                    #mean_2 = torch.mean(img2.contiguous().view(-1))
                    #xm = img0.contiguous().view(-1).sub(mean_0)
                    #ym = img2.contiguous().view(-1).sub(mean_2)
                    #r_num = xm.dot(ym)
                    #r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
                    #r_val = r_num / r_den
                    #img3 =(img[1]*57.12)
                    #print(torch.add(img3,116.28))
                    #use SRCC instead of pcc https://jcs.biologists.org/content/joces/131/3/jcs211847.full.pdf
                    #-1 or 1 means corr
                    '''
                    img0 = img0-img0.mean()
                    img2 = img2-img2.mean()
                    newsum = ((img0*img2)).sum()
                    new_std = (img0.pow(2).sum()*img2.pow(2).sum()).sqrt()
                    over_loss = newsum/new_std
                    over_loss = torch.abs(over_loss)
                    '''
                    img0 = torch.flatten(img0)
                    img2 = torch.flatten(img2)
                    img0_rank = torch.argsort(img0,dim = 0)+1
                    img2_rank = torch.argsort(img2,dim = 0)+1
                    img0_rank = (img0_rank - img0_rank.float().mean()).float()
                    img2_rank = (img2_rank - img2_rank.float().mean()).float()
                    newsum = ((img0_rank*img2_rank)).sum()
                    new_std = (img0_rank.pow(2).sum()*img2_rank.pow(2).sum()).sqrt()
                    over_loss = newsum/new_std
                    over_loss = torch.abs(over_loss)
                    #over_loss_tensor  = torch.tensor(over_loss**(1/2),dtype=torch.float)

                
                    if torch.isnan(over_loss):
                        loss_value.append(torch.tensor(0, dtype=torch.float).unsqueeze(0))
                        loss_conv_moc_for_clcs.append(torch.tensor(1, dtype=torch.float).unsqueeze(0))
                        #loss_conv_moc.append(torch.tensor(0, dtype=torch.float).unsqueeze(0))
                    else:
                        #over_loss_tensor = over_loss.clone().detach().requires_grad_(True)
                        over_loss_tensor  = torch.tensor(over_loss,dtype=torch.float)
                        loss_value.append(over_loss_tensor.unsqueeze(0))
                        loss_conv_moc_for_clcs.append(1-over_loss_tensor.unsqueeze(0))

                    '''
                    elif over_loss >= torch.tensor(0.01, dtype=torch.float).to(img.device):
                    loss_value.append(over_loss_tensor.pow(1/2).unsqueeze(0))
                    else:
                        loss_value.append(torch.tensor(0.01**(1/2), dtype=torch.float).unsqueeze(0))
                        #loss_conv_moc.append(torch.tensor(0, dtype=torch.float).unsqueeze(0))
                    '''
                else:
                    loss_value.append(torch.tensor(0, dtype=torch.float).unsqueeze(0))
                    loss_conv_moc_for_clcs.append(torch.tensor(1, dtype=torch.float).unsqueeze(0))
        return torch.cat(loss_value),torch.cat(loss_conv_moc),torch.cat(loss_conv_moc_for_clcs)

    def convertlevel2img(self,flatten_bbox_targets,flatten_labels,flatten_bbox_preds,flatten_points,flatten_mocs,num_points,num_imgs):
            bbox_targets_len = 0
            bbox_index = 0
            flatten_bbox_targets_moc = []
            flatten_bbox_targets_reshape = []
            flatten_labels_targets_moc = []
            flatten_labels_targets_reshape= []
            flatten_bbox_preds_moc = []
            flatten_bbox_preds_reshape = []
            flatten_points_moc = []
            flatten_points_reshape = []
            flatten_conv_moc= []
            flatten_conv_reshape = []
            #convert from concat from level to per image
            for _num in num_points:
                flatten_bbox_targets_moc_level = []
                flatten_labels_targets_moc_level = []
                flatten_bbox_preds_moc_level = []
                flatten_points_moc_level = []
                flatten_conv_moc_level = []
                for _num_img in range(num_imgs):
                    flatten_bbox_targets_moc_level.append(flatten_bbox_targets[bbox_targets_len:bbox_targets_len+_num])
                    flatten_labels_targets_moc_level.append(flatten_labels[bbox_targets_len:bbox_targets_len+_num])
                    flatten_bbox_preds_moc_level.append(flatten_bbox_preds[bbox_targets_len:bbox_targets_len+_num])
                    flatten_points_moc_level.append(flatten_points[bbox_targets_len:bbox_targets_len+_num])
                    flatten_conv_moc_level.append(flatten_mocs[bbox_targets_len:bbox_targets_len+_num])
                    bbox_targets_len+=_num
                flatten_bbox_targets_moc.append(torch.cat(flatten_bbox_targets_moc_level)) 
                flatten_labels_targets_moc.append(torch.cat(flatten_labels_targets_moc_level)) 
                flatten_bbox_preds_moc.append(torch.cat(flatten_bbox_preds_moc_level))
                flatten_points_moc.append(torch.cat(flatten_points_moc_level))
                flatten_conv_moc.append(torch.cat(flatten_conv_moc_level))
            for _num_img_index in range(num_imgs):
                flatten_bbox_targets_reshape_temp = []
                flatten_labels_targets_reshape_temp = []
                flatten_bbox_preds_reshape_temp = []
                flatten_points_reshape_temp = []
                flatten_conv_reshape_temp = []
                for _num_point_index in range(len(num_points)):
                    flatten_bbox_targets_reshape_temp.append(
                        flatten_bbox_targets_moc[_num_point_index][_num_img_index*num_points[_num_point_index]:
                                                               (_num_img_index+1)*num_points[_num_point_index]])
                    flatten_labels_targets_reshape_temp.append(
                        flatten_labels_targets_moc[_num_point_index][_num_img_index*num_points[_num_point_index]:
                                                               (_num_img_index+1)*num_points[_num_point_index]])
                    flatten_bbox_preds_reshape_temp.append(
                        flatten_bbox_preds_moc[_num_point_index][_num_img_index*num_points[_num_point_index]:
                                                               (_num_img_index+1)*num_points[_num_point_index]])
                    flatten_points_reshape_temp.append(
                        flatten_points_moc[_num_point_index][_num_img_index*num_points[_num_point_index]:
                                                               (_num_img_index+1)*num_points[_num_point_index]])
                    flatten_conv_reshape_temp.append(
                        flatten_conv_moc[_num_point_index][_num_img_index*num_points[_num_point_index]:
                                                               (_num_img_index+1)*num_points[_num_point_index]])
                    
                flatten_bbox_targets_reshape.append( torch.cat(flatten_bbox_targets_reshape_temp))
                flatten_labels_targets_reshape.append(torch.cat(flatten_labels_targets_reshape_temp))
                flatten_bbox_preds_reshape.append(torch.cat(flatten_bbox_preds_reshape_temp))
                flatten_points_reshape.append(torch.cat(flatten_points_reshape_temp))
                flatten_conv_reshape.append(torch.cat(flatten_conv_reshape_temp))
            
            return [flatten_bbox_targets_reshape,flatten_labels_targets_reshape,flatten_bbox_preds_reshape,flatten_points_reshape,flatten_conv_reshape]
