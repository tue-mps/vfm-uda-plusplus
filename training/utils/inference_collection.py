#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as FV


@torch.no_grad()
def multiscale_inference(network, img, crop_size, num_classes, ratios=(0.6, 1.0, 1.7, 2.3)):
    b, c, h, w = img.shape
    orig_img_size = (h, w)
    img_sizes = [(int(orig_img_size[0] * r), int(orig_img_size[1] * r)) for r in ratios]

    rv_logit = []
    rv_segmentation = []
    for (h, w) in img_sizes:
        for flip in [True, False]:
            if flip:
                x = FV.hflip(img)
            else:
                x = img

            x = F.interpolate(x, size=(h, w), mode="bilinear")
            logit, pred = slide_inference_batched(network, x, crop_size, num_classes)
            logit = F.interpolate(logit, size=orig_img_size, mode="bilinear")
            pred = F.interpolate(pred, size=orig_img_size, mode="bilinear")

            rv_logit += [logit.unsqueeze(0)]
            rv_segmentation += [pred.unsqueeze(0)]

    rv_logit = torch.cat(rv_logit, dim=0).mean(0)
    rv_segmentation = torch.cat(rv_segmentation, dim=0).mean(0)
    return rv_logit, rv_segmentation


@torch.no_grad()
def multiscale_inference_batched(network, img, crop_size, num_classes, ratios=(1.0, 1.3, 1.7)):
    batch_size, c, h_img, w_img = img.shape
    orig_img_size = (h_img, w_img)
    img_sizes = [(int(orig_img_size[0] * r), int(orig_img_size[1] * r)) for r in ratios]

    all_imgs = []
    metadata = []

    for (h_new, w_new) in img_sizes:
        coords = []
        img_resized = F.interpolate(img, size=(h_new, w_new), mode="bilinear", align_corners=False)

        h_crop, w_crop = crop_size, crop_size
        h_stride, w_stride = crop_size // 2, crop_size // 2
        h_grids = max(h_new - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_new - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_new)
                x2 = min(x1 + w_crop, w_new)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                img_cropped = img_resized[:, :, y1:y2, x1:x2]
                all_imgs.append(img_cropped)
                all_imgs.append(FV.hflip(img_cropped))
                coords.append((y1, y2, x1, x2, False))
                coords.append((y1, y2, x1, x2, True))
        metadata.append(((h_new, w_new), coords))

    all_imgs = torch.cat(all_imgs, dim=0)
    crop_seg_logits = network(all_imgs)  # Single forward pass
    crop_seg_preds = torch.softmax(crop_seg_logits, dim=1)

    rv_logits = torch.zeros((batch_size, num_classes, h_img, w_img), device=img.device, dtype=torch.float32)
    rv_preds = torch.zeros((batch_size, num_classes, h_img, w_img), device=img.device, dtype=torch.float32)

    idx = 0
    for (h_new, w_new), coords in metadata:
        tmp_logits = torch.zeros((batch_size, num_classes, h_new, w_new), device=img.device, dtype=torch.float32)
        tmp_preds = torch.zeros((batch_size, num_classes, h_new, w_new), device=img.device, dtype=torch.float32)
        count_mat = torch.zeros((batch_size, 1, h_new, w_new), device=img.device, dtype=torch.float32)
        for (y1, y2, x1, x2, flip) in coords:
            tmp_crop_logit = crop_seg_logits[idx * batch_size:(idx + 1) * batch_size]
            tmp_crop_pred = crop_seg_preds[idx * batch_size:(idx + 1) * batch_size]
            if flip:
                tmp_crop_logit = FV.hflip(tmp_crop_logit)
                tmp_crop_pred = FV.hflip(tmp_crop_pred)

            tmp_logits += F.pad(
                tmp_crop_logit,
                (int(x1),
                 int(tmp_logits.shape[3] - x2),
                 int(y1),
                 int(tmp_logits.shape[2] - y2))
            )
            tmp_preds += F.pad(
                tmp_crop_pred,
                (int(x1),
                 int(tmp_preds.shape[3] - x2),
                 int(y1),
                 int(tmp_preds.shape[2] - y2))
            )
            count_mat[:, :, y1:y2, x1:x2] += 1
            idx += 1

        assert (count_mat == 0).sum() == 0
        tmp_logits /= count_mat
        tmp_preds /= count_mat

        rv_logits += F.interpolate(tmp_logits, size=(h_img, w_img), mode="bilinear", align_corners=False)
        rv_preds += F.interpolate(tmp_preds, size=(h_img, w_img), mode="bilinear", align_corners=False)
    rv_logits /= len(ratios)
    rv_preds /= len(ratios)
    return rv_logits, rv_preds


@torch.no_grad()
def slide_inference(network, img, crop_size, num_classes):
    batch_size, _, h_img, w_img = img.shape
    h_crop = crop_size
    w_crop = crop_size
    h_stride = crop_size // 2
    w_stride = crop_size // 2

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = torch.zeros((batch_size, num_classes, h_img, w_img)).to(img).float()
    count_mat = torch.zeros((batch_size, 1, h_img, w_img)).to(img).float()
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_seg_logit = network(crop_img)

            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0
    seg_logits = preds / count_mat

    return seg_logits


@torch.no_grad()
def slide_inference_batched(network, img, crop_size, num_classes):
    batch_size, _, h_img, w_img = img.shape
    h_crop = crop_size
    w_crop = crop_size
    h_stride = crop_size // 2
    w_stride = crop_size // 2

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    rv_logits = torch.zeros((batch_size, num_classes, h_img, w_img), device=img.device, dtype=torch.float32)
    rv_preds = torch.zeros((batch_size, num_classes, h_img, w_img), device=img.device, dtype=torch.float32)
    count_mat = torch.zeros((batch_size, 1, h_img, w_img), device=img.device, dtype=torch.float32)

    crop_imgs = []
    coords = []

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            crop_imgs.append(img[:, :, y1:y2, x1:x2])
            coords.append((y1, y2, x1, x2))

    crop_imgs = torch.cat(crop_imgs, dim=0)  # Concatenate all patches along batch dimension
    crop_seg_logits = network(crop_imgs)  # Single forward pass

    # Process predictions
    idx = 0
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1, y2, x1, x2 = coords[idx]
            rv_logits += F.pad(crop_seg_logits[idx * batch_size:(idx + 1) * batch_size],
                               (int(x1), int(rv_logits.shape[3] - x2), int(y1),
                                int(rv_logits.shape[2] - y2)))
            rv_preds += F.pad(torch.softmax(crop_seg_logits[idx * batch_size:(idx + 1) * batch_size], dim=1),
                              (int(x1), int(rv_preds.shape[3] - x2), int(y1),
                               int(rv_preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1
            idx += 1

    assert (count_mat == 0).sum() == 0
    rv_logits = rv_logits / count_mat
    rv_preds = rv_preds / count_mat

    return rv_logits, rv_preds


@torch.no_grad()
def slide_inference_pre_cropped(network, crops, preds, count_mat, coords, num_classes):
    # we clip the classes only for the binning network
    crops = network(crops)[:, :num_classes, :, :]
    crops = torch.softmax(crops, dim=1)

    for (y1, y2, x1, x2), crp in zip(coords, crops):
        preds[:, y1:y2, x1:x2] += crp

    seg_logits = preds / count_mat.float()
    return seg_logits
