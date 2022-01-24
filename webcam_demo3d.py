import matplotlib.pyplot as plt
import glob
import cv2
from matplotlib import projections
import numpy as np
import torch
from torch.functional import Tensor
from torchvision import transforms
import torchvision
from torchvision.transforms.transforms import ToPILImage
from models import pfld
from insightface.app import FaceAnalysis


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0
    size = int(old_size * 1.2)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return np.array(roi_box)


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def view3d(pts3d):
    x, y, z = pts3d

    ax.cla()
    ax.scatter(x, y, z, c='r', marker='.', s=0.5, linewidth=3)
    plt.show(block=False)
    plt.pause(0.001)


def get_detector():
    app = FaceAnalysis("buffalo_s", allowed_modules=[
        'detection'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_thresh=0.5, det_size=(416, 416))
    return app


def predict(detector, model, img):

    objs = detector.get(img)

    for obj in objs:
        box = parse_roi_box_from_bbox(obj.bbox)
        face_img = crop_img(img, box).copy()
        cv2.imshow('face',face_img)
        w, h = box[2:] - box[:2]

        transform = transforms.Compose([
            ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        lmk, angles, prob = model(transform(face_img).to(0).unsqueeze(0))
        angles = angles/3.14*180

        angles = angles.cpu().detach().numpy()[0]
        lmk = lmk.cpu().detach().numpy()[0]

        lmk = lmk*0.5+0.5
        lmk *= w

        view3d(lmk)

        cv2.putText(img, "X: " + "{:7.2f}".format(angles[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        cv2.putText(img, "Y: " + "{:7.2f}".format(angles[1]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        cv2.putText(img, "Z: " + "{:7.2f}".format(angles[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)

        for i in range(lmk.shape[1]):
            cv2.circle(img, (int(box[0]+round(lmk[0, i])),
                             int(box[1]+round(lmk[1, i]))), 2, (0, 0, 255), -1)

        break


def get_model():
    model = pfld.getmodel()
    state_dict = torch.load("backbone3d.pth")
    model.load_state_dict(state_dict)
    model.to(0)
    model.eval()
    return model


def main():
    model = get_model()
    detector = get_detector()

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()

        predict(detector, model, img)

        cv2.imshow("1", img)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
