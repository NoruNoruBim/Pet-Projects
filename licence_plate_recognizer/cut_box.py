import cv2
from hough import transform


def viewImage(image, name="Display"):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(img_name, show=False):
    #   get coordinates of the box (rectangle)
    boxes = []
    with open(r"rois.txt", 'r') as file:
        for line in file:
            boxes += [[]]
            for i in line.split():
                boxes[-1] += [int(i)]
    if show:
        print(boxes)

    #   read original image
    orig = cv2.imread(r"images/" + img_name)

    width = len(orig[0])
    height = len(orig)

    if show:
        print(width, height)

    offset_x = 0 
    offset_y = 0
    k = 0

    if width > height:
        k = 1024 / width
        offset_y = (1024 - height * k) // 2
    else:
        k = 1024 / height
        offset_x = (1024 - width * k) // 2

    dim = (int(width * k), int(height * k))

    if show:
        print(dim)

    orig = cv2.resize(orig, dim, interpolation = cv2.INTER_AREA)#       try bound without resize

    ind = 0
    for i in range(len(boxes)):
        y1, x1, y2, x2 = boxes[i]
        h = abs(y1 - y2)
        w = abs(x2 - x1)
        if h / w < 0.8 and h / w > 0.15:
            ind = i
            break

    x1 = boxes[ind][1] - int(offset_x)
    y1 = boxes[ind][0] - int(offset_y)

    x2 = boxes[ind][3] - int(offset_x)
    y2 = boxes[ind][2] - int(offset_y)

    if show:
        print(x1, y1, x2, y2)

    h = y1 - y2
    w = x2 - x1
    
    x1 -= int(w * 0.07)
    x2 += int(w * 0.07)
    y1 += int(h * 0.25)
    y2 -= int(h * 0.22)

    out = orig.copy()[y1 : y2, x1 : x2]
    if show:
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 3)

    if show:
        viewImage(orig, "box")
        viewImage(out, "box")
    out = transform(out, show)#     normalize box
    cv2.imwrite("box.png", out)
    
    return orig, x1, y1, abs(x2 - x1), abs(y1 - y2)
