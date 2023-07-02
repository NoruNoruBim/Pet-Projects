import cv2
import numpy as np
from itertools import combinations


def viewImage(image, name="Display"):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# y = kx + b
def find_k(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)

def find_b(x1, y1, x2, y2):
    return -x1 * (y2 - y1) / (x2 - x1) + y1


#   continue line to the end of the image
def bigger_line(x1, y1, x2, y2, img):
    k = find_k(x1, y1, x2, y2)
    b = -x1 * (y2 - y1) / (x2 - x1) + y1

    tmp_x1 = 0
    tmp_y1 = tmp_x1 * k + b
    
    tmp_x2 = len(img[0])
    tmp_y2 = tmp_x2 * k + b
    
    return int(tmp_x1), int(tmp_y1), int(tmp_x2), int(tmp_y2)


#   try to find two correct and parallel lines in Hough lines set
def get_4_points(img, show=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    if show:
        viewImage(edges, 'houghlines')

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180 , 100, minLineLength, maxLineGap)
    if show:
        print("lines:", len(lines))
        
    '''
    github
    here is removed code
    '''

    return [(x1b1, y1b1), (x2b1, y2b1), (x1b2, y1b2), (x2b2, y2b2)]


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped


def transform(image, show=False):
    pts = np.array(get_4_points(image, show), dtype = "float32")

    warped = four_point_transform(image, pts)
    if show:
        viewImage(warped, "warped")

    return warped
