import cv2
import sys
import numpy as np


win_contours_name = "Recognized contours Image"


# insert string before
def insert_name(name, str2add):
    dot_idx = name.find(".")
    new_name = name[:dot_idx] + str2add + name[dot_idx:]
    return new_name


# check size (bounding box) is square
def isSquare(siz):
    ratio = abs(siz[0] - siz[1]) / siz[0]
    # print (siz, ratio)
    if ratio < 0.1:
        return True
    else:
        return False


# check circle from the arc length ratio
def isCircle(cnt):
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    len = cv2.arcLength(cnt, True)
    ratio = abs(len - np.pi * 2.0 * radius) / (np.pi * 2.0 * radius)
    # print(ratio)
    if ratio < 0.1:
        return True
    else:
        return False


if __name__ == "__main__":
    # get input arguments
    args = sys.argv
    assert (len(args) == 2)  # make sure two arguments input
    img_name = args[1]  # input image path

    # Read image
    img = cv2.imread(img_name)

    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary
    thr, dst = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dst = cv2.erode(dst, kernel)
    for i in range(3):
        dst = cv2.dilate(dst, kernel)
    for i in range(2):
        dst = cv2.erode(dst, kernel)

    # find contours with hierachy
    cont, hier = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # each contoure
    for i in range(len(cont)):
        c = cont[i]
        h = hier[0, i]
        if h[2] == -1 and h[3] == 0:
            # no child and parent is image outer
            img = cv2.drawContours(img, cont, i, (0, 0, 255), -1)
        elif h[3] == 0 and hier[0, h[2]][2] == -1:
            # with child
            if isCircle(c):
                # child has circle outer
                if isCircle(cont[h[2]]):
                    # double circle
                    img = cv2.drawContours(img, cont, i, (0, 255, 0), -1)
                else:
                    # only outer circle
                    img = cv2.drawContours(img, cont, i, (128, 0, 128), -1)
            else:
                # child has no circle outer
                if isCircle(cont[h[2]]):
                    img = cv2.drawContours(img, cont, i, (0, 255, 255), -1)
                # 1 child and shape bounding box is not squre
                if not isSquare(cv2.minAreaRect(c)[1]) and hier[0, h[2]][0] == -1 and hier[0, h[2]][1] == -1:
                    img = cv2.drawContours(img, cont, i, (255, 0, 0), -1)

    # Show result
    cv2.namedWindow(win_contours_name)
    cv2.imshow(win_contours_name, img)
    print("Press 'S' to save and exit")
    key = cv2.waitKey(-1)

    if key == ord("s"):
        #  press 'S' to save
        img_output_name = insert_name(img_name, "-output")
        cv2.imwrite(img_output_name, img)
        cv2.destroyAllWindows()

    elif key == ord("q") or key == ord("x") or key == 27:
        # press 'q', 'x', 'ESC' to quit
        cv2.destroyAllWindows()