import cv2
import sys
import numpy as np


win_contours_name = "Recognized contours Image"


# insert string before
def insert_name(name, str2add):
    dot_idx = name.find(".")
    new_name = name[:dot_idx] + str2add + name[dot_idx:]
    return new_name


if __name__ == "__main__":
    # # get input arguments
    # args = sys.argv
    # assert (len(args) == 2)  # make sure two arguments input
    # img_name = args[1]  # input image path
    img_name = "spade-terminal.png"

    # Read image
    img = cv2.imread(img_name)

    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary
    thr, dst = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dst = cv2.erode(dst, kernel)
    for i in range(2):
        dst = cv2.dilate(dst, kernel)
    for i in range(1):
        dst = cv2.erode(dst, kernel)

    # find contours with hierarchy
    cont, hier = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find template candidate
    i = 1
    while cv2.matchShapes(cont[i], cont[i+1], cv2.CONTOURS_MATCH_I2, 0) >= 0.5:
        i += 1
    c0 = cont[i]

    # each contour calculate distance
    dist = np.zeros(len(cont))
    for i in range(0, len(cont)):
        c = cont[i]
        h = hier[0, i]
        if h[2] == -1 and h[3] == 0:
            dist[i] = cv2.matchShapes(c, c0, cv2.CONTOURS_MATCH_I2, 0)

    # sort distance for maximum outlier
    idx = (-dist).argsort()[:3]
    for i in idx:
        img = cv2.drawContours(img, cont, i, (0, 0, 255), -1)

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