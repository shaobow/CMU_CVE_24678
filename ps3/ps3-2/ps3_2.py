import cv2
import sys
import numpy as np

MAXVALUE = 255
thresh1 = 100  # init threshold value
thresh2 = 200
aperture = 3
l2grad = False
THRESH_MAX = MAXVALUE  # max gamma value
APERTURE_MAX = 7
APERTURE_MIN = 3

win_original_name = "Original Image"
win_detected_name = "Edge detection"


# insert string before
def insert_name(name, str2add):
    dot_idx = name.find(".")
    new_name = name[:dot_idx] + str2add + name[dot_idx:]
    return new_name


# trackbar helper function
def on_thresh1_trackbar(val):
    global thresh1
    thresh1 = val


def on_thresh2_trackbar(val):
    global thresh2
    thresh2 = val


def on_aperture_trackbar(val):
    global aperture
    aperture = max(APERTURE_MIN, val)
    if np.mod(aperture, 2) == 0:
        aperture = aperture + 1
    cv2.setTrackbarPos("Aperture", win_detected_name, pos=aperture)


def on_l2grad_checkbox(val):
    global l2grad
    if val == 1:
        l2grad = True
    else:
        l2grad = False


def my_filter(img_in, kernel):
    (M, N) = img_in.shape  # MxN array input
    (m, n) = kernel.shape  # mxn kernel input
    # apply padding to get original img size
    if m % 2 == 0:
        p_row = int(m/2)
    else:
        p_row = int((m-1)/2)
    if n % 2 == 0:
        p_col = int(n/2)
    else:
        p_col = int((n-1)/2)
    img_pad = cv2.copyMakeBorder(img_in, p_row, p_row, p_col, p_col, cv2.BORDER_DEFAULT)

    # apply kernel iteratively
    img_out = np.empty((M, N))  # datatype conversion later
    for i in np.arange(p_row, M+p_row):
        for j in np.arange(p_col, N+p_col):
            block = img_pad[i-p_row:i+p_row+1, j-p_col:j+p_col+1]
            fil_val = np.sum(np.multiply(block, kernel))
            img_out[[i-p_row], [j-p_col]] = fil_val

    # datatype conversion
    img_out = np.clip(img_out, 0, MAXVALUE).astype(np.uint8)
    return img_out


# sobel filter
def sobel(img_in):
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    img_x = my_filter(img_blur, kernel_x)
    img_y = my_filter(img_blur, kernel_y)
    img_out = cv2.addWeighted(img_x, 0.5, img_y, 0.5, 0)

    # img_test_x = cv2.Sobel(img_blur, -1, 1, 0)
    # img_test_y = cv2.Sobel(img_blur, -1, 0, 1)
    # img_test = cv2.addWeighted(img_test_x, 0.5, img_test_y, 0.5, 0)
    # cv2.imshow("opencv sobel", img_test)
    return img_out


if __name__ == "__main__":
    # get input arguments
    args = sys.argv
    assert (len(args) == 3)  # make sure two arguments input
    img_name = args[1]  # input image path
    if args[2] == "C" or args[2] == "c":
        use_canny = True  # use canny edge detector
    elif args[2] == "S" or args[2] == "s":
        use_canny = False  # use sobel filter
    else:
        sys.exit("Wrong input arguments.")

    cv2.namedWindow(win_original_name)
    cv2.namedWindow(win_detected_name)

    # read original image
    img = cv2.imread(img_name)
    if img is None:
        cv2.destroyAllWindows()
        sys.exit("Could not read the image.")

    cv2.imshow(win_original_name, img)

    if use_canny:
        thresh1_init = thresh1  # init position in trackbar
        thresh2_init = thresh2
        cv2.createTrackbar("Thresh1", win_detected_name, thresh1_init, THRESH_MAX, on_thresh1_trackbar)
        cv2.createTrackbar("Thresh2", win_detected_name, thresh2_init, THRESH_MAX, on_thresh2_trackbar)
        cv2.createTrackbar("Aperture", win_detected_name, aperture, APERTURE_MAX, on_aperture_trackbar)
        cv2.createTrackbar("L2gradient", win_detected_name, 0, 1, on_l2grad_checkbox)

        print("Press 'S' to save and exit")
        while True:
            img_output = cv2.Canny(img, thresh1, thresh2,
                                   apertureSize=aperture, L2gradient=l2grad)
            cv2.imshow(win_detected_name, img_output)

            key = cv2.waitKey(30)  # wait 30ms

            if key == ord("s"):
                # press 'S' to save
                img_output_name = insert_name(img_name, "-canny")
                cv2.imwrite(img_output_name, img_output)
                print("Parameters combination: " + str(thresh1) + "\t" + str(thresh2) + "\t"
                      + str(aperture) + "\t" + str(l2grad))
                break
            elif key == ord("q") or key == ord("x") or key == 27:
                # press 'q', 'x', 'ESC' to quit
                break
        cv2.destroyAllWindows()

    else:
        img_output = sobel(img)
        cv2.imshow(win_detected_name, img_output)

        print("Press 'S' to save and exit")
        key = cv2.waitKey(-1)

        if key == ord("s"):
            # press 'S' to save
            img_output_name = insert_name(img_name, "-sobel")
            cv2.imwrite(img_output_name, img_output)
            cv2.destroyAllWindows()

        elif key == ord("q") or key == ord("x") or key == 27:
            cv2.destroyAllWindows()
            # press 'q', 'x', 'ESC' to quit
