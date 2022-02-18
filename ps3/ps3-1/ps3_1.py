import cv2
import sys
import numpy as np

MAXVALUE = 255

win_original_name = "Original Image"
win_improved_name = "Improved Image"


#  insert string before
def insert_name(name, str2add):
    dot_idx = name.find(".")
    new_name = name[:dot_idx] + str2add + name[dot_idx:]
    return new_name


# #  image sharpening
# def sharpen(img_in, hor_vert=False):
#     if hor_vert:
#         kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     else:
#         kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#     img_out = cv2.filter2D(img_in, -1, kernel)  # negative to be same depth as input
#     return img_out


#  unsharp image sharpening
def unsharp(img_in):
    smoothed = cv2.GaussianBlur(img_in, (9, 9), 3)  # gaussian blur mask kernel size 9x9 standard div 3
    img_out = cv2.addWeighted(img_in, 2.5, smoothed, -1.5, 0)  # 2.5f1-1.5f2
    return img_out


if __name__ == "__main__":
    # get input arguments
    args = sys.argv
    assert (len(args) == 2)  # make sure two arguments input
    img_name = args[1]  # input image path

    cv2.namedWindow(win_original_name)
    cv2.namedWindow(win_improved_name)

    # read original image
    img = cv2.imread(img_name)
    if img is None:
        sys.exit("Could not read the image.")
    cv2.imshow(win_original_name, img)

    # hardcode image process for known examples
    if img_name == "golf.png":
        img_output = cv2.medianBlur(img, 3)  # median filter
    elif img_name == "pots.png":
        img_output = unsharp(img)  # unsharp sharpen
    elif img_name == "rainbow.png":
        img_output = cv2.bilateralFilter(img, 15, 80, 80)  # bilateral filter
    elif img_name == "pcb.png":
        img_output = cv2.medianBlur(img, 3)  # first median filter
        img_output = unsharp(img_output)  # then unsharp sharpen
    else:
        print("Adjust manually")
        img_output = img  # manually adjust later if unknown

    print("Press 'S' to save and exit")
    while True:
        key = cv2.waitKey(30)  # wait 30ms
        cv2.imshow(win_improved_name, img_output)

        if key == ord("s"):
            #  press 'S' to save
            img_output_name = insert_name(img_name, "-improved")
            cv2.imwrite(img_output_name, img_output)
            break

        # manually adjust
        elif key == ord("z"):
            #  press 'z' to restore origin img
            img_output = img

        elif key == ord("u"):
            #  press 'u' to use unsharp
            img_output = unsharp(img_output)

        elif key == ord("m"):
            #  press 'm' to use median filter
            img_output = cv2.medianBlur(img_output, 3)

        elif key == ord("g"):
            #  press 'g' to use gaussian filter
            img_output = cv2.GaussianBlur(img_output, (5, 5), 0)  # kernel size 5x5 standard div 0

        elif key == ord("b"):
            #  press 'b' to use bilateral filter

            img_output = cv2.bilateralFilter(img_output, -1, 0, 50)  # less color merge and large space merge

        elif key == ord("q") or key == ord("x") or key == 27:
            # press 'q', 'x', 'ESC' to quit
            break

    cv2.destroyAllWindows()
