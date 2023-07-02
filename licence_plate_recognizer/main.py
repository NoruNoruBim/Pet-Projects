import cv2
import find_box
import cut_box
import recognize_symbols

import os


#   create .json file needed to mask r-cnn
def make_text(img_name):
    with open('template.json', 'r') as f:
        text = f.read()
    text = text.replace("filename.xxx", img_name)
    
    with open(r"images\via_region_data.json", 'w') as file:
        file.write(text)


if __name__ == '__main__':
    #   get image to recognize
    img_name = input("\t\t---    Enter name of image    ---\n")
    path_to_img = r"images/"

    if '\\' in img_name:
        img_name = img_name.split('\\')[-1]

    show = True if (input("\t\t---    Show temporary results?    ---\n")) == "y" else False

    #   create .json file to mask r-cnn
    make_text(img_name)

    #   find where licence plate is (coordinations of result locate in rois.txt)
    find_box.main(img_name, show=show)

    #   just cut licence plate from whole image and save separately
    orig, x, y, w, h = cut_box.main(img_name, show=show)

    #   recognize symbols on the plate
    labels = recognize_symbols.main(show=show)
    print(labels)

    #   a little improvement for some cases
    s = recognize_symbols.make_better(labels)

    #   draw final result on original image
    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(orig, s, (x + w, y + h * 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
    cv2.imshow("Recognised", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
