import numpy
import cv2

def readPicture():

    img = cv2.imread("Input/test-image-1.jpg")
    if img is None:
        print('Image not loaded')
    else:
        print('Image loaded')
        '''
        #Print a single pixel
        px = img[100, 100]
        print px
        '''

        '''
        #Image addition
        out = cv2.addWeighted(img, 0.2, img, 0.3, 1)
        cv2.imshow("Preview", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        #Image blurring
        kernel = numpy.ones((5, 5), numpy.float32) / 25
        dst = cv2.filter2D(img, -1, kernel)
        cv2.imshow("Window 1", img)
        cv2.imshow("Window 2", dst)
        cv2.waitKey(0)
        print(len(kernel))
        print(len(kernel[0]))
        for i in range(0, 5):
            print(kernel[0, i])

    return

# def openImage():
#
#     img = cv2.imread("F:\Desktop\iphone-se-colors.jpg")
#     if img is None:
#         print "Image not loaded"
#     else:
#
#         # Print a single pixel
#         px = img[100, 100]
#         print px
#
#         # Image addition
#         # out = cv2.addWeighted(img, 0.2, img, 0.3, 1)
#         # cv2.imshow("Preview", out)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#
#         # Image blurring
#         # kernel = numpy.ones((5, 5), numpy.float32) / 25
#         # dst = cv2.filter2D(img, -1, kernel)
#         # cv2.imshow("Window 1", img)
#         # cv2.imshow("Window 2", dst)
#         # cv2.waitKey(0)
#         # print len(kernel)
#         # print len(kernel[0])
#         # for i in range(0, 5):
#         #     print kernel[0, i]