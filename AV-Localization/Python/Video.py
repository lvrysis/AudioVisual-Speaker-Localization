import cv2 as cv
import numpy as np
import Functions as functions


# Global variables
frames = []
analysis = [['TIME', 'FRAME-WIDTH', 'FRAME-HEIGHT', 'PERSONS',
             'P1-X', 'P1-Y', 'P1-WIDTH', 'P1-HEIGHT', 'P1-SPEAKS',
             'P2-X', 'P2-Y', 'P2-WIDTH', 'P2-HEIGHT', 'P2-SPEAKS',
             'P3-X', 'P3-Y', 'P3-WIDTH', 'P3-HEIGHT', 'P3-SPEAKS',]]


# Capture video from device
def capture(width, height, fps):

    global frames, analysis

    # capture video from a device and write the frame into a buffer
    capture = cv.VideoCapture(0) # the argument denotes the device index
    capture.set(3, width)
    capture.set(4, height)
    capture.set(5, fps)

    while capture.isOpened:

        # read current frame
        ret, frame = capture.read()

        # stop if video is finished or q key is pressed
        key_pressed = cv.waitKey(40)
        if ret is False or key_pressed == ord("q"):
            break

        # perform any processing
        # frame = np.concatenate((frame, frame), axis=1)
        functions.detect_face(frame, analysis, fps)
        # functions.histogram(frame)

        # frame = cv.GaussianBlur(frame, (5, 5), 0)
        # frame = cv.medianBlur(frame, 7)
        # frame = cv.bilateralFilter(frame, 9, 75, 75)

        # display current frame
        cv.imshow('Camera', frame)

        # write current frame into frame buffer
        frames.append(frame)

    # release the capture object
    capture.release

    return frames, analysis


# Process video from a file
def read(path, fps):

    functions.open_gt()

    global frames, analysis

    capture = cv.VideoCapture(path)
    # fps = capture.get(cv.CAP_PROP_FPS)

    while capture.isOpened:

        # read the file frame by frame
        ret, frame = capture.read()

        # stop if video is finished or q key is pressed
        key_pressed = cv.waitKey(100)
        if ret is False or key_pressed == ord("q"):
            break

        functions.detect_face(frame, analysis, fps)

        # display current frame
        cv.imshow("Video", frame)

    capture.release

    return frames, analysis


# Save video as a file
def save(path, frames, fps):

    # fourcc = -1
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    for i in range(0, len(frames)):
        out.write(frames[i])

    out.release()


# OpenCV Video Properties
'''
# 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
# 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
# 3. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
# 4. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# 5. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
# 6. CV_CAP_PROP_FPS Frame rate.
# 7. CV_CAP_PROP_FOURCC 4-character code of codec.
# 8. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
# 9. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
# 10. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
# 11. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
# 12. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
# 13. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
# 14. CV_CAP_PROP_HUE Hue of the image (only for cameras).
# 15. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
# 16. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
# 17. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
# 18. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
# 19. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
'''