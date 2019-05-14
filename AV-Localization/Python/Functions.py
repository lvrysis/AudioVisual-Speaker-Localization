import math as mt
import cv2 as cv
import numpy as np
import csv
from scipy.signal import correlate2d


# pre_gray = np.random.rand(240, 320).astype(np.uint8)
# pre_color = np.random.rand(240, 320, 3).astype(np.uint8)
# pre_roi_gray = np.random.rand(10, 10).astype(np.uint8)
# pre_roi_color = np.random.rand(10, 10, 3).astype(np.uint8)

time = 0
buffer = []
lister = []

def open_gt():
    global lister
    with open('E:/Desktop/PhD/Datasets/M3C Speakers Localization v3/00001-FRAMES.csv', 'r') as f:
        reader = csv.reader(f)
        lister = list(reader)


def detect_face(frame, analysis, fps):

    global time, buffer, lister #, pre_gray, pre_color, pre_roi_gray, pre_roi_color

    face_cascade = cv.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('Data/haarcascade_eye.xml')
    mouth_cascade = cv.CascadeClassifier('Data/haarcascade_mcs_mouth.xml')

    color = frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # pre_gray = gray

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for f in range(0, len(faces)):

        fx, fy, fw, fh = faces[f][0], faces[f][1], faces[f][2], int(faces[f][3]*1.15)

        # cv.rectangle(color, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

        face_gray = gray[fy:fy + fh, fx:fx + fw]
        face_color = color[fy:fy + fh, fx:fx + fw]

        dict = {'fx': fx, 'fy': fy, 'fw': fw, 'fh': fh, 'face': face_gray, 'mx': None, 'my': None, 'mw': None, 'mh': None, 'mouth': None}

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_gray)
        for ex, ey, ew, eh in eyes:
            if ey < fh*0.5:
                eye_gray = face_gray[ey:ey + eh, ex:ex + ew]
                eye_color = face_color[ey:ey + eh, ex:ex + ew]
                # cv.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect mouths
        mouths = mouth_cascade.detectMultiScale(face_gray)
        for mx, my, mw, mh in mouths:

            if fh*0.55 < my+mh/2 < fh*0.85 and fw*0.35 < mx+mw/2 < fw*0.65:

                new_mw = int(fw*0.5)
                new_mh = int(new_mw*0.6)

                mx = mx + int(mw*0.5) - int(new_mw*0.5)
                my = my + int(mh*0.4) - int(new_mh*0.5)
                mw = new_mw
                mh = new_mh

                mouth_gray = face_gray[my:my + mh, mx:mx + mw]
                mouth_color = face_color[my:my + mh, mx:mx + mw]

                dict = {'fx': fx, 'fy': fy, 'fw': fw, 'fh': fh, 'face': face_gray, 'mx': mx, 'my': my, 'mw': mw, 'mh': mh, 'mouth': mouth_gray}

                break

        # Fill buffer
        added = False
        for bf in range(0, len(buffer)):
            for bt in range(0, len(buffer[bf])):
                if mt.fabs(buffer[bf][bt]['fx'] - fx) < 100:
                    buffer[bf].append(dict)
                    added = True
                    # print('added at:', bf, ',', len(buffer[bf]) - 1)
                    break
            if added is True:
                break
        if added is False:
            buffer.append([])
            buffer[len(buffer) - 1].append(dict)
            # print('added at:', len(buffer) - 1, ', 0')

    time += 1
    buffer_length = 12
    buffer_confidence_length = 2

    # Remove previous instances from each face
    for bf in range(0, len(buffer)):
        if time % 2 == 1 and len(buffer[bf]) > 0:
            buffer[bf].pop(0)
        if len(buffer[bf]) > buffer_length:
            buffer[bf].pop(0)

    # Remove a face with no instances
    for bf in range(len(buffer)-1, -1, -1):
        if len(buffer[bf]) == 0:
            buffer.pop(bf)

    # Count number of faces
    faces = 0
    for bf in range(0, len(buffer)):
        if len(buffer[bf]) >= buffer_confidence_length:
            faces += 1

    output_cur = [float(time)/fps, color.shape[1], color.shape[0], faces]
    cv.putText(color, 'Time: {:.2f}s'.format(float(time)/fps), (5, frame.shape[0] - 10), cv.FONT_ITALIC, 0.3, (0, 0, 0))

    # Perform visual recognitions
    for bf in range(0, len(buffer)):

        speaks = None
        delta = None

        # Visualize faces
        if len(buffer[bf]) >= buffer_confidence_length:
            mfx = int(buffer[bf][len(buffer[bf]) - 1]['fx'])
            mfy = int(buffer[bf][len(buffer[bf]) - 1]['fy'])
            mfw = int(buffer[bf][len(buffer[bf]) - 1]['fw'])
            mfh = int(buffer[bf][len(buffer[bf]) - 1]['fh'])
            cv.putText(color, '{}'.format(len(buffer[bf])), (mfx+8, mfy+16), cv.FONT_ITALIC, 0.4, (255, 255, 255))
            cv.rectangle(color, (mfx, mfy), (mfx + mfw, mfy + mfh), (255, 0, 0), 2)

        # Calculate mouth movement
        if len(buffer[bf]) >= buffer_length and buffer[bf][len(buffer[bf]) - 1]['mouth'] is not None and buffer[bf][len(buffer[bf]) - 2]['mouth'] is not None:
            mmx = int(buffer[bf][len(buffer[bf]) - 1]['mx'])
            mmy = int(buffer[bf][len(buffer[bf]) - 1]['my'])
            mmw = int(buffer[bf][len(buffer[bf]) - 1]['mw'])
            mmh = int(buffer[bf][len(buffer[bf]) - 1]['mh'])
            cv.rectangle(color, (mfx+mmx, mfy+mmy), (mfx+mmx+mmw, mfy+mmy+mmh), (0, 0, 255), 2)
            mouth_cur = buffer[bf][len(buffer[bf]) - 1]['face'][mmy:mmy + mmh, mmx:mmx + mmw]
            mouth_pre = buffer[bf][len(buffer[bf]) - 2]['face'][mmy:mmy + mmh, mmx:mmx + mmw]
            speaks, delta, mouth_delta, diff = motion_by_delta(mouth_cur, mouth_pre)

            s = int(int(mfx+mfw/2)/(1280/4))
            print("Frame {},  speaker x {}, id {} and gt {}".format(time, int(mfx+mfw/2), s, lister[time][0]))
            if str(lister[time][0]).find(str(s)) >= 0:
                cv.imwrite('E:/Desktop/PhD/Datasets/M3C Speakers Localization v3/Mouths (Raw)/1/1-{:01d}-{:05d}.jpg'.format(s, time), mouth_cur)
                cv.imwrite('E:/Desktop/PhD/Datasets/M3C Speakers Localization v3/Mouths (Diff)/1/1-{:01d}-{:05d}.jpg'.format(s, time), diff)
            else:
                cv.imwrite('E:/Desktop/PhD/Datasets/M3C Speakers Localization v3/Mouths (Raw)/0/1-{:01d}-{:05d}.jpg'.format(s, time), mouth_cur)
                cv.imwrite('E:/Desktop/PhD/Datasets/M3C Speakers Localization v3/Mouths (Diff)/0/1-{:01d}-{:05d}.jpg'.format(s, time), diff)

        # Prepare mouth movement image
        if speaks is not None and int(mfx-mfw/2) > 0 and int(mfx-mfw/2) + mouth_delta.shape[1] < color.shape[1]:
            color[:mouth_delta.shape[0], int(mfx - mfw / 2):int(mfx - mfw / 2) + mouth_delta.shape[1], 0] = mouth_delta
            color[:mouth_delta.shape[0], int(mfx - mfw / 2):int(mfx - mfw / 2) + mouth_delta.shape[1], 1] = mouth_delta
            color[:mouth_delta.shape[0], int(mfx - mfw / 2):int(mfx - mfw / 2) + mouth_delta.shape[1], 2] = mouth_delta

        # Visualize mouths and output analysis
        if len(buffer[bf]) >= buffer_confidence_length:
            label_height = 60
            cv.rectangle(color, (int(mfx-mfw/2), label_height-10), (int(mfx-mfw/2) + 100, label_height+5), (0, 0, 0), cv.FILLED)
            cv.putText(color, 'Speaks: {}'.format(speaks), (int(mfx-mfw/2), label_height), cv.FONT_ITALIC, 0.3, (255, 255, 255))
            output_cur += (mfx + int(mfw / 2), mfy + int(mfh / 2), mfw, mfh, delta)

    analysis.append(output_cur)


def motion_by_correlation(im1, im2):

    a = (im1 - np.mean(im1)).astype(float) / np.std(im1)
    b = (im2 - np.mean(im2)).astype(float) / np.std(im2)
    corr = correlate2d(im1, im2)
    corr = (corr + np.min(corr)) / (np.max(corr) - np.min(corr))

    return corr


def motion_by_delta(im1, im2):

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 1000;

    # Specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    try:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)

        if warp_mode == cv.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im3 = cv.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im3 = cv.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        delta = cv.absdiff(im1, im3)
        pad_size = int(im1.shape[0] * 0.2)
        delta = delta[pad_size:-pad_size, pad_size:-pad_size]
        delta = np.lib.pad(delta, (pad_size, pad_size), 'constant', constant_values=0)
        diff = delta
        delta = cv.threshold(delta, 15, 255, cv.THRESH_BINARY)[1]
        mouths = np.concatenate((im1, im2, im3, delta), axis=1)

        delta = np.mean(delta)

        if delta > 1.0:
            speaks = True
        else:
            speaks = False

    except:
        return None, 0, None, None

    return speaks, delta, mouths, diff


def motion_by_flow(im1, im2):

    flow = cv.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # flow = cv.calcOpticalFlowFarneback(im1, im2, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3)).astype(np.uint8)
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = mag * 40  # cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


def motion_by_histogram(im1, im2):

    hist = 0
    for i in range(0, 3):
        new_hist = cv.calcHist([im1], [i], None, [256], [0, 256])
        pre_hist = cv.calcHist([im2], [i], None, [256], [0, 256])
        hist += np.sum(np.fabs(new_hist[:, 0] - pre_hist[:, 0])) / new_hist.shape[0]

    return hist


def histogram(frame):

    histogram = cv.calcHist([frame], [0], None, [256], [0, 256])
    centroid = 0

    for i in range(0, len(histogram)):
        centroid += histogram[i]

    centroid /= len(histogram)


