from imutils import face_utils, resize
import dlib
import cv2
from scipy.spatial import distance as dist


def draw(img, xr, plot=False):
    if plot:
        for i in xr:
            if i != xr[-1]:
                (x0, y0) = img[i]
                (x, y) = img[i + 1]
                cv2.line(frame, (x0, y0), (x, y), (0, 0, 0), 2)

def landmarks(img, nums=True, plot=False):
    if plot:
        for i, (x, y) in enumerate(img):
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            if nums:
                cv2.putText(frame, f"{i + 1}", (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 2)

def ratio(eye):
    ''' An eye has 6 points according to dlib.                      -1-  -2-
    The opposite points are measured and the ratios are taken   -0-    eye   -3-
                                                                    -5-  -4-       '''
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    constant = 0.5 * (a + b) / c
    return constant

def blink (right_eye, left_eye, plot=False):
    if plot:
        temp = img[left_eye]
        l_eye = ratio(temp)
        if l_eye <=0.25: # because of mirror effect l_eye represents our right eye
            cv2.putText(frame, f"right eye closed {l_eye:.2}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)

        temp = img[right_eye]
        r_eye = ratio(temp)
        if r_eye <=0.25: # # because of mirror effect r_eye represents our left eye
            cv2.putText(frame, f"left eye closed {r_eye:.2}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)


face = range(0, 68)
chin = range(0, 17)
right_brow = range(17, 22)
left_brow = range(22, 27)
nose = range(27, 31)
dip_nose = range(31, 36)
right_eye = range(36, 42)
left_eye = range(42, 48)
mouth_in = range(48, 60)
mouth_out = range(61, 68)
parts = [chin, right_brow, left_brow, nose, dip_nose, right_eye, left_eye, mouth_in, mouth_out]

detector = dlib.get_frontal_face_detector()
way = "deep learning/misc/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(way)
"deep learning/misc/video.mp4"
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ok, frame = cap.read()
    if not ok: break

    # prep for processing
    frame = cv2.flip(frame, 1) # mirror effect
    frame = resize(frame, 1280)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # finding 68 points
    boxes = detector(gray, 0)
    for box in boxes:
        img = predictor(gray, box)
        img = face_utils.shape_to_np(img)

        # drawing lines
        for part in parts:
            draw(img, part)

        # facial landmarks
        landmarks(img)

        # blinking
        blink(right_eye, left_eye, True)


    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
