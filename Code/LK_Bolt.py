import cv2
import glob
import numpy as np
import copy

Data = "Bolt2/img"
ref = "Bolt2/img/0001.jpg"
ref_ = "Bolt2/img/0020.jpg"
ref_1 = "Bolt2/img/0079.jpg"
ref_2 = "Bolt2/img/0111.jpg"
ref_3 = "Bolt2/img/0128.jpg"
ref_4 = "Bolt2/img/0210.jpg"
ref_5 = "Bolt2/img/0230.jpg"
input_Image = cv2.imread(ref)
input_Image_ = cv2.imread(ref_)
input_Image_1 = cv2.imread(ref_1)
input_Image_2 = cv2.imread(ref_2)
input_Image_3 = cv2.imread(ref_3)
input_Image_4 = cv2.imread(ref_4)
input_Image_5 = cv2.imread(ref_5)
images = cv2.cvtColor(input_Image, cv2.COLOR_BGR2GRAY)
images_ = cv2.cvtColor(input_Image_, cv2.COLOR_BGR2GRAY)
images_1 = cv2.cvtColor(input_Image_1, cv2.COLOR_BGR2GRAY)
images_2 = cv2.cvtColor(input_Image_2, cv2.COLOR_BGR2GRAY)
images_3 = cv2.cvtColor(input_Image_3, cv2.COLOR_BGR2GRAY)
images_4 = cv2.cvtColor(input_Image_4, cv2.COLOR_BGR2GRAY)
images_5 = cv2.cvtColor(input_Image_5, cv2.COLOR_BGR2GRAY)
#Bolt
Rec = np.array([[285,71],[308,81],[305,143],[264,143]])
Rec_ = np.array([[240,66],[260,68],[270,140],[250,140]])
Rec_1 = np.array([ [242, 78],[274, 70], [274,180],[230, 175]])
Rec_2 = np.array([[200, 50], [243, 50], [243, 160], [200, 160]])
Rec_3 = np.array([[270, 65], [320, 65], [320, 165], [270, 165]])
Rec_4 = np.array([[327, 88], [367, 88], [367, 175], [327, 175]])
Rec_5 = np.array([[345, 95], [387, 95], [387, 165], [345,165]])
p = np.zeros(6)
template = images[Rec[0, 1]:Rec[2, 1], Rec[0, 0]:Rec[2, 0]]
template_ = images_[Rec_[0, 1]:Rec_[2, 1], Rec_[0, 0]:Rec_[2, 0]]
template_1 = images_1[Rec_1[0, 1]:Rec_1[2, 1], Rec_1[0, 0]:Rec_1[2, 0]]
template_2 = images_2[Rec_2[0, 1]:Rec_2[2, 1], Rec_2[0, 0]:Rec_2[2, 0]]
template_3 = images_3[Rec_3[0, 1]:Rec_3[2, 1], Rec_3[0, 0]:Rec_3[2, 0]]
template_4 = images_4[Rec_4[0, 1]:Rec_4[2, 1], Rec_4[0, 0]:Rec_4[2, 0]]
template_5 = images_5[Rec_5[0, 1]:Rec_5[2, 1], Rec_5[0, 0]:Rec_5[2, 0]]
RecNew = copy.deepcopy(Rec)

out = cv2.VideoWriter('BoltVideo.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                      (images.shape[1], images.shape[0]))

def Check_illumination(image, gamma=1.0):
    inv_G = 1.0 / gamma
    T = np.array([((i / 255.0) ** inv_G) * 255
                  for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, T)

def Lk_Algorithm(img, tmp, Rec, p):
    W = Matrix_Affine(p)
    Norm_P = 1
    Thr_ = 0.005
    count = 0
    img = cv2.GaussianBlur(img, (3, 3), 0)
    tmp = cv2.GaussianBlur(tmp, (3, 3), 0)

    while Norm_P > Thr_:
        count += 1
        Iw = cv2.warpAffine(img, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        Iw = Iw[Rec[0, 1]:Rec[2, 1], Rec[0, 0]:Rec[2, 0]]
        if np.linalg.norm(Iw) < np.linalg.norm(tmp):
            img = Check_illumination(img, gamma=1.5)
        Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize=5)
        Ix = cv2.warpAffine(Ix, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        Iy = cv2.warpAffine(Iy, W, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        Ix = Ix[Rec[0, 1]:Rec[2, 1], Rec[0, 0]:Rec[2, 0]]
        Iy = Iy[Rec[0, 1]:Rec[2, 1], Rec[0, 0]:Rec[2, 0]]
        error = tmp.flatten().astype(np.int) - Iw.flatten().astype(np.int)
        X_G = np.asarray(list(range(Ix.shape[1])))
        Y_G = np.asarray(list(range(Ix.shape[0])))
        X_G, Y_G = np.meshgrid(X_G, Y_G)
        Steep_Img = np.array([ \
            np.multiply(Ix.flatten(), X_G.flatten()),
            np.multiply(Iy.flatten(), X_G.flatten()),
            np.multiply(Ix.flatten(), Y_G.flatten()),
            np.multiply(Iy.flatten(), Y_G.flatten()),
            Ix.flatten(),
            Iy.flatten() \
            ]).T
        H = np.dot(Steep_Img.T, Steep_Img)
        dp = np.dot(np.linalg.pinv(H), np.dot(Steep_Img.T, error))
        Norm_P = np.linalg.norm(dp)
        p = p + (dp * 10)
        W = Matrix_Affine(p)
        if (count > 1000):
            break
    return p

def Matrix_Affine(p):
    W = np.hstack([np.eye(2), np.zeros((2, 1))]) + p.reshape((2, 3), order='F')
    return W

def LK_Tracker(inputImage, template, Rec, p):
    grey = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    p = Lk_Algorithm(grey, template, Rec, p)
    w = Matrix_Affine(p)
    RecDraw_ = np.dot(w, np.vstack((Rec.T, np.ones((1, 4))))).T
    RecTemp = RecDraw_.astype(np.int32)
    [xmax, ymax] = list(np.max(RecTemp, axis=0).astype(np.int))
    [xmin, ymin] = list(np.min(RecTemp, axis=0).astype(np.int))
    RecNew = np.array([[xmin, ymin],
                       [xmax, ymin],
                       [xmax, ymax],
                       [xmin, ymax]])
    output = cv2.polylines(inputImage, [RecNew], True, (0, 255, 255), thickness=5)
    cv2.imshow("Output Video", output)
    out.write(output)
    cv2.waitKey(10)

i = 1
for img_ in glob.glob(f"{Data}/*.jpg"):
    print("Frame-->", i)
    i += 1

    if i >= 20 and i <= 78:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_, Rec_, p)

    elif (i >= 79 and i <= 110):
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_1, Rec_1, p)

    elif i >= 111 and i <= 127:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_2, Rec_2, p)

    elif i >= 128 and i <= 180:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_3, Rec_3, p)

    elif i >= 181 and i < 230:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_4, Rec_4, p)

    elif i >= 230:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_5, Rec_5, p)

    else:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template, Rec, p)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
out.release()
cv2.waitKey(10)
cv2.destroyAllWindows()

