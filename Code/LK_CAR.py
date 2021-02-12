import cv2
import glob
import numpy as np
import copy

Data = "Car4/img"
ref = "Car4/img/0020.jpg"
ref_ = "Car4/img/0226.jpg"
ref_1 = "Car4/img/0277.jpg"
ref_2 = "Car4/img/0355.jpg"
ref_3 = "Car4/img/0635.jpg"
ref_4 = "Car4/img/0590.jpg"
input_Image = cv2.imread(ref)
input_Image_ = cv2.imread(ref_)
input_Image_1 = cv2.imread(ref_1)
input_Image_2 = cv2.imread(ref_2)
input_Image_3 = cv2.imread(ref_3)
input_Image_4 = cv2.imread(ref_4)
images = cv2.cvtColor(input_Image, cv2.COLOR_BGR2GRAY)
images_ = cv2.cvtColor(input_Image_, cv2.COLOR_BGR2GRAY)
images_1 = cv2.cvtColor(input_Image_1, cv2.COLOR_BGR2GRAY)
images_2 = cv2.cvtColor(input_Image_2, cv2.COLOR_BGR2GRAY)
images_3 = cv2.cvtColor(input_Image_3, cv2.COLOR_BGR2GRAY)
images_4 = cv2.cvtColor(input_Image_4, cv2.COLOR_BGR2GRAY)
# CAR
Rec = np.array([[70, 55], [170, 55], [170, 115], [70, 115]])
Rec_ = np.array([[150, 70], [223, 70], [223, 118], [150, 118]])
Rec_1 = np.array([[188, 67], [255, 67], [255, 110], [188, 110]])
Rec_2 = np.array([[219, 77], [282, 77], [282, 115], [219, 115]])
Rec_3 = np.array([[255, 60], [345, 60], [345, 120], [255, 120]])
Rec_4 = np.array([[215, 60], [285, 60], [285, 120], [215, 120]])
p = np.zeros(6)
template = images[Rec[0, 1]:Rec[2, 1], Rec[0, 0]:Rec[2, 0]]
template_ = images_[Rec_[0, 1]:Rec_[2, 1], Rec_[0, 0]:Rec_[2, 0]]
template_1 = images_1[Rec_1[0, 1]:Rec_1[2, 1], Rec_1[0, 0]:Rec_1[2, 0]]
template_2 = images_2[Rec_2[0, 1]:Rec_2[2, 1], Rec_2[0, 0]:Rec_2[2, 0]]
template_3 = images_3[Rec_3[0, 1]:Rec_3[2, 1], Rec_3[0, 0]:Rec_3[2, 0]]
template_4 = images_4[Rec_4[0, 1]:Rec_4[2, 1], Rec_4[0, 0]:Rec_4[2, 0]]
RecNew = copy.deepcopy(Rec)

out = cv2.VideoWriter('CarVideo.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
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

def LK_Tracker(inputImage,template, Rec, p):
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
    print("Frame-->",i)
    i += 1

    if i >= 614:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_3, Rec_3, p)

    elif i >= 550 and i <= 613:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_4, Rec_4, p)

    elif i >= 330 and i <= 447:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_2, Rec_2, p)

    elif (i >= 256 and i <= 329) or (i >= 448 and i <= 549):
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_1, Rec_1, p)

    elif i >= 169 and i <= 255:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template_, Rec_, p)

    elif i > 659:
        break

    else:
        inputImage = cv2.imread(img_)
        LK_Tracker(inputImage, template, Rec, p)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.waitKey(10)
cv2.destroyAllWindows()

