import numpy as np
import cv2

L = 256


def Negative(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imgout[x, y] = s
    return imgout


def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    c = (L - 1) / np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r == 0:
                r = 1
            s = c * np.log(1 + r)
            imgout[x, y] = np.uint8(s)
    return imgout


# def LogaritColor(imgin):
#     B = imgin[:, :, 0]
#     G = imgin[:, :, 1]
#     R = imgin[:, :, 1]

#     B = Logarit(B)
#     G = Logarit(G)
#     R = Logarit(R)

#     imgout = np.array([B, G, R])
#     imgout = np.transpose(imgout, axes=[1, 2, 0])

#     return imgout


def Power(imgin):
    M, N = imgin.shape
    gamma = 5.0
    imgout = np.zeros((M, N), np.uint8)
    c = np.power(L - 1, 1 - gamma)

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = c * np.power(r, gamma)
            imgout[x, y] = np.uint8(s)
    return imgout


def PiecewiseLinear(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    rmin = np.min(imgin)
    rmax = np.max(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r < rmin:
                s = s1 / r1 * r
            elif r < r2:
                s = (s2 - s1) / (r2 - r1) * (r - r1) + s1
            else:
                s = (L - 1 - s2) / (L - 1 - r2) * (r - r2) + s2
            imgout[x, y] = np.uint8(s)
    return imgout


def Histogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    h = np.zeros(L, int)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            h[r] = h[r] + 1
    p = h / (M * N)
    scale = 2000
    for r in range(0, L):
        cv2.line(imgout, (r, M - 1), (r, M - 1 - int(scale * p[r])), (0, 0, 0))
    return imgout


def HistEqual(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    h = np.zeros(L, int)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            h[r] = h[r] + 1
    p = h / (M * N)
    s = np.zeros(L, np.float64)
    for k in range(0, L):
        for j in range(0, k + 1):
            s[k] = s[k] + p[j]
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            imgout[x, y] = np.uint8((L - 1) * s[r])
    return imgout


def HistEqualColor(imgin):
    M, N, C = imgin.shape
    imgout = np.zeros((M, N, C), np.uint8)

    R = imgin[:, :, 2]
    G = imgin[:, :, 1]
    B = imgin[:, :, 0]

    R = cv2.equalizeHist(R)
    G = cv2.equalizeHist(G)
    B = cv2.equalizeHist(B)

    imgout[:, :, 2] = R
    imgout[:, :, 1] = G
    imgout[:, :, 0] = B

    return imgout


def LocalHist(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 3
    n = 3
    w = np.zeros((m, n), np.uint8)
    a = m // 2
    b = n // 3
    for x in range(a, M - a):
        for y in range(b, N - b):
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    w[s + a, t + b] = imgin[x + s, y + t]
            w = cv2.equalizeHist(w)
            imgout[x, y] = w[a, b]
    return imgout


def HistStat(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 3
    n = 3
    w = np.zeros((m, n), np.uint8)
    a = m // 2
    b = n // 3

    mG, sigmaG = cv2.meanStdDev(imgin)
    k0, k1, k2, k3 = 0.0, 0.1, 0.0, 0.1
    C = 22.8

    for x in range(a, M - a):
        for y in range(b, N - b):
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    w[s + a, t + b] = imgin[x + s, y + t]
            msxy, sigmasy = cv2.meanStdDev(w)
            r = imgin[x, y]
            if (k0 * mG <= msxy <= k1 * mG) and (k2 * sigmaG <= sigmasy <= k3 * sigmaG):
                imgout[x, y] = np.uint8(C * r)
            else:
                imgout[x, y] = r
    return imgout


def BoxFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 3
    n = 3
    w = np.ones((m, n), np.float64)
    w = w / (m * n)
    a = m // 2
    b = n // 2

    for x in range(0, M):
        for y in range(0, N):
            r = 0.0
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    r = r + w[s + a, t + a] * imgin[((x + s) % M, (y + t) % N)]
            imgout[x, y] = np.uint8(r)
    return imgout
# def BoxFilter(imgin):
#     m = 21
#     n = 21
#     w = np.ones((m, n), np.float64)
#     w = w / (m * n)
#     imgout = cv2.filter2D(imgin, cv2.CV_)
def Smooth(imgin):
    temp = cv2.blur(imgin,(15,15))
    retval, imgout = cv2.threshold(temp, 64, L-1,cv2.THRESH_BINARY)
    return imgout
def MedianFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 7
    n = 7
    w = np.ones((m, n), np.float64)
    w = w / (m * n)
    a = m // 2
    b = n // 2

    for x in range(0, M):
        for y in range(0, N):
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                   w[s + a, t + a] = imgin[(x + s) % M, (y + t) % N]
            w_1D = np.reshape(w,(m*n,))
            w_1D = np.sort(w_1D)
            imgout[x,y]= w_1D[m*n//2]
    return imgout
def Sharpen(imgin):
    # Đạo hàm cấp 2 của ảnh
    w = np.array([[0,1,0],[1,-4,1],[0,1,0]],np.float32)
    temp = cv2.filter2D(imgin,cv2.CV_32FC1,w)
    # Làm tăng độ nét của ảnh
    imgout = imgin - temp
    imgout = np.clip(imgout,0,L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

