import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


N = 200
M = 200

a = np.random.random((N, 2)) * 0.9 + 0.05
b = np.random.random((M, 2)) * 0.9 + 0.05

c = np.random.random(a.shape)




def sinkhorn(probs : np.ndarray, n_iterations = 2):
    probs = probs.copy()
    for _ in range(n_iterations):
        probs /= probs.sum(0, keepdims=True)
        probs /= probs.sum(1, keepdims=True)
    return probs

def asignement(probs, *args, **kwargs):
    row_ind, col_ind = linear_sum_assignment(probs)
    probs = np.zeros_like(probs)
    probs[row_ind, col_ind] = 1
    return probs


n_iters = 100
for i in range(10000):

    dist = ((a[:, None, :] - b[None, :, :])**2).sum(-1)**0.5

    P = sinkhorn(np.exp(-dist*5000), n_iters)
    #P = asignement(dist)

    a_to = np.einsum("nm,md->nd", P, b)


    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image.fill(255)

    for p in a:
        p = p * 1000
        image = cv2.circle(image, p.astype(int), 8, (0, 0, 0), -1)

    for p, p_to in zip(a, a_to):
        p = p * 1000
        p_to = p_to * 1000
        image = cv2.line(image, p.astype(int), p_to.astype(int), (0, 0, 0), 2)

    for p in b:
        p = p * 1000
        image = cv2.circle(image, p.astype(int), 6, (0, 256, 0), -1)

    a = a * 0.95 + a_to * 0.05
    #dif = c-a
    #a += 0.005 * dif / np.linalg.norm(dif, 2)
    #n_iters += 1
    cv2.imshow("show", image)
    key = cv2.waitKey(-1)
    if key == 27 or key == ord("q"):
        break
cv2.destroyAllWindows()
