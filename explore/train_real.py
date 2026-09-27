# -*- coding: utf-8 -*-
"""Train real classification with group-theoretic continuous mapping."""
import random, math
import minimal_principle as M

SZ = 14


def digit(d, noise=0.0):
    """draw digit d as 14x14 dot matrix."""
    img = [[0.0]*SZ for _ in range(SZ)]
    if d == 0:
        pts = [(1+i, 4+j) for i in range(7) for j in range(6)
               if (i == 0 or i == 6 or j == 0 or j == 5)]
    elif d == 1:
        pts = [(1+i, 4+j) for i in range(7) for j in range(6) if i % 2 == 0]
    else:
        pts = [(1+i, 4+j) for i in range(7) for j in range(6) if i == j]
    for i, j in pts:
        img[i][j] = 200.0
    if noise:
        for i in range(SZ):
            for j in range(SZ):
                img[i][j] += random.gauss(0, noise)
    return img


def levels(img):
    """L0, L1=down2, L2=down4 active; plus passive channel."""
    la = [M.flat(img)]
    cur = img
    for f in (2, 4):
        cur = M.down(cur, f)
        la.append(M.flat(cur))
    lp = [M.flat(M.passive(img))]
    cur = img
    for f in (2, 4):
        cur = M.down(cur, f)
        lp.append(M.flat(M.passive(cur)))
    return la, lp


def main():
    random.seed(42)
    train = {}
    for d in range(3):
        train[d] = levels(digit(d))
    ok = 0
    n = 30
    for d in range(3):
        for _ in range(n):
            ql, qp = levels(digit(d, noise=10))
            best, bl = -1, 0.0
            for k in range(3):
                a = max(M.mod(ql[i], train[k][0][i]) for i in range(3))
                p = max(M.mod(qp[i], train[k][1][i]) for i in range(3))
                s = max(a, p)
                if s > bl: bl, best = s, k
            if best == d: ok += 1
    print("accuracy: %.1f%%" % (100.0*ok/(3*n)))


if __name__ == "__main__":
    main()
