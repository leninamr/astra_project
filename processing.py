import numpy as np


def processing(A, y):
    # b - Kx1 vector of observations
    b = y
# первое приближение с наименьшей затратой энергии, х0 имеет нечто общее с итоговым восстановленным сигналом
    x0 = (A.transpose()).dot(y)
    x = x0
    u = 0.95 * abs(x0) + 0.10 * max(abs(x0))

    pdtol = 1e-3
    pdmaxiter = 50
    cgtol = 1e-8
    cgmaxiter = 200

    N = len(x0)

    alpha = 0.01
    beta = 0.5
    mu = 10

    gradf0 = np.concatenate([np.zeros((N, 1)), np.ones((N, 1))])

    # set up for the first iteration
    fu1 = x - u
    fu2 = -x - u
    lamu1 = -1. / fu1
    lamu2 = -1. / fu2
    v = -A.dot(lamu1 - lamu2)
    Atv = A.transpose().dot(v)


    sdg = -(fu1.transpose().dot(lamu1) + fu2.transpose().dot(lamu2))
    tau = mu * 2. * N / sdg

    # Условия Каруша-Куна-Таккера:
    rcent = np.concatenate([-lamu1 * fu1, -lamu2 * fu2]) - (1 / tau)
    rdual = gradf0 + np.concatenate([lamu1 - lamu2, -lamu1 - lamu2]) + np.concatenate([Atv, np.zeros((N, 1))])
    rpri = A.dot(x) - b
    # считаем норму чтобы проверить близость условий к нулю
    resnorm = np.linalg.norm(np.concatenate([rdual, rcent, rpri]))

    pditer = 0
    done = (sdg < pdtol) | (pditer >= pdmaxiter)
    # Начинаются вычисления методом Ньютона
    while not done:

        pditer = pditer + 1

        w1 = (-1. / tau) * (-1. / fu1 + 1. / fu2) - Atv
        w2 = -1 - (1 / tau) * (1. / fu1 + 1. / fu2)
        w3 = -rpri

        sig1 = np.divide(-lamu1, fu1) - np.divide(lamu2, fu2)
        sig2 = np.divide(lamu1, fu1) - np.divide(lamu2, fu2)
        sigx = sig1 - np.divide(sig2 * sig2, sig1)

        w1p = -(w3 - A.dot(np.divide(w1, sigx) - np.divide(w2 * sig2, (sigx * sig1))))
        temp = np.diag(1. / sigx.transpose()[0])
        H11p = A.dot(temp.dot(A.transpose()))
        dv = np.linalg.solve(H11p, w1p)
        hcond = np.linalg.cond(H11p)
        if hcond < 1e-14:
            print(
                "Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)")
            xp = x
            return xp

        dx = np.divide(w1 - np.divide(w2 * sig2, sig1) - (A.transpose()).dot(dv), sigx)
        Adx = A.dot(dx)
        Atdv = (A.transpose()).dot(dv)

        du = np.divide((w2 - sig2 * dx), sig1)

        dlamu1 = np.divide(lamu1, fu1) * (-dx + du) - lamu1 - ((1 / tau) * (1. / fu1))
        dlamu2 = np.divide(lamu2, fu2) * (dx + du) - lamu2 - 1 / tau * 1. / fu2

        # необходимо чтобы шаг был выполнимый: сохраняет lamu1, lamu2 > 0, fu1, fu2 < 0
        indp = np.where(dlamu1 < 0)
        indn = np.where(dlamu2 < 0)
        s = min(np.concatenate([[1], np.divide(-lamu1[indp], dlamu1[indp]), np.divide(-lamu2[indn], dlamu2[indn])]))
        indp = np.where((dx - du) > 0)
        indn = np.where((-dx - du) > 0)
        s = 0.99 * min(np.concatenate([[s], np.divide(-fu1[indp], (dx[indp] - du[indp])), np.divide(-fu2[indn], (-dx[indn] - du[indn]))]))

        # поиск оптимального s
        suffdec = 0
        backiter = 0
        while not suffdec:
            xp = x + s * dx
            up = u + s * du
            vp = v + s * dv
            Atvp = Atv + s * Atdv
            lamu1p = lamu1 + s * dlamu1
            lamu2p = lamu2 + s * dlamu2
            fu1p = xp - up
            fu2p = -xp - up

            rdp = gradf0 + np.concatenate([lamu1p - lamu2p, -lamu1p - lamu2p]) + np.concatenate([Atvp, np.zeros((N, 1))])
            rcp = np.concatenate([-lamu1p * fu1p, -lamu2p * fu2p]) - (1 / tau)
            rpp = rpri + s * Adx
            suffdec = (np.linalg.norm(np.concatenate([rdp, rcp, rpp])) <= (1 - alpha * s) * resnorm)
            s = beta * s
            backiter = backiter + 1
            if backiter > 32:
                print("Stuck backtracking, returning last iterate.  (See Section 4 of notes for more information.)")
                xp = x
                return xp

        # следующий шаг
        x = xp
        u = up
        v = vp
        Atv = Atvp
        lamu1 = lamu1p
        lamu2 = lamu2p
        fu1 = fu1p
        fu2 = fu2p

        # surrogate duality gap
        sdg = -(fu1.transpose().dot(lamu1) + fu2.transpose().dot(lamu2))
        tau = mu * 2 * N / sdg
        rpri = rpp
        rcent = np.concatenate([-lamu1 * fu1, -lamu2 * fu2]) - (1 / tau)
        rdual = gradf0 + np.concatenate([lamu1 - lamu2, -lamu1 - lamu2]) + np.concatenate([Atv, np.zeros((N, 1))])
        resnorm = np.linalg.norm(np.concatenate([rdual, rcent, rpri]))

        done = (sdg < pdtol) | (pditer >= pdmaxiter)

        print("Iteration = ", pditer, "tau = ", tau, " Primal = ", sum(u), " PDGap = ", sdg, " Dual res = ",
              np.linalg.norm(rdual), ", Primal res = ", np.linalg.norm(rpri))
        print('H11p condition number = ', hcond)
    return xp
