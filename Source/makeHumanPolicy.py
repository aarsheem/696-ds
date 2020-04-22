def getPolicy():
    (u, r, d, l, z) = (0, 1, 2, 3, 4)

    policy_ff = [
        r, r, d, r, r, r, d, d, l, d,
        r, u, d, r, r, l, r, d, u, l,
        r, r, d, u, r, r, d, d, d, u,
        d, r, d, u, r, r, r, d, d, d,
        d, r, r, u, r, r, r, r, z, d,
        d, r, r, r, r, r, u, u, u, d,
        d, r, u, r, r, d, r, d, d, d,
        d, u, u, r, r, r, u, r, l, d,
        d, r, r, u, r, d, u, d, r, r,
        r, r, u, u, r, r, u, r, r, d]

    policy_tf = [
        r, r, d, d, l, l, d, d, l, d,
        d, u, d, d, l, l, r, d, u, l,
        r, r, d, d, r, r, d, d, d, u,
        d, r, d, d, r, r, r, d, d, d,
        d, r, r, r, r, r, r, r, z, d,
        d, r, r, r, r, r, u, u, u, d,
        d, r, u, r, r, d, r, d, d, d,
        d, u, u, r, r, r, u, r, l, d,
        d, r, r, u, r, d, u, d, r, r,
        r, r, u, u, r, r, u, r, r, d]

    policy_tt = [
        r, r, d, d, l, l, d, d, l, d,
        d, u, d, d, l, l, r, d, u, l,
        r, r, d, d, r, r, d, d, d, u,
        d, r, d, d, r, r, r, d, d, d,
        d, r, r, r, r, r, r, r, r, d,
        d, r, r, r, r, r, u, u, u, d,
        d, r, u, r, r, d, r, d, d, d,
        d, u, u, r, r, r, u, r, l, d,
        d, r, r, u, r, d, u, d, r, d,
        r, r, u, u, r, r, u, r, r, d]

    stateSets = [policy_ff, policy_tf, policy_tt]
    boolIds = ["11", "21", "22"]
    count = 0
    policy = dict()
    for pol in stateSets:
        tag = boolIds[count]
        for i in range(len(pol)):
            stateId = str(i) + "," + tag
            policy[stateId] = pol[i]
        count = count + 1

    return policy