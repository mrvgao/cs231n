P = 0.5
Q = 0.75


def T(inPr, left, already, get_one, before_get_pr, step=0):
    if inPr <= 0:
        return 0

    if left == 0:
        return 1

    if get_one:
        pr = get_half_pr(P, already)
    else:
        pr = min(before_get_pr + Q, 1)

    return pr * (T(pr, left-1, already+1, get_one=True, before_get_pr=pr)) + (1 - pr) * (T(1-pr, left, already, get_one=False, before_get_pr=pr) + 1)

def get_success(left, already, before_get_pr):
    if left == 0:
        return 0

def get_half_pr(P, L):
    return ((100 * P)*1. / (2 ** L)) * .01

N = 2
print(T(1, N, 0, True, P))



