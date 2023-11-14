import torch
from model import CuriosityRewardTransformer


def test(curiosity, state1, state2, n_steps=100):
    #state1 = state1[:5]
    #state2 = state2[:5]
    mask1 = state1 != 0
    # mask2 = state2 != 0
    desc = []
    asc = []
    for _ in range(n_steps):
        cur1 = curiosity(state1)            # feeding s1
        meancur = (cur1 * mask1).mean()     # and measuring curiosity on s1
        desc.append(meancur.item())
    for _ in range(n_steps):
        curiosity(state2)                               # feeding s2
        cur1 = curiosity(state1, optim_step=False)      # but measuring curiosity on s1
        meancur = (cur1 * mask1).mean()
        asc.append(meancur.item())
    return desc, asc


if __name__ == "__main__":
    st1 = torch.tensor([[ 1,  7, 20, 20, 25, 17, 10, 24, 24,  2,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 24, 19,  6, 23, 10,  9,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  8, 14, 23,  8, 26, 18, 11, 20, 23,  6, 19, 10, 20, 26, 24,  2,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 27, 10, 23,  7, 20, 24, 14, 25, 30,  2,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 13, 30, 10, 18,  6, 25, 14, 20, 19,  2,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  7, 26, 25, 25, 30,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 25,  6, 23, 24, 20, 25, 20, 18, 30,  2,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 21, 26, 25, 25, 30, 23, 20, 20, 25,  2,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  7, 26, 17,  7, 26, 17,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 19, 20, 25, 14, 20, 19,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0]])

    st2 = torch.tensor([[ 1, 26, 24, 26, 11, 23, 26,  8, 25, 26,  6, 23, 30,  2,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  9, 30, 19,  6, 18, 14,  8, 24,  2,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 25, 13, 14, 25, 13, 10, 23, 28,  6, 23,  9,  2,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 24,  6, 19, 16, 13, 30,  6,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  7,  6, 19, 24, 13, 10, 10,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 24, 17, 10, 10, 25, 10,  9,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 24, 17, 14,  8, 10, 23,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  8, 14, 19, 25, 10, 23,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  9, 14, 17, 26, 10, 19, 25,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 10, 18, 21, 17,  6, 24, 25, 23,  6, 25, 14, 20, 19,  2,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0]])

    st1 = st1.repeat(100, 1)
    st2 = st2.repeat(100, 1)
    n_tokens = 32
    curiosity = CuriosityRewardTransformer(n_tokens, scale=1000., lr=1e-2, temperature=(4., 1., 1.), lr_t=1e-3)
    curiosity.calibrate_scale(st1, 1.)
    desc, asc = test(curiosity, st1, st2, 100)
    print("Boredom tendency. Expecting descending sequence of curiosity rewards: ")
    print(f"{desc=}")
    print("Forgetting tendency. Expecting ascending sequence of curiosity rewards as the module gradually forgets the s1 state")
    print(f"{asc=}")
    # TODO: curiosity needs moar tests!!11
    print("woah!")
