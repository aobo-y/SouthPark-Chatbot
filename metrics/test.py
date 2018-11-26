from bleu import *

if __name__ == "__main__":
    # test decide_which_bleu function

    h1=['hello','world']
    r1=['hello','world']
    h2=['this', 'is', 'small', 'test']
    r2=['this', 'is', 'a', 'test']
    print(decide_which_bleu(h1, 1, 'individual'))
    print(decide_which_bleu(h1, 1, ))
    print(decide_which_bleu(h1, 2, 'individual'))
    print(decide_which_bleu(h1, 2, ))
    print(decide_which_bleu(h1, 3, 'individual'))
    print(decide_which_bleu(h1, 3, ))
    print(decide_which_bleu(h1, 5, 'individual'))
    print(decide_which_bleu(h1, 5, ))
    print(decide_which_bleu([], 3, 'individual'))
    print(decide_which_bleu([], 3, ))

    # test cal_bleu function
    print(cal_bleu(h1,[r1], 1, 'individual'))
    print(cal_bleu(h1,[r1], 1, ))
    print(cal_bleu(h1,[r1], 2, 'individual'))
    print(cal_bleu(h1,[r1], 2, ))
    print(cal_bleu(h2,[r2], 1, 'individual'))
    print(cal_bleu(h2,[r2], 1, ))
    print(cal_bleu(h2,[r2], 2, 'individual'))
    print(cal_bleu(h2,[r2], 2, ))
