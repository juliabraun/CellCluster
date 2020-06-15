import numpy as np

y_control = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [0, 0, 0]])
y_pred = np.array([[0, 1, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

in_im = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]])
in_pred = np.array([[1, 0, 1],
                    [1, 0, 1],
                    [0, 1, 0]])


def dice_score(im, control_im):
    match = np.array(control_im == im)
    true = match.sum()
    dice = (2 * true) / (np.size(im) + np.size(control_im))
    return dice


def dicescore(im, control_im):
    intersection = np.logical_and(im, control_im)
    dice = (2 * intersection.sum()) / (im.sum() + control_im.sum())
    return dice


dice1 = dicescore(y_control, y_pred)
dice2 = dicescore(in_im, in_pred)

dice3 = (dice1 + dice2) / 2

print(dice3)
print(dice_score(y_control, y_pred))


def new_dice_score(im, control_im):
    """
    Computes Dice score to compare the similarity of two arrays or images and returns the result
    :param im: Binary or boolean array.
    :param control_im: Binary or boolean array of the same shape as im. Represents the 'ground truth'
    :return: Dice score as float value
    """
    # sum images, to find true positives
    summation = im + control_im
    tp = (summation == 2)
    # subtract images, to find false positives and false negatives.
    # Note that switching im and control_im also changes fp and fn, but does not change the resulting Dice score
    subtraction = im - control_im
    fp = (subtraction == 1)
    fn = (subtraction == -1)
    new_dice = (2 * tp.sum() / (2 * tp.sum() + fn.sum() + fp.sum()))
    return new_dice


dice4 = new_dice_score(y_pred, y_control)
print(dice4)
