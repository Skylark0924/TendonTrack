import io
import random
import math


def DrunkMan_MonteCarlo(filename, count):
    x_pos, y_pos = 0, 0
    d_pos = 0.05
    with io.open(filename, 'w') as f:
        for i in range(count):
            x_action = 2 * d_pos * (random.random() - 0.5)
            y_action = 2 * d_pos * (random.random() - 0.5)
            x_pos = constrain(x_pos + x_action, -5, 5)
            y_pos = constrain(y_pos + y_action, -5, 5)
            meta_txt = '{} {} {} {}\r\n'.format(x_pos, y_pos, x_action, y_action)
            f.write(meta_txt)
            if i % 10 == 0:
                print(i)


def constrain(x, lower_bound, upper_bound):
    x = (upper_bound + lower_bound) / 2 if math.isnan(x) else x
    x = upper_bound if x > upper_bound else x
    x = lower_bound if x < lower_bound else x
    return x  # int(round(x))


if __name__ == '__main__':
    DrunkMan_MonteCarlo('drunkman.txt', 1000)
