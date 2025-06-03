import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq, irfft
from scipy.signal import savgol_filter


def choose_chunk_peak(all_flat_signals, all_points, window_size=5000, wave_len=15):
    num_window = all_flat_signals[0].shape[0] // window_size
    waves_all = np.zeros([len(all_flat_signals), num_window, 2 * wave_len])

    start_time = time.time()
    for index in range(len(all_flat_signals)):
        if np.mod(index, 100) == 0:
            print(index)
            print("Elapsed time: {}".format(time.time() - start_time))

        flat = all_flat_signals[index]
        points = all_points[index]
        if len(points) > 0:
            for i in range(num_window):
                flat_interval = flat[(i * window_size) : (i + 1) * window_size]
                loc = (points[:, 0] >= i * window_size) & (
                    points[:, 0] <= (i + 1) * window_size
                )
                points_interval = points[loc]
                # points_interval = points[(points[:,0] >= i*window_size) & (points[:,0] <= (i+1)*window_size)]
                if len(points_interval) > 0:
                    point_keep = points_interval[
                        np.argmax(np.abs(points_interval[:, 1]))
                    ]
                    start = int(point_keep[0] - 15)
                    end = int(point_keep[0] + 15)
                    f = flat[start:end]
                    waves_all[index, i, :] = f

    return waves_all


def spike_detection_ori5_fast(x, size=250, noise_level=3):
    length = x.shape[0]
    x_abs = abs(x)

    NUM_PART = 20
    LEN_PART = int(length / NUM_PART)
    large_points = []
    for i in range(NUM_PART):
        large_points.append(
            np.argpartition(x_abs[i * LEN_PART : (i + 1) * LEN_PART], 40000 - 100)[
                -100:
            ]
            + i * LEN_PART
        )
    large_points = np.concatenate(large_points)

    x_clear = abs(x_abs)
    for point in large_points:
        if point - 50 > 0 and point + 50 < 800000:
            x_clear[point - 50 : point + 50] = 0
    large_points_2 = []
    for i in range(NUM_PART):
        # large_points_2.append(np.argsort(x_clear[i*LEN_PART: (i+1)*LEN_PART])[-50:] + i*LEN_PART)
        # large_points_2.append(bottleneck.argpartition(-x_clear[i*LEN_PART: (i+1)*LEN_PART], 100)[:100] + i*LEN_PART)
        large_points_2.append(
            np.argpartition(x_clear[i * LEN_PART : (i + 1) * LEN_PART], 40000 - 100)[
                -100:
            ]
            + i * LEN_PART
        )
    large_points_2 = np.concatenate(large_points_2)
    large_points = np.concatenate([large_points, large_points_2])

    x_clear_2 = abs(x_clear)
    for point in large_points_2:
        if point - 50 > 0 and point + 50 < 800000:
            x_clear_2[point - 50 : point + 50] = 0
    large_points_3 = []
    for i in range(NUM_PART):
        # large_points_3.append(np.argsort(x_clear_2[i*LEN_PART: (i+1)*LEN_PART])[-50:] + i*LEN_PART)
        # large_points_3.append(bottleneck.argpartition(-x_clear_2[i*LEN_PART: (i+1)*LEN_PART], 100)[:100] + i*LEN_PART)
        large_points_3.append(
            np.argpartition(x_clear_2[i * LEN_PART : (i + 1) * LEN_PART], 40000 - 100)[
                -100:
            ]
            + i * LEN_PART
        )
    large_points_3 = np.concatenate(large_points_3)
    large_points = np.concatenate([large_points, large_points_3])

    # large_points = np.argsort(x)[-2000:]
    points = []
    for point in large_points:
        if point - 25 > 0 and point + 25 < 800000:
            window = x_abs[max(0, point - 25) : min(length, point + 25)]
            if x_abs[point] == np.max(window):
                if (
                    x_abs[point] > 4
                    and x_abs[point] < 50
                    and x_abs[point] > noise_level * 1.155
                ):

                    FLAG = True
                    while FLAG == True:
                        if (
                            np.sign(x[point - 1]) * np.sign(x[point]) == -1
                            and x_abs[point - 1] > 0.5 * x_abs[point]
                        ):
                            point = point - 1
                        else:
                            FLAG = False

                    if (
                        np.sign(x[point - 2]) * np.sign(x[point]) == -1
                        and x_abs[point - 2] > 0.5 * x_abs[point]
                    ):
                        points.append([point - 2, x[point]])

                    elif (
                        np.sign(x[point - 3]) * np.sign(x[point]) == -1
                        and x_abs[point - 3] > 0.5 * x_abs[point]
                    ):
                        points.append([point - 3, x[point]])

                    else:
                        points.append([point, x[point]])
    return points


def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2 / s.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)


def get_crossing(x, phase=0):
    x_1 = low_pass(x, threshold=100)
    x = x_1.reshape((-1,))
    zero_crossing = np.where(np.diff(np.sign(x)))[0]
    up_crossing = -1
    for zc in zero_crossing:
        if x[zc] < 0 and x[zc + 1] > 0:
            up_crossing = zc
    return up_crossing


def phase_shift(x, cross):
    if cross > 0:
        x = np.hstack([x[cross:], x[:cross]])
    return x


def noise_estimation_fixed(x):
    NUM_PATCH = 1000
    LEN_PATCH = 1000
    # indexes = np.random.uniform(low=0, high=799000, size=NUM_PATCH)
    indexes = np.linspace(0, 799000, NUM_PATCH)

    patches = np.zeros((NUM_PATCH, LEN_PATCH))
    for i in range(NUM_PATCH):
        patches[i, :] = x[int(indexes[i]) : int(indexes[i]) + LEN_PATCH]
    coverage = 0
    diff = []
    over_th = 0
    for index, i in enumerate(np.linspace(0, 15, 31)):
        num_cover = np.sum(i > np.max(patches, axis=-1))
        diff.append(num_cover - coverage)

        if num_cover - coverage > 80:
            over_th = index

        coverage = num_cover

    loc_max_diff = np.argmax(diff)
    # print(diff)
    # plt.plot(diff)
    return max(loc_max_diff, over_th) * 0.5 + 0.5


def peaks_on_flatten(train_df, signal_ids, visualization=False):
    start_time = time.time()
    all_aligned_signals = []
    all_flat_signals = []
    all_points = []

    for index in signal_ids:
        if np.mod(index, 100) == 0:
            print(index)
            print("Elapsed time: {}".format(time.time() - start_time))

        signal = train_df[str(index)].values

        crossing = get_crossing(signal)
        signal = phase_shift(signal, crossing)
        yhat = savgol_filter(signal, 99, 3)
        flat = signal - yhat
        noise_level = noise_estimation_fixed(flat)
        points = spike_detection_ori5_fast(flat, noise_level=noise_level)
        points = np.array(points)

        all_aligned_signals.append(signal)
        all_flat_signals.append(flat)
        all_points.append(points)

    if visualization:
        plt.plot(signal)
        plt.plot(yhat)
        plt.plot(flat)
        plt.axhline(noise_level, color="y")
        plt.axhline(-noise_level, color="y")
        plt.scatter(points[:, 0], points[:, 1], color="red")
        plt.legend(["aligned", "high pass", "flatten", "noise"])
        plt.xlim([0, 800000])
        plt.show()

    return all_aligned_signals, all_flat_signals, all_points
