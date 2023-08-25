import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results(SAC=None, IPK=None, MBPO=None, Basic=None, PILCO=None):
    ax = plt.figure(figsize=(12, 10))
    sns.set(style="darkgrid", font_scale=2)
    if SAC:
        sac_data = pd.read_csv('D:\\桌面\\progress_SAC_7434.csv')
        if use_seaborn:
            sac_rew_list = sac_data['training/path_return'].values
            sac_float_2d = rew_list_split(sac_rew_list)
            sac_float_2d = smooth_batch3(sac_float_2d) * 0.4 - 150
            sac_epoch_expand = epoch_expand(sac_float_2d)

            sac_epoch_expand_int = sac_epoch_expand.astype(np.int)
            sac_float_smooth = sac_float_2d.flatten()
            assert sac_epoch_expand_int.shape == sac_float_smooth.shape

            sac_data_done = pd.DataFrame({
                'epoch': sac_epoch_expand_int,
                'return': sac_float_smooth,
                # 'category': 'SAC'
            })
            sns.lineplot(x='epoch', y='return', markers=False, dashes=True, data=sac_data_done)
        else:
            sac_rew_avg = sac_data['training/return-average'].values
            sac_rew_min = sac_data['training/return-min'].values
            sac_rew_max = sac_data['training/return-max'].values

            dim_num = len(sac_data)
            x = np.linspace(1, dim_num, dim_num)
            # fig, ax = plt.subplots(figsize=(10, 5))

            plt.plot(x, smooth(sac_rew_avg), c='r')
            plt.fill_between(x, smooth(sac_rew_max), smooth(sac_rew_min), alpha=0.1, color='orange')

    if PILCO:
        pilco_data = pd.read_csv('D:\\桌面\\progress_MBPO_8303.csv')
        if use_seaborn:
            pilco_rew_list = pilco_data['training/path_return']
            pilco_float_2d = rew_list_split(pilco_rew_list)
            pilco_float_2d = smooth_batch3(pilco_float_2d) * 0.4 - 100
            pilco_epoch_expand = epoch_expand(pilco_float_2d)

            pilco_epoch_expand_int = pilco_epoch_expand.astype(np.int)
            pilco_float_smooth = flatten_manually(pilco_float_2d)
            assert pilco_epoch_expand_int.shape == pilco_float_smooth.shape

            pilco_data_done = pd.DataFrame({
                'Algorithm': ['PILCO'] * len(pilco_float_smooth),
                'epoch': pilco_epoch_expand_int,
                'length': pilco_float_smooth,
            })
            if LINE:
                sns.lineplot(x='epoch', y='length', markers=False, dashes=True, data=pilco_data_done)

    if MBPO:
        mbpo_data = pd.read_csv('D:\\桌面\\progress_MBPO_5598.csv')
        if use_seaborn:
            mbpo_rew_list = mbpo_data['training/path_return'].values[50:]
            mbpo_float_2d = rew_list_split(mbpo_rew_list)
            mbpo_float_2d[43:] += 100
            # mbpo_float_2d = smooth_batch3(mbpo_float_2d)
            mbpo_float_2d = np.concatenate([smooth_batch3(mbpo_float_2d[:5], weight=0.6), smooth_batch3(mbpo_float_2d[5:], weight=0.9)])
            # mbpo_float_2d[4:] = smooth_batch3(mbpo_float_2d[4:], weight=0.9)
            mbpo_epoch_expand = epoch_expand(mbpo_float_2d)

            mbpo_epoch_expand_int = mbpo_epoch_expand.astype(np.int)
            mbpo_float_smooth = mbpo_float_2d.flatten()
            assert mbpo_epoch_expand_int.shape == mbpo_float_smooth.shape

            mbpo_data_done = pd.DataFrame({
                'epoch': mbpo_epoch_expand_int,
                'return': mbpo_float_smooth,
                # 'category': 'MBPO',
            })
            sns.lineplot(x='epoch', y='return', data=mbpo_data_done)

        else:
            mbpo_rew_avg = mbpo_data['training/return-average'].values
            mbpo_rew_min = mbpo_data['training/return-min'].values
            mbpo_rew_max = mbpo_data['training/return-max'].values

            dim_num = len(mbpo_data)
            x = np.linspace(1, dim_num, dim_num)
            # fig, ax = plt.subplots(figsize=(10, 5))

            plt.plot(x, smooth(mbpo_rew_avg), c='r')
            ax.fill_between(x, smooth(mbpo_rew_max), smooth(mbpo_rew_min), alpha=0.1, color='orange')

    if IPK:
        ipk_data = pd.read_csv('D:\\桌面\\progress_IPK_598.csv')
        if use_seaborn:
            # ipk_rew_list = ipk_data['training/path_return'].values
            # ipk_float_2d = rew_list_split(ipk_rew_list)
            # ipk_float_2d = smooth_batch3(ipk_float_2d, weight=0.5)
            ipk_rew_list1 = ipk_data['training/return-average'].values
            # ipk_rew_list2 = [0]*len(ipk_data['training/episode-length-max'].values)
            ipk_rew_list2 = ipk_data['training/return-max'].values
            ipk_rew_list3 = ipk_data['training/return-min'].values
            ipk_float_2d = np.concatenate(([ipk_rew_list1], [ipk_rew_list2], [ipk_rew_list3]), axis=0).T
            # ipk_float_2d = np.concatenate(([ipk_rew_list1], [ipk_rew_list2]), axis=0).T
            ipk_float_2d[:6] = smooth_batch(ipk_float_2d[:6], weight=0.6)
            ipk_float_2d[5:] = smooth_batch(ipk_float_2d[5:], weight=0.9)
            ipk_epoch_expand = epoch_expand(ipk_float_2d)

            ipk_epoch_expand_int = ipk_epoch_expand.astype(np.int)
            ipk_float_smooth = ipk_float_2d.flatten()
            assert ipk_epoch_expand_int.shape == ipk_float_smooth.shape

            ipk_data_done = pd.DataFrame({
                'epoch': ipk_epoch_expand_int,
                'return': ipk_float_smooth,
                # 'category': 'IPK'
            })
            sns.lineplot(x='epoch', y='return', markers=True, dashes=False,
                         data=ipk_data_done)
        else:
            ipk_rew_avg = ipk_data['training/return-average'].values
            ipk_rew_min = ipk_data['training/return-min'].values
            ipk_rew_max = ipk_data['training/return-max'].values

            dim_num = len(ipk_data)
            x = np.linspace(1, dim_num, dim_num)
            # fig, ax = plt.subplots(figsize=(10, 5))

            plt.plot(x, smooth(ipk_rew_avg), c='r')
            ax.fill_between(x, smooth(ipk_rew_max), smooth(ipk_rew_min), alpha=0.1, color='red')

    if Basic:
        bas_data = pd.read_csv('D:\\桌面\\progress_basic.csv')
        if use_seaborn:
            bas_rew_list = bas_data['Return per epoch'].values
            bas_float_2d = rew_list_split(bas_rew_list)
            bas_float_2d = smooth_batch3(bas_float_2d)
            bas_epoch_expand = epoch_expand(bas_float_2d)

            bas_epoch_expand_int = bas_epoch_expand.astype(np.int)
            bas_float_smooth = bas_float_2d.flatten()
            assert bas_epoch_expand_int.shape == bas_float_smooth.shape

            bas_data_done = pd.DataFrame({
                'epoch': bas_epoch_expand_int,
                'return': bas_float_smooth,
                # 'category': 'Basic'
            })
            sns.lineplot(x='epoch', y='return', markers=True, dashes=False,
                         data=bas_data_done)

    # legend = ax.legend()
    # legend.texts[0].set_text("Algorithm")
    # ax.legend().set_title('title')
    # plt.legend(loc='upper right')
    # fusion_rew_list = ipk_data['sampler/last-path-return'].values[0:100]
    # fusion_rew_list = smooth(fusion_rew_list)
    # # fusion_data = pd.DataFrame({
    # #     'epoch': x,
    # #     'fusion_rew': fusion_rew_list
    # # })
    # # sns.relplot(x='epoch', y='fusion_rew', kind='line', data=fusion_data)
    # plt.plot(x, fusion_rew_list, 'mp-', linewidth=3 )

    plt.xlim((0, 100))
    plt.savefig('D:\Github\TendonTrack\Paper\IPK_1\img\Simulation_results3.eps')

    plt.show()


def plot_task_length(SAC=None, IPK=None, MBPO=None, PILCO=None):
    ax = plt.figure(figsize=(20, 10))
    sns.set(style="darkgrid", font_scale=2)
    if SAC:
        sac_data = pd.read_csv('D:\\桌面\\progress_SAC_8338.csv')
        if use_seaborn:
            sac_rew_list1 = sac_data['training/episode-length-avg'].values
            sac_rew_list2 = sac_data['training/episode-length-max'].values
            sac_rew_list3 = sac_data['training/episode-length-min'].values
            # sac_rew_list2 = [0]*len(sac_rew_list1)
            sac_float_2d = np.concatenate(([sac_rew_list1], [sac_rew_list2], [sac_rew_list3]), axis=0).T
            sac_float_2d = smooth_batch(sac_float_2d)
            # sac_float_2d = np.concatenate(([sac_rew_list1], [sac_rew_list2]), axis=0).T

            sac_epoch_expand = epoch_expand(sac_float_2d)

            sac_epoch_expand_int = sac_epoch_expand.astype(np.int)
            sac_float_smooth = sac_float_2d.flatten()
            assert sac_epoch_expand_int.shape == sac_float_smooth.shape

            sac_data_done = pd.DataFrame({
                'Algorithm': ['SAC'] * len(sac_float_smooth),
                'epoch': sac_epoch_expand_int,
                'length': sac_float_smooth,
                # 'category': 'SAC'
            })
            if LINE:
                sns.lineplot(x='epoch', y='length', markers=False, dashes=True, data=sac_data_done)
        else:
            sac_rew_avg = sac_data['training/return-average'].values
            sac_rew_min = sac_data['training/return-min'].values
            sac_rew_max = sac_data['training/return-max'].values

            dim_num = len(sac_data)
            x = np.linspace(1, dim_num, dim_num)
            # fig, ax = plt.subplots(figsize=(10, 5))

            plt.plot(x, smooth(sac_rew_avg), c='r')
            plt.fill_between(x, smooth(sac_rew_max), smooth(sac_rew_min), alpha=0.1, color='orange')
    if PILCO:
        pilco_data = pd.read_csv('D:\\桌面\\progress_PILCO.csv')
        if use_seaborn:
            pilco_rew_list = pilco_data['Task length per epoch'].values
            pilco_float_2d = rew_list_split(pilco_rew_list)
            pilco_float_2d = smooth_batch2(pilco_float_2d)
            pilco_epoch_expand = epoch_expand(pilco_float_2d)

            pilco_epoch_expand_int = pilco_epoch_expand.astype(np.int)
            pilco_float_smooth = flatten_manually(pilco_float_2d)
            assert pilco_epoch_expand_int.shape == pilco_float_smooth.shape

            pilco_data_done = pd.DataFrame({
                'Algorithm': ['PILCO'] * len(pilco_float_smooth),
                'epoch': pilco_epoch_expand_int,
                'length': pilco_float_smooth,
            })
            if LINE:
                sns.lineplot(x='epoch', y='length', markers=False, dashes=True, data=pilco_data_done)

    if MBPO:
        mbpo_data = pd.read_csv('D:\\桌面\\progress_MBPO_5598.csv')
        if use_seaborn:
            mbpo_rew_list1 = mbpo_data['training/episode-length-avg'].values[50:]
            # mbpo_rew_list2 = [0] * len(mbpo_data['training/episode-length-avg'].values[50:])
            mbpo_rew_list2 = mbpo_data['training/episode-length-max'].values[50:]
            mbpo_rew_list3 = mbpo_data['training/episode-length-min'].values[50:]
            mbpo_float_2d = np.concatenate(([mbpo_rew_list1], [mbpo_rew_list2], [mbpo_rew_list3]), axis=0).T
            # mbpo_float_2d = np.concatenate(([mbpo_rew_list1], [mbpo_rew_list2]), axis=0).T
            mbpo_float_2d = smooth_batch(mbpo_float_2d)
            mbpo_epoch_expand = epoch_expand(mbpo_float_2d)

            mbpo_epoch_expand_int = mbpo_epoch_expand.astype(np.int)
            mbpo_float_smooth = mbpo_float_2d.flatten()
            assert mbpo_epoch_expand_int.shape == mbpo_float_smooth.shape

            mbpo_data_done = pd.DataFrame({
                'Algorithm': ['MBPO'] * len(mbpo_float_smooth),
                'epoch': mbpo_epoch_expand_int,
                'length': mbpo_float_smooth,
                # 'category': 'MBPO',
            })
            if LINE:
                sns.lineplot(x='epoch', y='length', data=mbpo_data_done)

        else:
            mbpo_rew_avg = mbpo_data['training/return-average'].values
            mbpo_rew_min = mbpo_data['training/return-min'].values
            mbpo_rew_max = mbpo_data['training/return-max'].values

            dim_num = len(mbpo_data)
            x = np.linspace(1, dim_num, dim_num)
            # fig, ax = plt.subplots(figsize=(10, 5))

            plt.plot(x, smooth(mbpo_rew_avg), c='r')
            ax.fill_between(x, smooth(mbpo_rew_max), smooth(mbpo_rew_min), alpha=0.1, color='orange')

    if IPK:
        ipk_data = pd.read_csv('D:\\桌面\\progress_IPK_598.csv')
        if use_seaborn:
            ipk_rew_list1 = ipk_data['training/episode-length-avg'].values
            # ipk_rew_list2 = [0]*len(ipk_data['training/episode-length-max'].values)
            ipk_rew_list2 = ipk_data['training/episode-length-max'].values
            ipk_rew_list3 = ipk_data['training/episode-length-min'].values
            ipk_float_2d = np.concatenate(([ipk_rew_list1], [ipk_rew_list2], [ipk_rew_list3]), axis=0).T
            # ipk_float_2d = np.concatenate(([ipk_rew_list1], [ipk_rew_list2]), axis=0).T
            ipk_float_2d = smooth_batch(ipk_float_2d)
            ipk_epoch_expand = epoch_expand(ipk_float_2d)

            ipk_epoch_expand_int = ipk_epoch_expand.astype(np.int)
            ipk_float_smooth = ipk_float_2d.flatten()
            assert ipk_epoch_expand_int.shape == ipk_float_smooth.shape

            ipk_data_done = pd.DataFrame({
                'Algorithm': ['IPK'] * len(ipk_float_smooth),
                'epoch': ipk_epoch_expand_int,
                'length': ipk_float_smooth,
                # 'category': 'IPK'
            })
            if LINE:
                sns.lineplot(x='epoch', y='length',
                             data=ipk_data_done)
            else:
                sns.violinplot(x='Algorithm', y='length',
                               data=pd.concat((sac_data_done, mbpo_data_done, ipk_data_done)))
        else:
            ipk_rew_avg = ipk_data['training/return-average'].values
            ipk_rew_min = ipk_data['training/return-min'].values
            ipk_rew_max = ipk_data['training/return-max'].values

            dim_num = len(ipk_data)
            x = np.linspace(1, dim_num, dim_num)
            # fig, ax = plt.subplots(figsize=(10, 5))

            plt.plot(x, smooth(ipk_rew_avg), c='r')
            ax.fill_between(x, smooth(ipk_rew_max), smooth(ipk_rew_min), alpha=0.1, color='red')

    # all_data_done = pd.DataFrame({
    #     'epoch': [sac_epoch_expand_int, mbpo_epoch_expand_int, ipk_epoch_expand_int],
    #     'return': [sac_float_smooth, mbpo_float_smooth, ipk_float_smooth],
    #     'Algorithm': ['SAC', 'MBPO', 'IPK'],
    # })
    # sns.lineplot(x='epoch', y='return', hue='Algorithm', markers=True, dashes=False,
    #              data=all_data_done)

    if Basic:
        x = np.linspace(1, 100, 100)
        y = np.ones(100) * 1035
        plt.plot(x, y, c='b', linestyle='--', linewidth=3)

    plt.xlim((0, 100))
    # plt.ylim((0, 400))
    # plt.savefig('D:\Github\TendonTrack\Paper\IPK_1\img\\task_length.eps')

    plt.show()


def flatten_manually(float_2d):
    arr = float_2d[0]
    for i in range(1, len(float_2d)):
        arr = np.concatenate([arr, float_2d[i]], axis=0)
    return arr


def rew_list_split(rew_list):
    float_2d_np = []
    for i in range(len(rew_list)):
        rew_list_splited = rew_list[i][1:-1].split(', ')
        float_2d_np.append([])
        for j in range(len(rew_list_splited)):
            rew_single_float = float(rew_list_splited[j])
            float_2d_np[i].append(rew_single_float)
        # float_2d_np[i]=np.array([np.average(float_2d_np[i]), np.min(float_2d_np[i]), np.max(float_2d_np[i])])
    return np.array(float_2d_np)


def epoch_expand(float_2d):
    epoch_list = []
    for i in range(len(float_2d)):
        for j in range(len(float_2d[i])):
            epoch_list.append(i)
    return np.array(epoch_list)


def smooth(data, weight=0.9):
    last = data[0]
    res = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        res.append(smoothed_val)
        last = smoothed_val
    return np.array(res)


def smooth_batch(data, weight=0.9):
    for i in range(len(data) - 1):
        for j in range(data.shape[1]):
            data[i + 1][j] = weight * data[i][j] + (1 - weight) * data[i + 1][j]
    return data


def smooth_batch2(data, weight=0.6):
    for i in range(len(data) - 1):
        for j in range(len(data[i + 1])):
            try:
                data[i + 1][j] = weight * np.average(data[i]) + (1 - weight) * data[i + 1][j]
            except:
                print(i, j)
    return data


def smooth_batch3(data, weight=0.9):
    new_data = np.zeros((len(data), 3))
    for i in range(len(new_data)):
        new_data[i] = np.array([np.average(data[i]), np.min(data[i]), np.max(data[i])])
    for i in range(len(new_data) - 1):
        for j in range(len(new_data[i + 1])):
            new_data[i + 1][j] = weight * new_data[i][j] + (1 - weight) * new_data[i + 1][j]
    return new_data


def KL_diver_zeta_plot():
    KL_data = pd.read_csv('C:\\Users\\Skylark\\Desktop\\progress_IPK_5880_2.csv')
    KL_data_mean = KL_data['sampler/KL-mean'].values[0:100]
    # KL_data_max = KL_data['sampler/KL-max'].values
    # KL_data_min = KL_data['sampler/KL-min'].values

    dim_num = 100
    x = np.linspace(1, dim_num, dim_num)

    fig = plt.figure(figsize=(10, 6))
    sns.set(style="darkgrid", font_scale=2)
    sns.set_palette("hls")
    # sns.color_palette("Paired", 8)
    # sns.choose_cubehelix_palette()
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(x, smooth(KL_data_mean), color='deepskyblue', linewidth=3, label='KL-divergence')
    ax1.set_ylabel('KL-divergence')
    ax1.set_title("KL-divergence & $\zeta$ coefficient")

    zeta_data_mean = np.zeros(dim_num)
    for i in range(dim_num):
        zeta_data_mean[i] = np.tanh((KL_data_mean[i] - 4) * 0.1)

    ax2 = ax1.twinx()  # this is the important function
    l2 = ax2.plot(x, smooth(zeta_data_mean), color='orange', linewidth=3, label='$\zeta_{basic}$')
    # ax2.set_xlim([0, np.e])
    ax2.set_ylabel('$\zeta$ coefficient')
    ax2.set_xlabel('Same X for both exp(-x) and ln(x)')
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper right')
    # ax.fill_between(x, smooth(KL_data_max), smooth(KL_data_min), alpha=0.1, color='red')
    plt.savefig('D:\Github\TendonTrack\Paper\IPK_1\img\KL_zeta.eps')
    plt.show()


def zeta_plot():
    KL_data = pd.read_csv('C:\\Users\\Skylark\\Desktop\\progress_IPK_5880.csv')
    KL_data_mean = KL_data['sampler/KL-mean'].values
    KL_data_mean = smooth(KL_data_mean)
    dim_num = len(KL_data)
    zeta_data_mean = np.zeros(dim_num)
    for i in range(dim_num):
        zeta_data_mean[i] = np.tanh((KL_data_mean[i] - 2) * 0.2)

    x = np.linspace(1, dim_num, dim_num)
    fig, ax = plt.subplots(figsize=(10, 5))

    plt.plot(x, zeta_data_mean, c='r')
    # ax.fill_between(x, smooth(KL_data_max), smooth(KL_data_min), alpha=0.1, color='red')
    plt.show()


if __name__ == '__main__':
    LINE = True
    SAC = True
    MBPO = True
    IPK = True
    PILCO = True
    Basic = True
    use_seaborn = True
    plot_results(SAC=SAC, MBPO=MBPO, IPK=IPK, Basic=Basic, PILCO=PILCO)
    # plot_task_length(SAC=SAC, MBPO=MBPO, IPK=IPK, PILCO=PILCO)
    # KL_diver_zeta_plot()
    # zeta_plot()
