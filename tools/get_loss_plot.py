import numpy as np



if __name__ == '__main__':
    file_name = "./log.txt"
    with open(file_name) as f:
        lines = f.readlines()
    loss, lbox, lobj, lcls = [], [], [], []
    for l in lines:
        if "loss: " in l:
            loss.append(float(l[l.find("loss: ") + len("loss: "):l.find(", lbox:")]))
            lbox.append(float(l[l.find("lbox: ") + len("lbox: "):l.find(", lobj:")]))
            lobj.append(float(l[l.find("lobj: ") + len("lobj: "):l.find(", lcls:")]))
            lcls.append(float(l[l.find("lcls: ") + len("lcls: "):l.find(", cur_lr:")]))


    import numpy as np
    import matplotlib.pyplot as plt
    import pylab as pl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    loss, lbox, lobj, lcls = np.array(loss), np.array(lbox), np.array(lobj), np.array(lcls)
    x = np.array(range(len(loss)))

    loss = loss[924:]
    x = x[924:]

    fig = plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签
    # pl.plot(np.array(range(len(loss))), loss, 'b-', label=u'main loss')
    plt.plot(x, loss, linewidth=0.1, linestyle="solid", label="main loss", color='b')
    plt.legend()
    plt.title('Loss curve')
    plt.show()



