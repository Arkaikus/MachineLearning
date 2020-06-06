import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_compare_nodes(warehouse, plot_save_location, **kwargs):
    models = kwargs.get('models', [])
    start_offset = kwargs.get('start_offset', 5)
    end_offset = kwargs.get('end_offset', 50)
    x_label = kwargs.get('x_label', "")
    y_label = kwargs.get('y_label', "")
    figsize = kwargs.get('figsize',  (18, 50))
    dpi = kwargs.get('dpi', 120)
    title = kwargs.get('title', "Datos")
    title_fontsize = kwargs.get('title_fontsize', 20)
    label_fontsize = kwargs.get('label_fontsize', 16)
    legend_fontsize = kwargs.get('legend_fontsize', 16)
    ticks_fontsize = kwargs.get('ticks_fontsize', 14)
    xticks_div = kwargs.get('xticks_div', 3)
    yticks_div = kwargs.get('yticks_div', 6)

    # Units between 0 and dataset X shape
    len_values = sum(warehouse.df.shape)

    if end_offset > len_values - 1 or start_offset < warehouse.len_inputs:
        raise Exception(
            'must: end_offset({})<={} and start_offset({})>={}'.format(end_offset, len_values, start_offset,
                                                                       warehouse.len_inputs))

    # X axis values for predictions
    x_values = list(range(len_values))
    node_index = x_values[start_offset:end_offset]

    # X axis values for next nodes
    next_nodes_index = x_values[start_offset:end_offset]
    # Y axis values
    next_nodes_values = warehouse.y_values[start_offset - warehouse.len_inputs + 1:end_offset - warehouse.len_inputs + 1]

    x_ticks = list(np.concatenate((warehouse.X_values[0], warehouse.y_values[:])))
    x_ticks = list(map(lambda x: str(x), x_ticks))[start_offset:end_offset]

    #         print('len_values',len_values)
    #         print('node_index',node_index,len(node_index))
    #         print('next_nodes_index',next_nodes_index,len(next_nodes_index))
    #         print('next_nodes_values',next_nodes_values,len(next_nodes_values))
    #         print('x_ticks',x_ticks,len(x_ticks))
    #         return

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.plot(next_nodes_index, next_nodes_values, 'Dk', label="Datos",
            markerfacecolor='w',
            markeredgewidth=1.5,
            markeredgecolor=(0, 0, 0, 1))

    color_markers = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]

    for model,color in zip(models, color_markers):
        if model.trained:
            predictions = model.get_predictions()
            predictions = predictions[start_offset - warehouse.len_inputs:end_offset - warehouse.len_inputs]
            ax.plot(node_index, predictions, 'D', label=model.name,
                    markerfacecolor='w', markeredgewidth=1.5, markeredgecolor=color)

    # ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    ax.legend(loc='upper left', fontsize=legend_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_xticks(node_index)
    ax.set_xticklabels(x_ticks)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=ticks_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=ticks_fontsize)
    plt.tight_layout()
    plt.show()
    fig.savefig(plot_save_location, dpi=dpi)


def plot_grid_nodes(self,
                    div,
                    plot_save_location,
                    figsize=(10, 30),
                    dpi=120,
                    alphaData=0.05,
                    alpha=0.1,
                    linewidth=0.01,
                    title_fontsize=20,
                    label_fontsize=16,
                    legend_fontsize=16,
                    ticks_fontsize=14,
                    xticks_div=3,
                    yticks_div=6
                    ):
    
    fig, ax = plt.subplots(3, 1, figsize=figsize)
    # ax.imshow(data, cmap=cmap, norm=norm)

    min_val = min(self.y_values)
    max_val = max(self.y_values)
    print(max_val)
    # print(sqrt(max_val))
    div = 1900 / div

    vals_xy = (max_val % div, div)
    for i in range(3):
        # draw gridlines
        ax[i].grid(which='major', axis='both', linestyle='-', color='k', linewidth=linewidth)
        ax[i].set_xticks(np.arange(0, vals_xy[0] + 1, 1));
        ax[i].set_yticks(np.arange(0, vals_xy[1] + 1, 1));
        #             ax[i].set_xticklabels(np.arange(0,vals_xy[0]+1, 20))
        #             ax[i].set_yticklabels(np.arange(0,vals_xy[1]+1, 20));
        plt.setp(ax[i].get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=ticks_fontsize)
        plt.setp(ax[i].get_yticklabels(), fontsize=ticks_fontsize)

        for value in self.y_values:
            nx = int(value / div) - 1
            ny = (value % div) - 1
            xy = (ny, nx)
            rectangle = Rectangle(xy, 1, 1, alpha=alphaData, facecolor='k')
            ax[i].add_patch(rectangle)

    if self.svm_trained:
        for svm_value in self.get_svm_predictions():
            nx = int(svm_value / div) - 1
            ny = (svm_value % div) - 1
            xy = (ny, nx)
            # print(xy)
            rectangleSVM = Rectangle(xy, 1, 1, alpha=alpha, facecolor='r')
            ax[0].add_patch(rectangleSVM)
            ax[0].set_title('Máquina de Soporte Vectorial (SVR)', fontsize=title_fontsize)

    if self.mlp_trained:
        for mlp_value in self.get_mlp_predictions():
            nx = int(mlp_value / div) - 1
            ny = (mlp_value % div) - 1
            xy = (ny, nx)
            rectangleMLP = Rectangle(xy, 1, 1, alpha=alpha, facecolor='g')
            ax[1].add_patch(rectangleMLP)
            ax[1].set_title('Red Neuronal MLPRegressor', fontsize=title_fontsize)

    if self.sequential_trained:
        for sequential_value in self.get_sequential_predictions():
            nx = int(sequential_value / div) - 1
            ny = (sequential_value % div) - 1
            xy = (ny, nx)
            rectangleSequential = Rectangle(xy, 1, 1, alpha=alpha, facecolor='b')
            ax[2].add_patch(rectangleSequential)
            ax[2].set_title('Red Neuronal Keras', fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()
    fig.savefig(plot_save_location, dpi=dpi)
