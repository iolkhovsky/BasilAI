import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tempfile


matplotlib.use('Agg')


def plot_attention_scores(input_words, output_words, scores, figsize=(4, 4), dpi=250, fontsize=4):
    outputs_size, inputs_size = scores.shape
    assert len(input_words) == inputs_size
    assert len(output_words) == outputs_size
    assert np.amax(scores) <= 1. and np.amin(scores) >= 0.

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(scores, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(inputs_size))
    ax.set_xticklabels(input_words, rotation=90, fontsize=fontsize)
    ax.tick_params(axis='x', which='both', length=0, pad=10, labelbottom=False, labeltop=True)
    ax.set_yticks(np.arange(outputs_size))
    ax.set_yticklabels(output_words, fontsize=fontsize)
    
    ax.set_xlabel('Input sequence')
    ax.set_ylabel('Output sequence')

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as temp_file:
        fig.savefig(temp_file.name)
        image = plt.imread(temp_file.name)[:, :, :3]

    plt.close(fig)
    
    return image
