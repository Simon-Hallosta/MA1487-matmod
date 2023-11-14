#%%
import numpy as np
import scipy.stats as st

import bokeh.io
import bokeh.layouts
import bokeh.plotting


def plot_pmf(p, x, dist, params, **kwargs):
    """
    Generate plot of PMF at specified values of x.
    """
    y = dist.pmf(x, *params, **kwargs)
    p.circle(x, y, size=7, color='dodgerblue')
    p.segment(x0=x, x1=x, y0=0, y1=y, line_width=3, color='dodgerblue')
    return p


def plot_pdf(p, x, dist, params, **kwargs):
    """
    Generate plot of PDF at specified values of x.
    """
    y = dist.pdf(x, *params, **kwargs)
    p.line(x, y, line_width=3, color='dodgerblue')
    return p


def plot_discrete_cdf(p, x, dist, params, show_segments=False, **kwargs):
    """
    Plot CDF of discrete distribution.
    """
    y = dist.cdf(x, *params, **kwargs)
    if show_segments:
        p.circle(x[:-1], y[:-1], size=7, color='dodgerblue')
        p.segment(x0=x[:-1], x1=x[1:], y0=y[:-1], y1=y[:-1], line_width=3,
                  color='dodgerblue')
    else:
        p.circle(x, y, size=7, color='dodgerblue')
    return p


def plot_continuous_cdf(p, x, dist, params, **kwargs):
    """
    Plot CDF of continuous distribution.
    """
    y = dist.cdf(x, *params, **kwargs)
    p.line(x, y, line_width=3, color='dodgerblue')
    return p


def plot_dists(x, dist, params, param_names, dist_name, **kwargs):
    """
    Plot PDF/PMF and CDF next to each other.
    """
    # Tools for plots
    tools='pan,wheel_zoom,reset'
    
    # Title for plots
    t1 = dist_name + ', ' \
            + ''.join([param_names[i] + ' = ' + str(params[i]) + ', ' 
                                      for i in range(len(params) - 1)])
    t1 += param_names[-1] + ' = ' + str(params[-1])
    
    # Last half of y-axis label
    ylabel = ''.join(pname + ', ' for pname in param_names[:-1])
    ylabel += param_names[-1] + ')'
    
    # Set up plots
    p1 = bokeh.plotting.figure(width=325, height=250, tools=tools, title=t1)
    p2 = bokeh.plotting.figure(width=325, height=250, tools=tools, 
                               y_range=[-0.05, 1.05])

    # Make plots
    if hasattr(dist, 'pmf'):
        p1 = plot_pmf(p1, x, dist, params, **kwargs)
        p1.xaxis.axis_label = 'k'
        p1.yaxis.axis_label = 'P(k; ' + ylabel
        p2 = plot_discrete_cdf(p2, x, dist, params, **kwargs)
        p2.xaxis.axis_label = 'k'
        p2.yaxis.axis_label = 'F(k; ' + ylabel
    else:
        p1 = plot_pdf(p1, x, dist, params, **kwargs)
        p1.xaxis.axis_label = 'x'
        p1.yaxis.axis_label = 'P(x; ' + ylabel
        p2.yaxis.axis_label = 'F(x; ' + ylabel
        p2 = plot_continuous_cdf(p2, x, dist, params, **kwargs)
        p2.xaxis.axis_label = 'x'
        p2.yaxis.axis_label = 'F(k; ' + ylabel
        
    # Link the x-axes
    p1.x_range = p2.x_range

    return bokeh.layouts.row([p1, p2])





#%% Binmoial distribution simulation functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from IPython.display import HTML

tree_depth = 5

# Function to plot the base of the probability tree with probabilities
def plot_tree_with_probabilities(num_trials, ax, p_success=0.5):
    ax.clear()
    for level in range(num_trials):
        for node in range(2 ** level):
            node_x = node * (1 / (2 ** level)) + (1 / (2 ** (level + 1)))
            node_y = 1 - (level / num_trials)
            # Draw the left edge for success
            ax.plot([node_x, node_x - (1 / (2 ** (level + 2)))], [node_y, node_y - (1 / num_trials)], 'grey')
            # Draw the right edge for failure
            ax.plot([node_x, node_x + (1 / (2 ** (level + 2)))], [node_y, node_y - (1 / num_trials)], 'grey')
            # Display the probability on the edges
            if level < num_trials - 1:  # No text for the last level
                ax.text((node_x + node_x - (1 / (2 ** (level + 2)))) / 2,
                        (node_y + node_y - (1 / num_trials)) / 2 - 0.05, f'{p_success:.2f}',
                        horizontalalignment='center', verticalalignment='center', fontsize=8, color='blue')
                ax.text((node_x + node_x + (1 / (2 ** (level + 2)))) / 2,
                        (node_y + node_y - (1 / num_trials)) / 2 - 0.05, f'{1-p_success:.2f}',
                        horizontalalignment='center', verticalalignment='center', fontsize=8, color='red')

# Function to animate the paths with probabilities
def animate_paths_with_probabilities(num_trials, ax, paths, p_success=0.5):
    for path in paths:
        current_prob = 1
        node_x = 0.5
        node_y = 1
        ax.plot(node_x, node_y, 'ko')  # Start point
        for i, step in enumerate(path):
            # Calculate probability
            current_prob *= p_success if step == 'S' else (1 - p_success)
            # Draw path
            next_node_x = node_x - (1 / (2 ** (i + 2))) if step == 'S' else node_x + (1 / (2 ** (i + 2)))
            next_node_y = node_y - (1 / num_trials)
            ax.plot([node_x, next_node_x], [node_y, next_node_y], 'b-' if step == 'S' else 'r-')
            ax.plot(next_node_x, next_node_y, 'ko')  # Node point
            # Move to next node
            node_x, node_y = next_node_x, next_node_y
        # Display the cumulative probability at the end of the path
        ax.text(node_x, node_y - 0.05, f'{current_prob:.5f}',
                horizontalalignment='center', verticalalignment='center', fontsize=8, color='black')


# Function to simulate trials
def simulate_trials(tree_depth, num_paths, p_success=0.5):
    paths = []
    for _ in range(num_paths):
        trial = ""
        for _ in range(tree_depth):
            trial += 'S' if np.random.rand() < p_success else 'F'
        paths.append(trial)
    return paths

# Animate the traversal of the tree
def animate(i, ax, p_success, tree_depth, trial_paths):
    # Clear the figure and redraw the base tree with probabilities
    plot_tree_with_probabilities(tree_depth, ax, p_success)
    # Animate the paths with probabilities for the current trial
    animate_paths_with_probabilities(tree_depth, ax, [trial_paths[i]], p_success)
    ax.set_title(f'Trial {i+1}')
    # Remove the axis
    ax.axis('off')
    return ax,

def generate_probability_tree_animation(num_paths, p_success, tree_depth, figsize=(12, 8)):
    # Function to generate the animation
    # Simulate trials
    trial_paths = simulate_trials(tree_depth, num_paths, p_success)

    # Set up the figure and axes for the animation
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # Set the background color of the figure to white

    # Create the animation with the updated function
    anim = FuncAnimation(fig, animate, frames=len(trial_paths), fargs=(ax, p_success, tree_depth, trial_paths), interval=1000, repeat=False)

    return HTML(anim.to_jshtml())
# %%
