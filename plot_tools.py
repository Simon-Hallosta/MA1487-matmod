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