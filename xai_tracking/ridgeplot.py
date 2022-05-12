import numpy as np
import pandas as pd
import plotly.graph_objects as go
import base64
from plotly.figure_factory import create_distplot
import plotly.express as px
from scipy.stats import gaussian_kde
def image(data):
    fig = px.imshow(data)
    return fig
    fig.show()
def ridge_plot(data, linewidth=2, sample_weights=None, highlight=None):

    # fig = go.Figure()
    # for ii, group in enumerate(data):
    #     i = len(data) - 1 - ii
    #     scale = 0.4 * i / len(data) + 0.6
    #     color = "white"
        
    #     fig.add_trace(go.Violin(x=data[group], line_color=color, fillcolor=f"rgba({int(scale * 39)},{int(scale * 119)},{int(scale * 180)}, 1.0)"))

    # fig.update_traces(orientation='h', side='positive', width=3, points=False)
    # fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    # fig.show()
    # return fig

    traces = []
    yaxis = {}
    height = 0.75 / (0.75 * len(data) + 0.25)
    overlapping_height = 4.0 / 3.0 * height
    colors = px.colors.sample_colorscale("emrld", len(data))
    xmin = np.inf
    xmax = -np.inf
    for group in data:
        tmp = data[group].min()
        if tmp < xmin:
            xmin = tmp
        tmp = data[group].max()
        if tmp > xmax:
            xmax = tmp
    if abs(xmax) > abs(xmin):
        xmin = -xmax
    else:
        xmax = -xmin
    ymax = 0
    for ii, group in enumerate(sorted(data.keys())):
        weights = None if sample_weights is None else sample_weights[group]
        i = len(data) - 1 - ii
        scale = 0.5 * i / len(data) + 0.5
        x = data[group]
        kde_x = np.linspace(1.25*xmin, 1.25*xmax, 1000)
        kde_y = gaussian_kde(x, weights=weights, bw_method=1e-1)(kde_x)
        f = px.line(x=kde_x, y=kde_y)
        # Restyle Trace a bit
        trace = f["data"][0]
        yaxis_name = f"yaxis{str(i + 1) if i else ''}"
        trace["yaxis"] = yaxis_name.replace("axis", "")
        trace["fill"] = "tozeroy"
        if group == highlight:
            trace["fillcolor"] = "rgb(247, 184, 1)"
            trace["line"] = dict(color="rgb(247, 184, 1)", width=linewidth)
        else:
            trace["fillcolor"] = colors[ii]#f"rgba({scale * 39},{scale * 119},{scale * 180},1.0)"
            trace["line"] = dict(color=trace["fillcolor"], width=linewidth)

        traces.append(trace)
        if kde_y.max() > ymax:
            ymax = kde_y.max()
        # Style overlapping y-Axis
        yaxis[yaxis_name] = dict(
            domain=[ii * height, ii * height + overlapping_height],
            position=0.0,
            title=dict(text=group),
            title_font=dict(color="rgb(247, 184, 1)" if group == highlight else None),
            showgrid=False,
            zeroline=True,
            zerolinecolor=trace["fillcolor"],
            zerolinewidth=2*linewidth,
            showticklabels=False,
            fixedrange=True,
        )
    
    # Style x-Axis
    yaxis["xaxis"] = dict(
        position=0,
        showgrid=False,
        zeroline=False,
        showticklabels=True,
        fixedrange=True,
        
    )
    for ax in yaxis:
        if ax.startswith("y"):
            yaxis[ax]["range"] = (0, ymax)

    layout = go.Layout(
        paper_bgcolor="rgba(137, 173, 187, 0.1)",
        plot_bgcolor='rgba(0,0,0,0)',
        **yaxis
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(showlegend=False, hovermode="x")
    return fig
    fig.show(config=dict(displayModeBar=False))

if __name__ == "__main__":
    # Create the data
    rs = np.random.RandomState(1979)
    data = {}
    sample_weights = {}
    for i, g in enumerate([x + "    " for x in ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck",]]):
        data[g] = rs.randn(5000) + i
        sample_weights[g] = rs.rand(5000) 
    print(data)
    fig = ridge_plot(data, sample_weights==sample_weights)
    fig.update_layout(margin=dict(t=80))
    
    plotly_logo = base64.b64encode(open("qrcode.png", "rb").read())
    fig.add_layout_image(
    dict(
            source='data:image/png;base64,{}'.format(plotly_logo.decode()),
            xref="paper", yref="paper",
            x=1, y=1,
            sizex=0.16, sizey=0.16,
            xanchor="center", yanchor="middle"
        )
    )
    import plotly.io as pio
    pio.kaleido.scope.default_height = 1000
    pio.kaleido.scope.default_width = 1400
    pio.kaleido.scope.mathjax = None
    fig.write_image("test.pdf")
