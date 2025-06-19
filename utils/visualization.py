import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pydeck as pdk
from typing import Dict, List, Any
from streamlit_elements import elements, dashboard, mui, nivo

from config import METRIC_DIMENSIONS


def create_radar_chart(
    dimension_scores: Dict[str, float], benchmark_scores: Dict[str, float] = None
):
    """
    Create an enhanced radar chart for dimension scores with advanced styling.

    Args:
        dimension_scores: Dictionary of dimension names and scores
        benchmark_scores: Optional dictionary of benchmark scores for comparison

    Returns:
        plotly.graph_objects.Figure: Radar chart figure
    """
    dimensions = list(dimension_scores.keys())
    scores = list(dimension_scores.values())

    # Colors from our theme
    primary_color = "#45C4AF"
    secondary_color = "#FFEB85"

    # Create benchmark data if provided
    if benchmark_scores:
        benchmark_values = [benchmark_scores.get(dim, 0) for dim in dimensions]
        df = pd.DataFrame(
            {
                "dimension": dimensions * 2,
                "score": scores + benchmark_values,
                "type": ["Texto Atual"] * len(dimensions)
                + ["Benchmark"] * len(dimensions),
            }
        )

        fig = px.line_polar(
            df,
            r="score",
            theta="dimension",
            color="type",
            line_close=True,
            color_discrete_map={
                "Texto Atual": primary_color,
                "Benchmark": secondary_color,
            },
            range_r=[0, 100],
        )
    else:
        df = pd.DataFrame({"dimension": dimensions, "score": scores})

        fig = px.line_polar(
            df,
            r="score",
            theta="dimension",
            line_close=True,
            color_discrete_sequence=[primary_color],
            range_r=[0, 100],
        )

    # Enhanced styling
    fig.update_traces(
        fill="toself",
        fillcolor="rgba(69, 196, 175, 0.2)",
        line=dict(width=3),
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="white"),
                gridcolor="rgba(255, 255, 255, 0.1)",
                linecolor="rgba(255, 255, 255, 0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(color="white", size=12),
                rotation=90,
                direction="clockwise",
                gridcolor="rgba(255, 255, 255, 0.1)",
                linecolor="rgba(255, 255, 255, 0.2)",
            ),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        showlegend=True,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(20, 45, 78, 0.5)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
        ),
    )

    return fig


def create_3d_radar_chart(
    dimension_scores: Dict[str, float], benchmark_scores: Dict[str, float] = None
):
    """
    Create a 3D radar chart using plotly's 3D surface plot.

    Args:
        dimension_scores: Dictionary of dimension names and scores
        benchmark_scores: Optional dictionary of benchmark scores for comparison

    Returns:
        plotly.graph_objects.Figure: 3D radar chart figure
    """
    dimensions = list(dimension_scores.keys())
    dimension_names = [METRIC_DIMENSIONS[dim]["name"] for dim in dimensions]
    scores = list(dimension_scores.values())

    # Create angles for the radar dimensions
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    # Close the loop
    angles.append(angles[0])
    dimension_names.append(dimension_names[0])
    scores.append(scores[0])

    # Calculate x and y coordinates
    x = [score * np.cos(angle) for score, angle in zip(scores, angles)]
    y = [score * np.sin(angle) for score, angle in zip(scores, angles)]
    z = [0] * len(x)  # Base at z=0

    # Create the 3D figure
    fig = go.Figure()

    # Add the radar outline
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+markers",
            line=dict(color="#45C4AF", width=5),
            marker=dict(size=6, color="#FFEB85"),
            name="Pontuação Atual",
            hovertemplate="%{text}: %{customdata:.1f}<extra></extra>",
            text=dimension_names,
            customdata=scores,
        )
    )

    # Add a filled surface below the radar line (3D effect)
    # Create a finer mesh for the surface
    r_mesh = np.linspace(0, 100, 20)
    theta_mesh = np.linspace(0, 2 * np.pi, len(dimensions) + 1)

    r_grid, theta_grid = np.meshgrid(r_mesh, theta_mesh)

    # Create surface values based on actual scores
    z_grid = np.zeros_like(r_grid)
    for i, angle in enumerate(theta_grid[:-1, 0]):
        # Find the closest angle in our scores
        idx = i % len(dimensions)
        # Create a gradient effect from 0 to the score value
        z_grid[i] = r_grid[i] * scores[idx] / 100

    # Close the loop
    z_grid[-1] = z_grid[0]

    # Convert to Cartesian coordinates
    x_surface = r_grid * np.cos(theta_grid)
    y_surface = r_grid * np.sin(theta_grid)

    # Create a colorscale from our theme colors
    colorscale = [
        [0, "rgba(14, 42, 79, 0.1)"],
        [0.5, "rgba(69, 196, 175, 0.3)"],
        [1, "rgba(69, 196, 175, 0.7)"],
    ]

    # Add the surface
    fig.add_trace(
        go.Surface(
            x=x_surface,
            y=y_surface,
            z=z_grid,
            colorscale=colorscale,
            showscale=False,
            opacity=0.8,
            name="Superfície 3D",
        )
    )

    # Add benchmark data if provided
    if benchmark_scores:
        benchmark_values = [benchmark_scores.get(dim, 0) for dim in dimensions[:-1]]
        benchmark_values.append(benchmark_values[0])  # Close the loop

        # Calculate x and y coordinates for benchmark
        x_benchmark = [
            score * np.cos(angle) for score, angle in zip(benchmark_values, angles)
        ]
        y_benchmark = [
            score * np.sin(angle) for score, angle in zip(benchmark_values, angles)
        ]
        z_benchmark = [5] * len(x_benchmark)  # Slightly above the main radar

        fig.add_trace(
            go.Scatter3d(
                x=x_benchmark,
                y=y_benchmark,
                z=z_benchmark,
                mode="lines",
                line=dict(color="#FFEB85", width=3, dash="dash"),
                name="Benchmark",
                hovertemplate="%{text}: %{customdata:.1f}<extra></extra>",
                text=dimension_names,
                customdata=benchmark_values,
            )
        )

    # Add dimension labels
    for i, (name, angle) in enumerate(zip(dimension_names[:-1], angles[:-1])):
        # Position labels slightly outside the maximum score point
        label_dist = 110
        x_label = label_dist * np.cos(angle)
        y_label = label_dist * np.sin(angle)

        fig.add_trace(
            go.Scatter3d(
                x=[x_label],
                y=[y_label],
                z=[0],
                mode="text",
                text=[name],
                textposition="middle center",
                textfont=dict(color="white", size=12),
                showlegend=False,
                hoverinfo="none",
            )
        )

    # Add axis lines for reference
    for angle in angles[:-1]:
        x_line = [0, 100 * np.cos(angle)]
        y_line = [0, 100 * np.sin(angle)]
        z_line = [0, 0]

        fig.add_trace(
            go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode="lines",
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
                showlegend=False,
                hoverinfo="none",
            )
        )

    # Add concentric circles for reference
    for radius in [25, 50, 75, 100]:
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = radius * np.cos(circle_theta)
        y_circle = radius * np.sin(circle_theta)
        z_circle = np.zeros_like(x_circle)

        fig.add_trace(
            go.Scatter3d(
                x=x_circle,
                y=y_circle,
                z=z_circle,
                mode="lines",
                line=dict(color="rgba(255, 255, 255, 0.1)", width=1),
                showlegend=False,
                hoverinfo="none",
            )
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "Análise Multidimensional 3D",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"color": "white", "size": 24},
        },
        scene=dict(
            xaxis=dict(
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[-120, 120],
            ),
            yaxis=dict(
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[-120, 120],
            ),
            zaxis=dict(
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[-10, 20],
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.4),
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.5), center=dict(x=0, y=0, z=-0.1)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(color="white"),
            bgcolor="rgba(20, 45, 78, 0.5)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
        ),
        autosize=True,
        height=700,
    )

    return fig


def create_bar_chart(metric_data: Dict[str, Any], dimension: str):
    """
    Create a bar chart for detailed metrics within a dimension.

    Args:
        metric_data: Dictionary containing metric data
        dimension: The dimension to display metrics for

    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    metrics = []
    scores = []
    expected_ranges = []

    # Process metric data
    for metric_key, metric_info in metric_data.items():
        metrics.append(metric_info.get("name", metric_key))
        scores.append(metric_info.get("score", 0))

        # Get expected range midpoint
        expected_range = metric_info.get("expected_range", (50, 80))
        expected_ranges.append(sum(expected_range) / 2)

    # Create dataframe
    df = pd.DataFrame(
        {"Métrica": metrics, "Pontuação": scores, "Referência": expected_ranges}
    )

    # Get dimension color
    dimension_color = METRIC_DIMENSIONS[dimension]["color"]

    # Create figure
    fig = go.Figure()

    # Add bars for scores
    fig.add_trace(
        go.Bar(
            x=df["Métrica"],
            y=df["Pontuação"],
            name="Pontuação",
            marker_color=dimension_color,
        )
    )

    # Add markers for reference values
    fig.add_trace(
        go.Scatter(
            x=df["Métrica"],
            y=df["Referência"],
            mode="markers",
            name="Referência",
            marker=dict(color="#10b981", size=10, symbol="diamond"),
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Métricas de {METRIC_DIMENSIONS[dimension]['name']}",
        xaxis_title="",
        yaxis_title="Pontuação",
        yaxis=dict(range=[0, 100]),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_score_gauge(score: float, title: str = "Pontuação Global"):
    """
    Create a gauge chart for displaying a score.

    Args:
        score: Score value (0-100)
        title: Title for the gauge

    Returns:
        plotly.graph_objects.Figure: Gauge chart figure
    """
    # Determine color based on score
    if score >= 80:
        color = "#10b981"  # Green
    elif score >= 60:
        color = "#f59e0b"  # Yellow/Orange
    else:
        color = "#ef4444"  # Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#b91c1c"},
                    {"range": [40, 70], "color": "#b45309"},
                    {"range": [70, 100], "color": "#047857"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=30, b=30),
    )

    return fig


def create_dimension_bar_chart(dimension_scores: Dict[str, float]):
    """
    Create a horizontal bar chart for dimension scores.

    Args:
        dimension_scores: Dictionary of dimension names and scores

    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    dimensions = []
    scores = []
    colors = []

    # Process dimension data
    for dim_key, score in dimension_scores.items():
        if dim_key in METRIC_DIMENSIONS:
            dimensions.append(METRIC_DIMENSIONS[dim_key]["name"])
            scores.append(score)
            colors.append(METRIC_DIMENSIONS[dim_key]["color"])

    # Create figure
    fig = go.Figure(
        go.Bar(x=scores, y=dimensions, orientation="h", marker_color=colors)
    )

    # Update layout
    fig.update_layout(
        title="Pontuação por Dimensão",
        xaxis_title="Pontuação",
        yaxis=dict(
            title="", autorange="reversed"  # To display the first dimension at the top
        ),
        xaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def highlight_text(text: str, annotations: List[Dict[str, Any]]):
    """
    Create HTML with highlighted text based on annotations.

    Args:
        text: Original text
        annotations: List of annotation dictionaries

    Returns:
        str: HTML string with highlighted text
    """
    if not annotations:
        return text

    # Sort annotations by start position
    sorted_annotations = sorted(annotations, key=lambda x: x["span_start"])

    # Build HTML with highlights
    html_parts = []
    last_pos = 0

    for annotation in sorted_annotations:
        start = annotation["span_start"]
        end = annotation["span_end"]

        # Text before the annotation
        if start > last_pos:
            html_parts.append(text[last_pos:start])

        # Severity color mapping
        severity = annotation.get("severity", "low")
        color_map = {
            "high": "#b91c1c",  # Dark red for better contrast
            "medium": "#b45309",  # Dark orange for better contrast
            "low": "#047857",  # Dark green for better contrast
        }
        bg_color = color_map.get(severity, "#f3f4f6")

        # Highlighted text with tooltip
        tooltip = annotation.get("description", "")

        html_parts.append(
            f'<span style="background-color: {bg_color}; padding: 0 2px; border-radius: 2px;" title="{tooltip}">{text[start:end]}</span>'
        )

        last_pos = end

    # Add remaining text
    if last_pos < len(text):
        html_parts.append(text[last_pos:])

    return "".join(html_parts)


def create_timeline_chart(metrics_history: List[Dict[str, Any]]):
    """
    Create an enhanced line chart showing metrics evolution over time with animations.

    Args:
        metrics_history: List of metrics dictionaries with timestamps

    Returns:
        plotly.graph_objects.Figure: Line chart figure
    """
    if not metrics_history:
        return None

    # Extract timestamps and scores
    timestamps = [entry["timestamp"] for entry in metrics_history]
    overall_scores = [entry["overall_score"] for entry in metrics_history]

    # Create figure
    fig = go.Figure()

    # Add overall score line with improved styling
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=overall_scores,
            mode="lines+markers",
            name="Pontuação Global",
            line=dict(color="#45C4AF", width=4, shape="spline"),
            marker=dict(size=10, color="#FFEB85", line=dict(color="#45C4AF", width=2)),
            fill="tozeroy",
            fillcolor="rgba(69, 196, 175, 0.1)",
        )
    )

    # Add dimension scores if available with improved styling
    if "dimensions" in metrics_history[0]:
        for dim_key, dim_info in METRIC_DIMENSIONS.items():
            if dim_key in metrics_history[0]["dimensions"]:
                dim_scores = [
                    entry["dimensions"][dim_key]["score"] for entry in metrics_history
                ]

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=dim_scores,
                        mode="lines+markers",
                        name=dim_info["name"],
                        line=dict(
                            color=dim_info["color"], width=2, dash="dot", shape="spline"
                        ),
                        marker=dict(size=6, color=dim_info["color"], opacity=0.7),
                    )
                )

    # Update layout with enhanced styling
    fig.update_layout(
        title={
            "text": "Evolução da Qualidade Textual",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"color": "white", "size": 24},
        },
        xaxis=dict(
            title="Data",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
        ),
        yaxis=dict(
            title="Pontuação",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            range=[0, 100],
            gridcolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.2)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="white"),
            bgcolor="rgba(20, 45, 78, 0.5)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode="x unified",
        autosize=True,
        # Add animation capabilities
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        label="Animar",
                        method="animate",
                    ),
                    dict(
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        label="Pausar",
                        method="animate",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=1.1,
                yanchor="top",
                bgcolor="rgba(20, 45, 78, 0.5)",
                bordercolor="rgba(255, 255, 255, 0.1)",
                font=dict(color="white"),
            )
        ],
    )

    # Add frames for animation
    frames = []
    for i in range(1, len(timestamps) + 1):
        frame_data = [
            go.Scatter(
                x=timestamps[:i],
                y=overall_scores[:i],
                mode="lines+markers",
                line=dict(color="#45C4AF", width=4, shape="spline"),
                marker=dict(
                    size=10, color="#FFEB85", line=dict(color="#45C4AF", width=2)
                ),
                fill="tozeroy",
                fillcolor="rgba(69, 196, 175, 0.1)",
            )
        ]

        if "dimensions" in metrics_history[0]:
            for dim_key, dim_info in METRIC_DIMENSIONS.items():
                if dim_key in metrics_history[0]["dimensions"]:
                    dim_scores = [
                        entry["dimensions"][dim_key]["score"]
                        for entry in metrics_history
                    ]

                    frame_data.append(
                        go.Scatter(
                            x=timestamps[:i],
                            y=dim_scores[:i],
                            mode="lines+markers",
                            line=dict(
                                color=dim_info["color"],
                                width=2,
                                dash="dot",
                                shape="spline",
                            ),
                            marker=dict(size=6, color=dim_info["color"], opacity=0.7),
                        )
                    )

        frames.append(go.Frame(data=frame_data, name=f"frame{i}"))

    fig.frames = frames

    return fig


def create_3d_metrics_visualization(metrics: Dict[str, Any]):
    """
    Create an interactive 3D visualization of the linguistic metrics relationships.

    Args:
        metrics: Dictionary containing the metrics results

    Returns:
        plotly.graph_objects.Figure: 3D visualization figure
    """
    # Extract all individual metrics from all dimensions
    all_metrics = []
    for dim_key, dim_metrics in metrics["dimensions"].items():
        for metric_key, metric_info in dim_metrics.items():
            if isinstance(metric_info, dict) and "score" in metric_info:
                all_metrics.append(
                    {
                        "dimension": dim_key,
                        "dimension_name": METRIC_DIMENSIONS[dim_key]["name"],
                        "metric": metric_key,
                        "name": metric_info.get("name", metric_key),
                        "score": metric_info["score"],
                        "description": metric_info.get("description", ""),
                        "color": METRIC_DIMENSIONS[dim_key]["color"],
                    }
                )

    # Sort metrics by score
    all_metrics.sort(key=lambda x: x["score"])

    # Create 3D figure with custom layout
    fig = go.Figure()

    # Calculate positions in 3D space using a spherical arrangement
    metrics_count = len(all_metrics)
    phi = np.linspace(0, np.pi, int(np.ceil(metrics_count / 8)))
    theta = np.linspace(0, 2 * np.pi, 8)

    phi_grid, theta_grid = np.meshgrid(phi, theta)
    phi_flat = phi_grid.flatten()[:metrics_count]
    theta_flat = theta_grid.flatten()[:metrics_count]

    # Calculate 3D coordinates on a sphere
    radius = 100
    x = radius * np.sin(phi_flat) * np.cos(theta_flat)
    y = radius * np.sin(phi_flat) * np.sin(theta_flat)
    z = radius * np.cos(phi_flat)

    # Scale score values for visual representation
    sizes = [30 + (m["score"] / 2) for m in all_metrics]
    colors = [m["color"] for m in all_metrics]

    # Add markers for each metric
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.7,
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
            ),
            text=[
                f"{m['name']}<br>Score: {m['score']:.1f}<br>{m['dimension_name']}"
                for m in all_metrics
            ],
            hoverinfo="text",
            name="Métricas Individuais",
        )
    )

    # Add lines connecting metrics of the same dimension
    for dim_key in METRIC_DIMENSIONS:
        dim_metrics = [m for m in all_metrics if m["dimension"] == dim_key]
        if len(dim_metrics) > 1:
            indices = [all_metrics.index(m) for m in dim_metrics]
            x_dim = [x[i] for i in indices]
            y_dim = [y[i] for i in indices]
            z_dim = [z[i] for i in indices]

            # Create a centroid for the dimension
            x_center = np.mean(x_dim)
            y_center = np.mean(y_dim)
            z_center = np.mean(z_dim)

            # Add lines from centroid to each metric in the dimension
            for i in range(len(dim_metrics)):
                fig.add_trace(
                    go.Scatter3d(
                        x=[x_center, x_dim[i]],
                        y=[y_center, y_dim[i]],
                        z=[z_center, z_dim[i]],
                        mode="lines",
                        line=dict(color=METRIC_DIMENSIONS[dim_key]["color"], width=2),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

            # Add a central node for the dimension
            fig.add_trace(
                go.Scatter3d(
                    x=[x_center],
                    y=[y_center],
                    z=[z_center],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color=METRIC_DIMENSIONS[dim_key]["color"],
                        symbol="diamond",
                        opacity=0.8,
                        line=dict(color="rgba(255, 255, 255, 0.5)", width=1),
                    ),
                    text=[METRIC_DIMENSIONS[dim_key]["name"]],
                    hoverinfo="text",
                    name=METRIC_DIMENSIONS[dim_key]["name"],
                )
            )

    # Update layout
    fig.update_layout(
        title={
            "text": "Rede de Relacionamento de Métricas 3D",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"color": "white", "size": 24},
        },
        scene=dict(
            xaxis=dict(
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False,
            ),
            yaxis=dict(
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False,
            ),
            zaxis=dict(
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False,
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5), projection=dict(type="perspective")
            ),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(20, 45, 78, 0.5)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
        ),
        autosize=True,
        height=700,
        hovermode="closest",
        # Add scene annotations explaining the visualization
        annotations=[
            dict(
                showarrow=False,
                x=0,
                y=0,
                z=0,
                text="Rede de relacionamento das métricas textuais.<br>Métricas da mesma dimensão estão conectadas.",
                xanchor="left",
                xshift=10,
                opacity=0.7,
                font=dict(color="white", size=12),
            )
        ],
    )

    # Add slider for interactive exploration
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Rotação: ", "font": {"color": "white"}},
                "pad": {"t": 50},
                "len": 0.5,
                "x": 0.25,
                "xanchor": "center",
                "y": 0,
                "yanchor": "bottom",
                "steps": [
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "scene.camera.eye": {
                                    "x": 1.5 * np.cos(angle),
                                    "y": 1.5 * np.sin(angle),
                                    "z": 1.5,
                                },
                            },
                        ],
                        "label": f"{int(angle * 180 / np.pi)}°",
                        "method": "animate",
                    }
                    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False)
                ],
            }
        ]
    )

    return fig


def create_pydeck_3d_map(metrics: Dict[str, Any]):
    """
    Create a 3D geographic visualization using PyDeck.

    Args:
        metrics: Dictionary containing the metrics results

    Returns:
        pydeck.Deck: PyDeck 3D visualization
    """
    # Calculate positions in a hemisphere
    dimension_scores = {}
    for dim_key, dim_metrics in metrics["dimensions"].items():
        if isinstance(dim_metrics, dict) and "score" in dim_metrics:
            dimension_scores[dim_key] = dim_metrics["score"]
        else:
            dimension_scores[dim_key] = np.mean(
                [
                    m["score"]
                    for m in dim_metrics.values()
                    if isinstance(m, dict) and "score" in m
                ]
            )

    # Generate points on a hemisphere
    dimensions = list(dimension_scores.keys())

    # Generate equally spaced points on a hemisphere
    # Convert dimension scores to colors with our theme
    def score_to_color(score):
        # Green for high, yellow for medium, red for low
        if score >= 80:
            return [69, 196, 175, 200]  # Primary green
        elif score >= 60:
            return [255, 235, 133, 200]  # Yellow
        else:
            return [255, 127, 127, 200]  # Red

    cols = 3

    hemisphere_data = []
    for i, dim_key in enumerate(dimensions):
        score = dimension_scores[dim_key]
        dim_name = METRIC_DIMENSIONS[dim_key]["name"]
        dim_color = METRIC_DIMENSIONS[dim_key]["color"]

        # Calculate grid position
        row = i // cols
        col = i % cols

        # Base position
        lon = -98 + col * 15  # Spread columns horizontally
        lat = 40 - row * 15  # Spread rows vertically

        # Create a hemisphere of points for each dimension
        n_points = 50
        radius = score / 10  # Scale radius by score

        for j in range(n_points):
            # Random position on hemisphere
            theta = np.random.random() * 2 * np.pi
            phi = np.random.random() * np.pi / 2

            # Convert to Cartesian
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)

            # Add random variation for more natural look
            x_var = np.random.normal(0, 0.5)
            y_var = np.random.normal(0, 0.5)

            # Add to base position
            point_lon = lon + x + x_var
            point_lat = lat + y + y_var

            # Add point
            point_color = score_to_color(score)

            hemisphere_data.append(
                {
                    "position": [point_lon, point_lat, z],
                    "color": point_color,
                    "dimension": dim_name,
                    "score": score,
                    "radius": 0.5 + (0.5 * score / 100),  # Vary point size by score
                }
            )

        # Add dimension center marker
        hemisphere_data.append(
            {
                "position": [lon, lat, 0],
                "color": [
                    int(dim_color[1:3], 16),
                    int(dim_color[3:5], 16),
                    int(dim_color[5:7], 16),
                    255,
                ],
                "dimension": dim_name,
                "score": score,
                "radius": 3.0,  # Larger marker for dimension center
                "is_center": True,
            }
        )

    # Create point cloud layer
    point_cloud_layer = pdk.Layer(
        "PointCloudLayer",
        hemisphere_data,
        get_position="position",
        get_color="color",
        get_normal=[0, 0, 1],
        auto_highlight=True,
        pickable=True,
        point_size="radius",
        stroked=False,
        filled=True,
        opacity=0.8,
    )

    # Create text layer for dimension labels
    text_data = [data for data in hemisphere_data if data.get("is_center", False)]
    text_layer = pdk.Layer(
        "TextLayer",
        text_data,
        get_position="position",
        get_text="dimension",
        get_size=18,
        get_color=[255, 255, 255, 255],
        get_angle=0,
        get_text_anchor="middle",
        get_alignment_baseline="center",
    )

    # Initial view state
    view_state = pdk.ViewState(
        longitude=-98,
        latitude=40,
        zoom=2.5,
        min_zoom=1,
        max_zoom=10,
        pitch=45,
        bearing=0,
    )

    # Create tooltip
    tooltip = {
        "html": "<b>{dimension}</b><br/><span>Score: {score}</span>",
        "style": {
            "backgroundColor": "rgba(20, 45, 78, 0.9)",
            "color": "white",
            "border": "1px solid rgba(255, 255, 255, 0.2)",
            "borderRadius": "4px",
            "padding": "8px",
            "fontFamily": "Arial, sans-serif",
        },
    }

    # Create deck
    deck = pdk.Deck(
        layers=[point_cloud_layer, text_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider=None,
        map_style=None,
    )

    return deck


def create_comparison_heatmap(
    texts_metrics: List[Dict[str, Any]], text_names: List[str]
):
    """
    Create a heatmap for comparing multiple texts across dimensions.

    Args:
        texts_metrics: List of metrics dictionaries for multiple texts
        text_names: Names or labels for each text

    Returns:
        plotly.graph_objects.Figure: Heatmap figure
    """
    # Extract dimension scores for each text
    data = []

    for metrics, name in zip(texts_metrics, text_names):
        dim_scores = {}
        for dim_key, dim_data in metrics.get("dimensions", {}).items():
            if isinstance(dim_data, dict) and "score" in dim_data:
                score = dim_data["score"]
            else:
                # Calculate average if the dimension contains multiple metrics
                scores = [
                    m.get("score", 0) for m in dim_data.values() if isinstance(m, dict)
                ]
                score = sum(scores) / len(scores) if scores else 0

            # Get dimension name from METRIC_DIMENSIONS
            dim_name = METRIC_DIMENSIONS.get(dim_key, {}).get("name", dim_key)
            dim_scores[dim_name] = score

        # Add to data list
        data.append(dim_scores)

    # Create DataFrame for the heatmap
    dimensions = [METRIC_DIMENSIONS[dim]["name"] for dim in METRIC_DIMENSIONS]
    df = pd.DataFrame(data, index=text_names)

    # Ensure all dimensions are included, even if not present in all texts
    for dim in dimensions:
        if dim not in df.columns:
            df[dim] = np.nan

    # Keep only the dimensions we have defined
    df = df[dimensions]

    # Create the heatmap
    fig = px.imshow(
        df,
        labels=dict(x="Dimensão", y="Texto", color="Pontuação"),
        x=dimensions,
        y=text_names,
        color_continuous_scale="Blues",
        zmin=0,
        zmax=100,
        aspect="auto",
    )

    # Add text annotations
    for i in range(len(text_names)):
        for j in range(len(dimensions)):
            value = df.iloc[i, j]
            if not pd.isna(value):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{value:.1f}",
                    showarrow=False,
                    font=dict(color="white" if value > 50 else "black"),
                )

    # Update layout
    fig.update_layout(
        title={
            "text": "Comparação de Textos por Dimensão",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"color": "white", "size": 20},
        },
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=40, r=40, t=80, b=40),
        coloraxis_colorbar=dict(
            title="Pontuação",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            thicknessmode="pixels",
            thickness=20,
            lenmode="pixels",
            len=300,
        ),
        xaxis=dict(tickfont=dict(color="white"), tickangle=-45),
        yaxis=dict(tickfont=dict(color="white")),
    )

    return fig


def create_text_heatmap(doc, metrics: Dict[str, Any]):
    """
    Create a heatmap visualization of the text by sentences, highlighting areas needing improvement.

    Args:
        doc: spaCy document containing the processed text
        metrics: Dictionary containing the metrics results

    Returns:
        plotly.graph_objects.Figure: Text heatmap figure
    """
    # Get sentences from the document
    sentences = [sent.text for sent in doc.sents]

    # Limit to a maximum of 50 sentences for better visualization
    if len(sentences) > 50:
        sentences = sentences[:50]

    # Find the 3 dimensions with lowest scores
    dim_scores = {}
    for dim_key, dim_data in metrics.get("dimensions", {}).items():
        if isinstance(dim_data, dict) and "score" in dim_data:
            score = dim_data["score"]
        else:
            # Calculate average if the dimension contains multiple metrics
            scores = [
                m.get("score", 0) for m in dim_data.values() if isinstance(m, dict)
            ]
            score = sum(scores) / len(scores) if scores else 0

        dim_scores[dim_key] = score

    # Sort dimensions by score and get the 3 lowest
    problem_dimensions = sorted(dim_scores.items(), key=lambda x: x[1])[:3]
    problem_dim_keys = [d[0] for d in problem_dimensions]

    # Simulate sentence-level evaluations for the problem dimensions
    # In a real system, this would use actual sentence-level metrics
    sentence_scores = []

    for i, sent in enumerate(sentences):
        # Create a score for each problem dimension
        # This is a simulation - real implementation would use actual metrics
        sent_scores = {}
        for dim_key in problem_dim_keys:
            # Base the score on the overall dimension score
            base_score = dim_scores[dim_key]

            # Add some variation based on sentence index (just for demonstration)
            variation = np.sin(i * 0.5) * 15

            # Ensure the score stays within 0-100 range
            score = max(0, min(100, base_score + variation))
            sent_scores[dim_key] = score

        sentence_scores.append(sent_scores)

    # Create a DataFrame for the heatmap
    df_data = []
    for i, sent_scores in enumerate(sentence_scores):
        for dim_key, score in sent_scores.items():
            dim_name = METRIC_DIMENSIONS[dim_key]["name"]
            df_data.append(
                {"Sentença": f"S{i+1}", "Dimensão": dim_name, "Pontuação": score}
            )

    df = pd.DataFrame(df_data)

    # Pivot the DataFrame for the heatmap
    df_pivot = df.pivot(index="Sentença", columns="Dimensão", values="Pontuação")

    # Create the heatmap
    fig = px.imshow(
        df_pivot,
        labels=dict(x="Dimensão", y="Sentença", color="Pontuação"),
        color_continuous_scale=[
            [0, "rgba(255, 127, 127, 0.8)"],  # Red for low scores
            [0.4, "rgba(255, 235, 133, 0.8)"],  # Yellow for medium scores
            [0.7, "rgba(151, 221, 212, 0.8)"],  # Light blue-green for good scores
            [1, "rgba(69, 196, 175, 0.8)"],  # Blue-green for high scores
        ],
        zmin=0,
        zmax=100,
        aspect="auto",
    )

    # Add text annotations for scores
    for i in range(len(df_pivot.index)):
        for j in range(len(df_pivot.columns)):
            try:
                value = df_pivot.iloc[i, j]
                if not pd.isna(value):
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=f"{value:.0f}",
                        showarrow=False,
                        font=dict(color="white" if value < 70 else "black", size=9),
                    )
            except IndexError:
                # Skip if indices are out of range
                continue

    # Update layout
    fig.update_layout(
        title={
            "text": "Análise de Qualidade por Sentença",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"color": "white", "size": 20},
        },
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=40, r=40, t=80, b=40),
        coloraxis_colorbar=dict(
            title="Pontuação",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            thicknessmode="pixels",
            thickness=20,
            lenmode="pixels",
            len=300,
        ),
        xaxis=dict(tickfont=dict(color="white")),
        yaxis=dict(tickfont=dict(color="white"), title="Índice da Sentença"),
        height=max(
            400, min(800, len(sentences) * 15 + 200)
        ),  # Dynamic height based on number of sentences
    )

    # Add sentence tooltip information
    for i, sent in enumerate(sentences):
        # Truncate sentences for hover display
        display_sent = sent[:100] + "..." if len(sent) > 100 else sent
        row = df_pivot.index[i] if i < len(df_pivot.index) else None

        if row:
            # Add hover information for each sentence
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=0),
                    hoverinfo="text",
                    text=f"<b>Sentença {i+1}:</b><br>{display_sent}",
                    showlegend=False,
                )
            )

    # Add explanation annotation
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text="Esta visualização mostra as pontuações para as três dimensões mais problemáticas identificadas no texto.<br>Vermelho indica áreas que precisam de melhorias, verde indica pontos fortes.",
        showarrow=False,
        font=dict(color="white", size=12),
        align="center",
        bgcolor="rgba(20, 45, 78, 0.7)",
        bordercolor="rgba(69, 196, 175, 0.2)",
        borderwidth=1,
        borderpad=10,
        opacity=0.8,
    )

    return fig


def create_interactive_metric_explorer(metrics: Dict[str, Any]):
    """
    Create an interactive metric explorer using streamlit_elements.

    Args:
        metrics: Dictionary containing the metrics results
    """
    # Prepare data for the interactive dashboard
    dimension_data = []
    dimension_scores = {}

    # Get dimension scores
    for dim_key, dim_metrics in metrics["dimensions"].items():
        if isinstance(dim_metrics, dict) and "score" in dim_metrics:
            dim_score = dim_metrics["score"]
        else:
            dim_score = np.mean(
                [
                    m["score"]
                    for m in dim_metrics.values()
                    if isinstance(m, dict) and "score" in m
                ]
            )

        dimension_scores[dim_key] = dim_score

        dimension_data.append(
            {
                "id": dim_key,
                "label": METRIC_DIMENSIONS[dim_key]["name"],
                "value": dim_score,
                "color": METRIC_DIMENSIONS[dim_key]["color"],
            }
        )

    # Create metrics data
    metrics_data = []
    for dim_key, dim_metrics in metrics["dimensions"].items():
        for metric_key, metric_info in dim_metrics.items():
            if isinstance(metric_info, dict) and "score" in metric_info:
                metrics_data.append(
                    {
                        "dimension": dim_key,
                        "dimension_name": METRIC_DIMENSIONS[dim_key]["name"],
                        "id": f"{dim_key}_{metric_key}",
                        "metric": metric_key,
                        "label": metric_info.get("name", metric_key),
                        "value": metric_info["score"],
                        "description": metric_info.get("description", ""),
                        "color": METRIC_DIMENSIONS[dim_key]["color"],
                    }
                )

    with elements("metric_explorer"):
        with dashboard.Grid(draggableHandle=".draggable"):
            # First row of dashboard - pie chart of dimensions
            with dashboard.Item(
                "dimensions_chart", x=0, y=0, w=6, h=10, dragHandle=".draggable"
            ):
                mui.Paper(
                    elevation=3,
                    sx={
                        "display": "flex",
                        "flexDirection": "column",
                        "borderRadius": 3,
                        "overflow": "hidden",
                        "background": "rgba(20, 45, 78, 0.7)",
                        "backdropFilter": "blur(10px)",
                        "border": "1px solid rgba(103, 193, 185, 0.2)",
                    },
                    children=[
                        mui.Box(
                            className="draggable",
                            sx={
                                "p": 2,
                                "display": "flex",
                                "alignItems": "center",
                                "borderBottom": "1px solid rgba(103, 193, 185, 0.2)",
                            },
                            children=[
                                mui.Typography(
                                    variant="h6",
                                    sx={"color": "white"},
                                    children="Distribuição de Dimensões",
                                )
                            ],
                        ),
                        mui.Box(
                            sx={
                                "p": 2,
                                "height": "calc(100% - 60px)",
                                "background": "rgba(20, 45, 78, 0.5)",
                            },
                            children=[
                                nivo.Pie(
                                    data=dimension_data,
                                    margin={
                                        "top": 40,
                                        "right": 80,
                                        "bottom": 80,
                                        "left": 80,
                                    },
                                    innerRadius={0.5},
                                    padAngle={0.7},
                                    cornerRadius={3},
                                    activeOuterRadiusOffset={8},
                                    colors={"scheme": "category10"},
                                    borderWidth={1},
                                    borderColor={
                                        "from": "color",
                                        "modifiers": [["darker", 0.2]],
                                    },
                                    arcLinkLabelsSkipAngle={10},
                                    arcLinkLabelsTextColor={
                                        "r": 255,
                                        "g": 255,
                                        "b": 255,
                                    },
                                    arcLinkLabelsThickness={2},
                                    arcLinkLabelsColor={"from": "color"},
                                    arcLabelsSkipAngle={10},
                                    arcLabelsTextColor={
                                        "from": "color",
                                        "modifiers": [["darker", 2]],
                                    },
                                    legends=[
                                        {
                                            "anchor": "bottom",
                                            "direction": "row",
                                            "justify": False,
                                            "translateX": 0,
                                            "translateY": 56,
                                            "itemsSpacing": 0,
                                            "itemWidth": 100,
                                            "itemHeight": 18,
                                            "itemTextColor": "#fff",
                                            "itemDirection": "left-to-right",
                                            "itemOpacity": 1,
                                            "symbolSize": 18,
                                            "symbolShape": "circle",
                                        }
                                    ],
                                )
                            ],
                        ),
                    ],
                )

            # Second row - bar chart of all metrics
            with dashboard.Item(
                "metrics_chart", x=6, y=0, w=6, h=10, dragHandle=".draggable"
            ):
                mui.Paper(
                    elevation=3,
                    sx={
                        "display": "flex",
                        "flexDirection": "column",
                        "borderRadius": 3,
                        "overflow": "hidden",
                        "background": "rgba(20, 45, 78, 0.7)",
                        "backdropFilter": "blur(10px)",
                        "border": "1px solid rgba(103, 193, 185, 0.2)",
                    },
                    children=[
                        mui.Box(
                            className="draggable",
                            sx={
                                "p": 2,
                                "display": "flex",
                                "alignItems": "center",
                                "borderBottom": "1px solid rgba(103, 193, 185, 0.2)",
                            },
                            children=[
                                mui.Typography(
                                    variant="h6",
                                    sx={"color": "white"},
                                    children="Pontuação por Métrica",
                                )
                            ],
                        ),
                        mui.Box(
                            sx={
                                "p": 2,
                                "height": "calc(100% - 60px)",
                                "background": "rgba(20, 45, 78, 0.5)",
                            },
                            children=[
                                nivo.Bar(
                                    data=[
                                        {
                                            "metric": m["label"],
                                            "score": m["value"],
                                            "scoreColor": m["color"],
                                        }
                                        for m in metrics_data
                                    ],
                                    keys=["score"],
                                    indexBy="metric",
                                    margin={
                                        "top": 50,
                                        "right": 50,
                                        "bottom": 80,
                                        "left": 120,
                                    },
                                    padding={0.3},
                                    valueScale={"type": "linear"},
                                    indexScale={"type": "band", "round": True},
                                    colors={"scheme": "nivo"},
                                    colorBy="indexValue",
                                    borderColor={
                                        "from": "color",
                                        "modifiers": [["darker", 1.6]],
                                    },
                                    axisTop=None,
                                    axisRight=None,
                                    axisBottom={
                                        "tickSize": 5,
                                        "tickPadding": 5,
                                        "tickRotation": 0,
                                        "legend": "Pontuação",
                                        "legendPosition": "middle",
                                        "legendOffset": 50,
                                        "truncateTickAt": 0,
                                    },
                                    axisLeft={
                                        "tickSize": 5,
                                        "tickPadding": 5,
                                        "tickRotation": 0,
                                        "legend": "Métrica",
                                        "legendPosition": "middle",
                                        "legendOffset": -100,
                                        "truncateTickAt": 0,
                                    },
                                    labelSkipWidth={12},
                                    labelSkipHeight={12},
                                    labelTextColor={
                                        "from": "color",
                                        "modifiers": [["darker", 1.6]],
                                    },
                                    legends=[
                                        {
                                            "dataFrom": "keys",
                                            "anchor": "bottom",
                                            "direction": "row",
                                            "justify": False,
                                            "translateX": 0,
                                            "translateY": 65,
                                            "itemsSpacing": 2,
                                            "itemWidth": 100,
                                            "itemHeight": 20,
                                            "itemDirection": "left-to-right",
                                            "itemOpacity": 0.85,
                                            "symbolSize": 20,
                                            "effects": [
                                                {
                                                    "on": "hover",
                                                    "style": {"itemOpacity": 1},
                                                }
                                            ],
                                        }
                                    ],
                                    role="application",
                                    ariaLabel="Métricas de qualidade",
                                    barAriaLabel={
                                        "enabled": True,
                                        "valueFormat": " >-",
                                    },
                                )
                            ],
                        ),
                    ],
                )

            # Third row - radar chart
            with dashboard.Item(
                "radar_chart", x=0, y=10, w=12, h=10, dragHandle=".draggable"
            ):
                mui.Paper(
                    elevation=3,
                    sx={
                        "display": "flex",
                        "flexDirection": "column",
                        "borderRadius": 3,
                        "overflow": "hidden",
                        "background": "rgba(20, 45, 78, 0.7)",
                        "backdropFilter": "blur(10px)",
                        "border": "1px solid rgba(103, 193, 185, 0.2)",
                    },
                    children=[
                        mui.Box(
                            className="draggable",
                            sx={
                                "p": 2,
                                "display": "flex",
                                "alignItems": "center",
                                "borderBottom": "1px solid rgba(103, 193, 185, 0.2)",
                            },
                            children=[
                                mui.Typography(
                                    variant="h6",
                                    sx={"color": "white"},
                                    children="Radar de Métricas",
                                )
                            ],
                        ),
                        mui.Box(
                            sx={
                                "p": 2,
                                "height": "calc(100% - 60px)",
                                "background": "rgba(20, 45, 78, 0.5)",
                            },
                            children=[
                                nivo.Radar(
                                    data=[
                                        {
                                            "dimension": METRIC_DIMENSIONS[dim_key][
                                                "name"
                                            ],
                                            "score": dimension_scores[dim_key],
                                            "color": METRIC_DIMENSIONS[dim_key][
                                                "color"
                                            ],
                                        }
                                        for dim_key in dimension_scores
                                    ],
                                    keys=["score"],
                                    indexBy="dimension",
                                    valueFormat=">-.2f",
                                    margin={
                                        "top": 70,
                                        "right": 80,
                                        "bottom": 40,
                                        "left": 80,
                                    },
                                    borderColor={"from": "color"},
                                    gridLabelOffset={36},
                                    dotSize={10},
                                    dotColor={"theme": "background"},
                                    dotBorderWidth={2},
                                    colors={"scheme": "nivo"},
                                    blendMode="multiply",
                                    motionConfig="wobbly",
                                    legends=[
                                        {
                                            "anchor": "top-left",
                                            "direction": "column",
                                            "translateX": -50,
                                            "translateY": -40,
                                            "itemWidth": 80,
                                            "itemHeight": 20,
                                            "itemTextColor": "#fff",
                                            "symbolSize": 12,
                                            "symbolShape": "circle",
                                            "effects": [
                                                {
                                                    "on": "hover",
                                                    "style": {"itemTextColor": "#000"},
                                                }
                                            ],
                                        }
                                    ],
                                )
                            ],
                        ),
                    ],
                )
