"""TensorFlow Memory Visualization"""

import csv
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, cast

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import dash
    import plotly.graph_objects as go
    from dash import dcc, html

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import numpy as np


class MemoryVisualizer:
    """TensorFlow memory visualization and dashboards."""

    def __init__(
        self, style: str = "default", figure_size: Tuple[int, int] = (12, 8)
    ) -> None:
        self.style = style
        self.figure_size = figure_size

        if MATPLOTLIB_AVAILABLE and style != "default":
            try:
                plt.style.use(style)
            except Exception as exc:
                logging.debug("Could not apply matplotlib style %r: %s", style, exc)

    def plot_memory_timeline(
        self, results: Any, interactive: bool = False, save_path: Optional[str] = None
    ) -> None:
        """Plot memory usage timeline."""
        if not hasattr(results, "snapshots") or not results.snapshots:
            logging.warning("No snapshots available for plotting")
            return

        timestamps = [s.timestamp for s in results.snapshots]
        memory_usage = [s.gpu_memory_mb for s in results.snapshots]

        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory_usage,
                    mode="lines+markers",
                    name="GPU Memory Usage",
                    line=dict(color="blue", width=2),
                )
            )

            fig.update_layout(
                title="TensorFlow GPU Memory Usage Timeline",
                xaxis_title="Time",
                yaxis_title="Memory Usage (MB)",
                hovermode="x unified",
            )

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()

        elif MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figure_size)
            plt.plot(timestamps, memory_usage, "b-", linewidth=2, label="GPU Memory")
            plt.title("TensorFlow GPU Memory Usage Timeline")
            plt.xlabel("Time")
            plt.ylabel("Memory Usage (MB)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()
        else:
            logging.error("No plotting libraries available")

    def plot_function_comparison(
        self,
        function_profiles: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot function memory usage comparison."""
        if not function_profiles:
            logging.warning("No function profiles available")
            return

        functions = list(function_profiles.keys())
        peak_memories = [
            profile.get("peak_memory", 0) for profile in function_profiles.values()
        ]

        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figure_size)
            bars = plt.bar(functions, peak_memories, color="skyblue", alpha=0.7)
            plt.title("Function Memory Usage Comparison")
            plt.xlabel("Functions")
            plt.ylabel("Peak Memory (MB)")
            plt.xticks(rotation=45, ha="right")

            # Add value labels on bars
            for bar, memory in zip(bars, peak_memories):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 10,
                    f"{memory:.1f}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()

    def create_memory_heatmap(
        self, results: Any, save_path: Optional[str] = None
    ) -> None:
        """Create memory usage heatmap."""
        if not hasattr(results, "snapshots") or len(results.snapshots) < 10:
            logging.warning("Insufficient data for heatmap")
            return

        # Create time-based heatmap data
        _timestamps = [s.timestamp for s in results.snapshots]
        memory_data = [s.gpu_memory_mb for s in results.snapshots]

        # Reshape data for heatmap
        chunk_size = 10
        chunks = [
            memory_data[i : i + chunk_size]
            for i in range(0, len(memory_data), chunk_size)
        ]

        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figure_size)

            # Pad chunks to same length
            max_len = max(len(chunk) for chunk in chunks)
            padded_chunks = [chunk + [0] * (max_len - len(chunk)) for chunk in chunks]

            heatmap_data = np.array(padded_chunks)

            if "seaborn" in str(self.style).lower() and "sns" in globals():
                sns.heatmap(
                    heatmap_data, cmap="viridis", cbar_kws={"label": "Memory (MB)"}
                )
            else:
                plt.imshow(heatmap_data, cmap="viridis", aspect="auto")
                plt.colorbar(label="Memory (MB)")

            plt.title("Memory Usage Heatmap")
            plt.xlabel("Time Chunks")
            plt.ylabel("Sample Groups")

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()

    def create_interactive_dashboard(self, results: Any, port: int = 8050) -> None:
        """Create interactive Plotly dashboard."""
        if not PLOTLY_AVAILABLE:
            logging.error("Plotly/Dash not available for interactive dashboard")
            return

        app = dash.Dash(__name__)

        # Prepare data
        if hasattr(results, "snapshots") and results.snapshots:
            timestamps = [s.timestamp for s in results.snapshots]
            memory_usage = [s.gpu_memory_mb for s in results.snapshots]
        else:
            timestamps = []
            memory_usage = []

        # Create plots
        timeline_fig = go.Figure()
        timeline_fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=memory_usage,
                mode="lines+markers",
                name="GPU Memory",
                line=dict(color="blue", width=2),
            )
        )
        timeline_fig.update_layout(title="Memory Timeline")

        # Function comparison
        if hasattr(results, "function_profiles") and results.function_profiles:
            functions = list(results.function_profiles.keys())
            peak_memories = [
                profile.get("peak_memory", 0)
                for profile in results.function_profiles.values()
            ]

            comparison_fig = go.Figure(
                data=[go.Bar(x=functions, y=peak_memories, name="Peak Memory")]
            )
            comparison_fig.update_layout(title="Function Memory Comparison")
        else:
            comparison_fig = go.Figure()

        # Dashboard layout
        app.layout = html.Div(
            [
                html.H1("TensorFlow Memory Profiler Dashboard"),
                html.Div(
                    [
                        html.H3("Memory Statistics"),
                        html.P(
                            f"Peak Memory: {getattr(results, 'peak_memory_mb', 0):.2f} MB"
                        ),
                        html.P(
                            f"Average Memory: {getattr(results, 'average_memory_mb', 0):.2f} MB"
                        ),
                        html.P(
                            f"Total Allocations: {getattr(results, 'total_allocations', 0)}"
                        ),
                    ],
                    style={"margin": "20px"},
                ),
                dcc.Graph(figure=timeline_fig),
                dcc.Graph(figure=comparison_fig),
            ]
        )

        try:
            app.run_server(debug=False, port=port, host="127.0.0.1")
        except Exception as e:
            logging.error(f"Could not start dashboard: {e}")

    def export_data(self, results: Any, output_path: str, format: str = "csv") -> None:
        """Export profiling data."""
        if format.lower() == "csv":
            self._export_csv(results, output_path)
        elif format.lower() == "json":
            self._export_json(results, output_path)
        else:
            logging.error(f"Unsupported export format: {format}")

    def _export_csv(self, results: Any, output_path: str) -> None:
        """Export data to CSV."""
        if not hasattr(results, "snapshots") or not results.snapshots:
            logging.warning("No data to export")
            return

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "name", "gpu_memory_mb", "cpu_memory_mb", "num_tensors"]
            )

            for snapshot in results.snapshots:
                writer.writerow(
                    [
                        snapshot.timestamp,
                        snapshot.name,
                        snapshot.gpu_memory_mb,
                        snapshot.cpu_memory_mb,
                        snapshot.num_tensors,
                    ]
                )

        logging.info(f"Data exported to {output_path}")

    def _export_json(self, results: Any, output_path: str) -> None:
        """Export data to JSON."""
        data: Dict[str, Any] = {
            "peak_memory_mb": getattr(results, "peak_memory_mb", 0),
            "average_memory_mb": getattr(results, "average_memory_mb", 0),
            "total_allocations": getattr(results, "total_allocations", 0),
            "snapshots": [],
        }
        snapshots_data = cast(List[Dict[str, Any]], data["snapshots"])

        if hasattr(results, "snapshots"):
            for snapshot in results.snapshots:
                snapshots_data.append(
                    {
                        "timestamp": snapshot.timestamp,
                        "name": snapshot.name,
                        "gpu_memory_mb": snapshot.gpu_memory_mb,
                        "cpu_memory_mb": snapshot.cpu_memory_mb,
                        "num_tensors": snapshot.num_tensors,
                    }
                )

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logging.info(f"Data exported to {output_path}")

    def save_plots(self, results: Any, output_dir: str = "./plots/") -> None:
        """Save all plots to directory."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        self.plot_memory_timeline(results, save_path=f"{output_dir}/timeline.png")

        if hasattr(results, "function_profiles"):
            self.plot_function_comparison(
                results.function_profiles,
                save_path=f"{output_dir}/function_comparison.png",
            )

        self.create_memory_heatmap(results, save_path=f"{output_dir}/heatmap.png")

        logging.info(f"Plots saved to {output_dir}")
