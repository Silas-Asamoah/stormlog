"""JAX Memory Visualization."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

plt: Any
try:
    import matplotlib.pyplot as _plt

    plt = _plt
    MATPLOTLIB_AVAILABLE = True
    try:
        import seaborn as sns
    except ImportError:
        sns = None
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

jax: Any
try:
    import jax as _jax  # noqa: F401

    jax = _jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None

try:
    import dash
    import plotly.graph_objects as go
    from dash import dcc, html

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class MemoryVisualizer:
    """JAX memory visualization and dashboards."""

    def __init__(
        self, style: str = "default", figure_size: Tuple[int, int] = (12, 8)
    ) -> None:
        self.style = style
        self.figure_size = figure_size

        if MATPLOTLIB_AVAILABLE and style != "default":
            try:
                plt.style.use(style)
            except Exception:
                pass

    def plot_memory_timeline(
        self, results: Any, interactive: bool = False, save_path: Optional[str] = None
    ) -> None:
        """Plot device memory usage timeline."""
        timeline = _timeline_samples(results)
        if timeline is None:
            return
        timestamps, memory_usage = timeline

        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory_usage,
                    mode="lines+markers",
                    name="Device Memory",
                    line=dict(color="crimson", width=2),
                )
            )

            fig.update_layout(
                title="Device Memory Usage Timeline",
                xaxis_title="Time",
                yaxis_title="Memory Usage (MB)",
                template="plotly_dark" if "dark" in self.style else "plotly",
            )

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()

        elif MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figure_size)
            plt.plot(
                timestamps,
                memory_usage,
                color="crimson",
                linewidth=2,
                label="Device Memory",
            )
            plt.title("Device Memory Usage Timeline")
            plt.xlabel("Time")
            plt.ylabel("Memory Usage (MB)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()

    def plot_function_comparison(
        self,
        function_profiles: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot memory usage comparison for functions/contexts."""
        if not function_profiles:
            return

        functions = list(function_profiles.keys())
        peak_memories = [
            profile.get("peak_memory_bytes", 0) / (1024 * 1024)
            for profile in function_profiles.values()
        ]

        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figure_size)
            plt.bar(functions, peak_memories, color="salmon", alpha=0.8)
            plt.title("Function Memory Comparison")
            plt.ylabel("Peak Memory (MB)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()

    def create_memory_heatmap(
        self, results: Any, save_path: Optional[str] = None
    ) -> None:
        """Create a heatmap from available JAX device-memory samples."""
        if not MATPLOTLIB_AVAILABLE:
            logging.error("Matplotlib is required for heatmaps")
            return
        samples = _heatmap_samples(results)
        if len(samples) < 10:
            logging.warning("Insufficient device-memory samples for heatmap")
            return
        chunks = [samples[index : index + 10] for index in range(0, len(samples), 10)]
        width = max(len(chunk) for chunk in chunks)
        padded = [chunk + [0.0] * (width - len(chunk)) for chunk in chunks]
        plt.figure(figsize=self.figure_size)
        if sns is not None:
            sns.heatmap(padded, cmap="viridis", cbar_kws={"label": "Memory (MB)"})
        else:
            plt.imshow(padded, cmap="viridis", aspect="auto")
            plt.colorbar(label="Memory (MB)")
        plt.title("JAX Device Memory Heatmap")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

    def export_data(self, results: Any, output_path: str, format: str = "csv") -> None:
        """Export available JAX timeline samples as CSV or JSON."""
        rows = self._rows(results)
        if format.lower() == "csv":
            with open(output_path, "w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output, fieldnames=["timestamp", "device_memory_mb"]
                )
                writer.writeheader()
                writer.writerows(rows)
        elif format.lower() == "json":
            Path(output_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        else:
            raise ValueError("format must be csv or json")

    def save_plots(self, results: Any, output_dir: str = "./plots/") -> None:
        """Save the standard JAX timeline, comparison, and heatmap outputs."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.plot_memory_timeline(results, save_path=str(output / "timeline.png"))
        if getattr(results, "function_profiles", None):
            self.plot_function_comparison(
                results.function_profiles,
                save_path=str(output / "function_comparison.png"),
            )
        self.create_memory_heatmap(results, save_path=str(output / "heatmap.png"))

    def create_interactive_dashboard(self, results: Any, port: int = 8050) -> None:
        """Serve an interactive JAX device-memory timeline when Dash is installed."""
        if not PLOTLY_AVAILABLE:
            logging.error("Plotly and Dash are required for the dashboard")
            return
        rows = self._rows(results)
        figure = go.Figure(
            data=[
                go.Scatter(
                    x=[row["timestamp"] for row in rows],
                    y=[row["device_memory_mb"] for row in rows],
                    mode="lines+markers",
                    name="Device Memory",
                )
            ]
        )
        figure.update_layout(title="JAX Device Memory Timeline")
        app = dash.Dash(__name__)
        app.layout = html.Div(
            [html.H1("JAX Stormlog Dashboard"), dcc.Graph(figure=figure)]
        )
        run_dashboard = getattr(app, "run", None)
        if run_dashboard is None:
            run_dashboard = app.run_server
        run_dashboard(debug=False, port=port, host="127.0.0.1")

    @staticmethod
    def _rows(results: Any) -> List[Dict[str, float]]:
        if hasattr(results, "snapshots"):
            return [
                {
                    "timestamp": snapshot.timestamp,
                    "device_memory_mb": snapshot.device_memory_mb,
                }
                for snapshot in results.snapshots
                if getattr(snapshot, "device_memory_available", True)
            ]
        return [
            {"timestamp": timestamp, "device_memory_mb": float(memory) / (1024 * 1024)}
            for timestamp, memory in zip(
                getattr(results, "timestamps", []),
                getattr(results, "memory_usage", []),
                strict=True,
            )
        ]


def _timeline_samples(results: Any) -> Optional[Tuple[List[float], List[float]]]:
    if hasattr(results, "snapshots") and results.snapshots:
        snapshots = [
            snapshot
            for snapshot in results.snapshots
            if getattr(snapshot, "device_memory_available", True)
        ]
        timestamps = [s.timestamp for s in snapshots]
        memory_usage = [s.device_memory_mb for s in snapshots]
    elif hasattr(results, "memory_usage") and results.memory_usage:
        # Fallback for simple track results
        memory_usage = [
            float(value) / (1024.0 * 1024.0) for value in results.memory_usage
        ]
        timestamps = getattr(results, "timestamps", list(range(len(memory_usage))))
    else:
        logging.warning("No memory data available for plotting")
        return None

    return timestamps, memory_usage


def _heatmap_samples(results: Any) -> List[float]:
    if hasattr(results, "snapshots"):
        samples = [
            snapshot.device_memory_mb
            for snapshot in results.snapshots
            if getattr(snapshot, "device_memory_available", True)
        ]
    else:
        samples = [
            float(value) / (1024 * 1024)
            for value in getattr(results, "memory_usage", [])
        ]
    return samples
