"""Visualization tools for GPU memory profiling data."""

from datetime import datetime
from typing import Dict, List, Optional, Union

# Plotting imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Interactive plotting
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

from .profiler import GPUMemoryProfiler, MemorySnapshot, ProfileResult


class MemoryVisualizer:
    """Comprehensive visualization tool for memory profiling data."""

    def __init__(self, profiler: Optional[GPUMemoryProfiler] = None):
        """
        Initialize the visualizer.

        Args:
            profiler: GPUMemoryProfiler instance to visualize
        """
        self.profiler = profiler
        self.style_config = {
            "figure_size": (12, 8),
            "dpi": 100,
            "color_palette": "viridis",
            "font_size": 10,
            "title_size": 14,
            "label_size": 12,
        }

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette(self.style_config["color_palette"])

    def plot_memory_timeline(
        self,
        results: Optional[List[ProfileResult]] = None,
        snapshots: Optional[List[MemorySnapshot]] = None,
        save_path: Optional[str] = None,
        interactive: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot memory usage over time.

        Args:
            results: List of ProfileResults to plot
            snapshots: List of MemorySnapshots to plot
            save_path: Path to save the plot
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib or Plotly figure
        """
        # Get data
        if results is None and self.profiler:
            results = self.profiler.results
        if snapshots is None and self.profiler:
            snapshots = self.profiler.snapshots

        if not results and not snapshots:
            raise ValueError("No data available for plotting")

        # Prepare data
        timestamps = []
        allocated_memory = []
        reserved_memory = []
        labels = []

        # Add snapshot data
        if snapshots:
            for snapshot in snapshots:
                timestamps.append(snapshot.timestamp)
                allocated_memory.append(snapshot.allocated_memory)
                reserved_memory.append(snapshot.reserved_memory)
                labels.append(snapshot.operation or "monitor")

        # Add result data
        if results:
            for result in results:
                # Before snapshot
                timestamps.append(result.memory_before.timestamp)
                allocated_memory.append(result.memory_before.allocated_memory)
                reserved_memory.append(result.memory_before.reserved_memory)
                labels.append(f"before_{result.function_name}")

                # After snapshot
                timestamps.append(result.memory_after.timestamp)
                allocated_memory.append(result.memory_after.allocated_memory)
                reserved_memory.append(result.memory_after.reserved_memory)
                labels.append(f"after_{result.function_name}")

        if not timestamps:
            raise ValueError("No timestamp data available")

        # Convert to relative time (seconds from start)
        start_time = min(timestamps)
        relative_times = [(t - start_time) for t in timestamps]

        if interactive:
            return self._create_interactive_timeline(
                relative_times, allocated_memory, reserved_memory, labels, save_path
            )
        else:
            return self._create_static_timeline(
                relative_times, allocated_memory, reserved_memory, labels, save_path
            )

    def _create_static_timeline(
        self,
        times: List[float],
        allocated: List[int],
        reserved: List[int],
        labels: List[str],
        save_path: Optional[str],
    ) -> plt.Figure:
        """Create static matplotlib timeline plot."""
        fig_obj, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=self.style_config["figure_size"],
            sharex=True,
            dpi=self.style_config["dpi"],
        )
        fig: Figure = fig_obj

        # Plot allocated memory
        ax1.plot(
            times,
            [m / (1024**3) for m in allocated],
            "b-",
            linewidth=2,
            label="Allocated",
        )
        ax1.fill_between(
            times, [m / (1024**3) for m in allocated], alpha=0.3, color="blue"
        )
        ax1.set_ylabel(
            "Allocated Memory (GB)", fontsize=self.style_config["label_size"]
        )
        ax1.set_title(
            "GPU Memory Usage Over Time", fontsize=self.style_config["title_size"]
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot reserved memory
        ax2.plot(
            times,
            [m / (1024**3) for m in reserved],
            "r-",
            linewidth=2,
            label="Reserved",
        )
        ax2.fill_between(
            times, [m / (1024**3) for m in reserved], alpha=0.3, color="red"
        )
        ax2.set_ylabel("Reserved Memory (GB)", fontsize=self.style_config["label_size"])
        ax2.set_xlabel("Time (seconds)", fontsize=self.style_config["label_size"])
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _create_interactive_timeline(
        self,
        times: List[float],
        allocated: List[int],
        reserved: List[int],
        labels: List[str],
        save_path: Optional[str],
    ) -> go.Figure:
        """Create interactive plotly timeline plot."""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Allocated Memory", "Reserved Memory"),
            vertical_spacing=0.1,
        )

        # Convert bytes to GB for better readability
        allocated_gb = [m / (1024**3) for m in allocated]
        reserved_gb = [m / (1024**3) for m in reserved]

        # Allocated memory trace
        fig.add_trace(
            go.Scatter(
                x=times,
                y=allocated_gb,
                mode="lines+markers",
                name="Allocated Memory",
                line=dict(color="blue", width=2),
                fill="tonexty",
                hovertemplate="<b>Time:</b> %{x:.2f}s<br>"
                + "<b>Allocated:</b> %{y:.2f} GB<br>"
                + "<b>Operation:</b> %{text}<extra></extra>",
                text=labels,
            ),
            row=1,
            col=1,
        )

        # Reserved memory trace
        fig.add_trace(
            go.Scatter(
                x=times,
                y=reserved_gb,
                mode="lines+markers",
                name="Reserved Memory",
                line=dict(color="red", width=2),
                fill="tonexty",
                hovertemplate="<b>Time:</b> %{x:.2f}s<br>"
                + "<b>Reserved:</b> %{y:.2f} GB<br>"
                + "<b>Operation:</b> %{text}<extra></extra>",
                text=labels,
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title="GPU Memory Usage Timeline",
            showlegend=True,
            height=800,
            hovermode="closest",
        )

        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Memory (GB)", row=1, col=1)
        fig.update_yaxes(title_text="Memory (GB)", row=2, col=1)

        if save_path:
            if save_path.endswith(".html"):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=1200, height=800)

        return fig

    def plot_function_comparison(
        self,
        results: Optional[List[ProfileResult]] = None,
        metric: str = "memory_allocated",
        save_path: Optional[str] = None,
        interactive: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """
        Compare memory usage across different functions.

        Args:
            results: List of ProfileResults to compare
            metric: Metric to compare ('memory_allocated', 'execution_time', 'peak_memory')
            save_path: Path to save the plot
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib or Plotly figure
        """
        if results is None and self.profiler:
            results = self.profiler.results

        if not results:
            raise ValueError("No results available for comparison")

        # Aggregate data by function name
        function_memory_allocated: Dict[str, List[float]] = {}
        function_execution_time: Dict[str, List[float]] = {}
        function_peak_memory: Dict[str, List[float]] = {}
        for result in results:
            func_name = result.function_name
            function_memory_allocated.setdefault(func_name, []).append(
                float(result.memory_allocated)
            )
            function_execution_time.setdefault(func_name, []).append(
                float(result.execution_time)
            )
            function_peak_memory.setdefault(func_name, []).append(
                float(result.peak_memory_usage())
            )

        # Prepare plot data
        functions = list(function_memory_allocated.keys())

        if metric == "memory_allocated":
            values = [
                float(np.mean(function_memory_allocated[func])) for func in functions
            ]
            ylabel = "Average Memory Allocated (GB)"
            title = "Average Memory Allocation by Function"
            values = [v / (1024**3) for v in values]  # Convert to GB
        elif metric == "execution_time":
            values = [
                float(np.mean(function_execution_time[func])) for func in functions
            ]
            ylabel = "Average Execution Time (seconds)"
            title = "Average Execution Time by Function"
        elif metric == "peak_memory":
            values = [float(np.max(function_peak_memory[func])) for func in functions]
            ylabel = "Peak Memory Usage (GB)"
            title = "Peak Memory Usage by Function"
            values = [v / (1024**3) for v in values]  # Convert to GB
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if interactive:
            return self._create_interactive_bar_chart(
                functions, values, ylabel, title, save_path
            )
        else:
            return self._create_static_bar_chart(
                functions, values, ylabel, title, save_path
            )

    def _create_static_bar_chart(
        self,
        labels: List[str],
        values: List[float],
        ylabel: str,
        title: str,
        save_path: Optional[str],
    ) -> plt.Figure:
        """Create static matplotlib bar chart."""
        fig_obj, ax = plt.subplots(
            figsize=self.style_config["figure_size"],
            dpi=self.style_config["dpi"],
        )
        fig: Figure = fig_obj

        bars = ax.bar(labels, values, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=self.style_config["label_size"])
        ax.set_title(title, fontsize=self.style_config["title_size"])
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels if they're too long
        if max(len(label) for label in labels) > 10:
            plt.xticks(rotation=45, ha="right")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=self.style_config["font_size"],
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _create_interactive_bar_chart(
        self,
        labels: List[str],
        values: List[float],
        ylabel: str,
        title: str,
        save_path: Optional[str],
    ) -> go.Figure:
        """Create interactive plotly bar chart."""
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    text=[f"{v:.2f}" for v in values],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>"
                    + f"{ylabel}: %{{y:.2f}}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=title, xaxis_title="Function", yaxis_title=ylabel, height=600
        )

        if save_path:
            if save_path.endswith(".html"):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=1000, height=600)

        return fig

    def plot_memory_heatmap(
        self,
        results: Optional[List[ProfileResult]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a heatmap showing memory usage patterns.

        Args:
            results: List of ProfileResults to analyze
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        if results is None and self.profiler:
            results = self.profiler.results

        if not results:
            raise ValueError("No results available for heatmap")

        # Create data matrix
        functions = list(set(r.function_name for r in results))
        metrics = ["execution_time", "memory_allocated", "memory_freed", "peak_memory"]

        data_matrix = np.zeros((len(functions), len(metrics)))

        for i, func in enumerate(functions):
            func_results = [r for r in results if r.function_name == func]

            # Calculate average metrics
            data_matrix[i, 0] = np.mean([r.execution_time for r in func_results])
            data_matrix[i, 1] = np.mean([r.memory_allocated for r in func_results]) / (
                1024**3
            )  # GB
            data_matrix[i, 2] = np.mean([r.memory_freed for r in func_results]) / (
                1024**3
            )  # GB
            data_matrix[i, 3] = np.mean(
                [r.peak_memory_usage() for r in func_results]
            ) / (
                1024**3
            )  # GB

        # Create heatmap
        fig_obj, ax = plt.subplots(
            figsize=(10, max(6, len(functions) * 0.5)),
            dpi=self.style_config["dpi"],
        )
        fig: Figure = fig_obj

        # Normalize data for better visualization
        normalized_data = np.zeros_like(data_matrix)
        for j in range(data_matrix.shape[1]):
            col_max: float = float(np.max(data_matrix[:, j]))
            if col_max > 0:
                normalized_data[:, j] = data_matrix[:, j] / col_max

        im = ax.imshow(normalized_data, cmap="YlOrRd", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(functions)))
        ax.set_xticklabels(
            [
                "Execution Time",
                "Memory Allocated (GB)",
                "Memory Freed (GB)",
                "Peak Memory (GB)",
            ]
        )
        ax.set_yticklabels(functions)

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Normalized Value", rotation=-90, va="bottom")

        # Add text annotations
        for i in range(len(functions)):
            for j in range(len(metrics)):
                if metrics[j] == "execution_time":
                    text = f"{data_matrix[i, j]:.3f}s"
                else:
                    text = f"{data_matrix[i, j]:.2f}GB"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

        ax.set_title(
            "Memory Usage Heatmap by Function", fontsize=self.style_config["title_size"]
        )
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_dashboard(
        self,
        results: Optional[List[ProfileResult]] = None,
        snapshots: Optional[List[MemorySnapshot]] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.

        Args:
            results: List of ProfileResults
            snapshots: List of MemorySnapshots
            save_path: Path to save the dashboard

        Returns:
            Plotly figure with subplots
        """
        if results is None and self.profiler:
            results = self.profiler.results
        if snapshots is None and self.profiler:
            snapshots = self.profiler.snapshots

        # Create subplot grid
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Memory Timeline",
                "Function Comparison",
                "Memory Distribution",
                "Peak Memory Usage",
            ),
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "scatter"}],
            ],
        )

        # Timeline plot (top left)
        if snapshots:
            times = [(s.timestamp - snapshots[0].timestamp) for s in snapshots]
            allocated = [s.allocated_memory / (1024**3) for s in snapshots]

            fig.add_trace(
                go.Scatter(x=times, y=allocated, mode="lines", name="Allocated Memory"),
                row=1,
                col=1,
            )

        # Function comparison (top right)
        if results:
            func_memory: Dict[str, List[float]] = {}
            for result in results:
                if result.function_name not in func_memory:
                    func_memory[result.function_name] = []
                func_memory[result.function_name].append(
                    result.memory_allocated / (1024**3)
                )

            functions = list(func_memory.keys())
            avg_memory = [np.mean(func_memory[f]) for f in functions]

            fig.add_trace(
                go.Bar(x=functions, y=avg_memory, name="Avg Memory"), row=1, col=2
            )

        # Memory distribution (bottom left)
        if results:
            memory_values = [r.memory_allocated / (1024**3) for r in results]
            fig.add_trace(
                go.Histogram(x=memory_values, name="Memory Distribution"), row=2, col=1
            )

        # Peak memory scatter (bottom right)
        if results:
            exec_times = [r.execution_time for r in results]
            peak_memory = [r.peak_memory_usage() / (1024**3) for r in results]

            fig.add_trace(
                go.Scatter(
                    x=exec_times,
                    y=peak_memory,
                    mode="markers",
                    name="Execution Time vs Peak Memory",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title_text="GPU Memory Profiling Dashboard", height=800, showlegend=True
        )

        # Update axis labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Memory (GB)", row=1, col=1)
        fig.update_xaxes(title_text="Function", row=1, col=2)
        fig.update_yaxes(title_text="Avg Memory (GB)", row=1, col=2)
        fig.update_xaxes(title_text="Memory (GB)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Execution Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Peak Memory (GB)", row=2, col=2)

        if save_path:
            if save_path.endswith(".html"):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=1400, height=800)

        return fig

    def export_data(
        self,
        results: Optional[List[ProfileResult]] = None,
        snapshots: Optional[List[MemorySnapshot]] = None,
        format: str = "csv",
        save_path: str = "memory_profile_data",
    ) -> str:
        """
        Export profiling data to various formats.

        Args:
            results: List of ProfileResults to export
            snapshots: List of MemorySnapshots to export
            format: Export format ('csv', 'json', 'excel')
            save_path: Base path for saved files

        Returns:
            Path to saved file
        """
        if results is None and self.profiler:
            results = self.profiler.results
        if snapshots is None and self.profiler:
            snapshots = self.profiler.snapshots

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "csv":
            # Export results
            if results:
                results_data = []
                for r in results:
                    results_data.append(
                        {
                            "function_name": r.function_name,
                            "execution_time": r.execution_time,
                            "memory_allocated": r.memory_allocated,
                            "memory_freed": r.memory_freed,
                            "peak_memory": r.peak_memory_usage(),
                            "memory_diff": r.memory_diff(),
                            "tensors_created": r.tensors_created,
                            "tensors_deleted": r.tensors_deleted,
                        }
                    )

                results_df = pd.DataFrame(results_data)
                results_path = f"{save_path}_results_{timestamp}.csv"
                results_df.to_csv(results_path, index=False)

            # Export snapshots
            if snapshots:
                snapshots_data = []
                for s in snapshots:
                    snapshots_data.append(
                        {
                            "timestamp": s.timestamp,
                            "operation": s.operation,
                            "allocated_memory": s.allocated_memory,
                            "reserved_memory": s.reserved_memory,
                            "active_memory": s.active_memory,
                            "inactive_memory": s.inactive_memory,
                            "device_id": s.device_id,
                        }
                    )

                snapshots_df = pd.DataFrame(snapshots_data)
                snapshots_path = f"{save_path}_snapshots_{timestamp}.csv"
                snapshots_df.to_csv(snapshots_path, index=False)

            return f"{save_path}_{timestamp}.csv"

        elif format == "json":
            import json

            export_data = {
                "metadata": {
                    "export_time": timestamp,
                    "num_results": len(results) if results else 0,
                    "num_snapshots": len(snapshots) if snapshots else 0,
                },
                "results": [r.to_dict() for r in results] if results else [],
                "snapshots": [s.to_dict() for s in snapshots] if snapshots else [],
            }

            json_path = f"{save_path}_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            return json_path

        else:
            raise ValueError(f"Unsupported format: {format}")

    def show(self, fig: Union[plt.Figure, go.Figure]) -> None:
        """Display a figure."""
        if isinstance(fig, plt.Figure):
            plt.show()
        else:
            fig.show()
