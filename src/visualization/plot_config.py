"""
Plot Configuration - Standardized visualization setup for solar energy analysis

Provides consistent plotting styles, color schemes, and configuration
for all solar energy analysis visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# Top-level color constants for easy import
SOLAR_COLORS = {
    'production': '#FFB000',      # Solar yellow/orange
    'consumption': '#1f77b4',     # Blue
    'export': '#2ca02c',          # Green
    'import': '#d62728',          # Red
    'weather': '#ff7f0e',         # Orange
    'efficiency': '#9467bd',      # Purple
    'financial': '#17becf',       # Cyan
    'seasonal': '#8c564b',        # Brown
    'anomaly': '#e377c2'          # Pink
}

# Color palette for easy access
SOLAR_COLOR_PALETTE = list(SOLAR_COLORS.values())

# Figure style configurations
FIGURE_STYLES = {
    'default': {
        'figsize': (12, 8),
        'dpi': 100,
        'style': 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default'
    },
    'presentation': {
        'figsize': (14, 10),
        'dpi': 150,
        'style': 'seaborn-v0_8-talk' if 'seaborn-v0_8-talk' in plt.style.available else 'default'
    },
    'dashboard': {
        'figsize': (16, 12),
        'dpi': 100,
        'style': 'seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default'
    }
}


def apply_solar_style(style: str = 'default') -> None:
    """
    Apply solar-themed matplotlib styling.

    Args:
        style: Style name from FIGURE_STYLES
    """
    style_config = FIGURE_STYLES.get(style, FIGURE_STYLES['default'])

    # Apply matplotlib style
    try:
        plt.style.use(style_config['style'])
    except OSError:
        # Fallback to default if style not available
        plt.style.use('default')

    # Apply custom solar parameters
    plt.rcParams.update({
        'figure.figsize': style_config['figsize'],
        'figure.dpi': style_config['dpi'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'patch.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333'
    })


def create_styled_figure(figsize: Optional[Tuple[float, float]] = None,
                        dpi: Optional[int] = None,
                        style: str = 'default') -> plt.Figure:
    """
    Create a styled figure with solar theme.

    Args:
        figsize: Figure size tuple (width, height)
        dpi: Figure DPI
        style: Style name from FIGURE_STYLES

    Returns:
        Configured matplotlib figure
    """
    apply_solar_style(style)

    style_config = FIGURE_STYLES.get(style, FIGURE_STYLES['default'])

    # Use provided parameters or fall back to style defaults
    fig_size = figsize or style_config['figsize']
    fig_dpi = dpi or style_config['dpi']

    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    return fig


def format_solar_axes(ax: plt.Axes,
                     xlabel: str,
                     ylabel: str,
                     title: Optional[str] = None,
                     show_legend: bool = True,
                     grid_alpha: float = 0.3,
                     grid_style: str = '-',
                     optimize_layout: bool = False) -> None:
    """
    Apply consistent formatting to axes.

    Args:
        ax: Matplotlib axes to format
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Optional axes title
        show_legend: Whether to show legend if present
        grid_alpha: Grid transparency
        grid_style: Grid line style
        optimize_layout: Whether to optimize layout
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, fontweight='bold', pad=15)

    # Configure grid
    ax.grid(True, alpha=grid_alpha, linestyle=grid_style)

    # Handle legend
    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True, fancybox=True, shadow=True)
    elif not show_legend:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    # Optimize layout if requested
    if optimize_layout:
        plt.tight_layout()


class SolarPlotConfig:
    """
    Centralized plotting configuration for solar energy visualizations
    """

    # Use the top-level constants
    SOLAR_COLORS = SOLAR_COLORS

    # Seasonal color palette
    SEASONAL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Winter, Spring, Summer, Fall

    # Monthly color palette (12 colors)
    MONTHLY_COLORS = plt.cm.Set3(np.linspace(0, 1, 12))

    def __init__(self):
        self._original_params = {}

    def setup_solar_style(self, figsize: Tuple[int, int] = (12, 8),
                         style: str = 'default',
                         palette: str = 'husl',
                         context: str = 'notebook') -> None:
        """
        Set up standardized plotting style for solar analysis

        Args:
            figsize: Default figure size
            style: Matplotlib style to use
            palette: Seaborn color palette
            context: Seaborn context (paper, notebook, talk, poster)
        """
        # Store original parameters for restoration
        self._original_params = plt.rcParams.copy()

        # Apply style
        plt.style.use(style)
        sns.set_palette(palette)
        sns.set_context(context)

        # Set common parameters
        plt.rcParams.update({
            'figure.figsize': figsize,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
            'patch.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    def create_solar_colormap(self, data_type: str = 'production') -> str:
        """
        Get appropriate color for solar data type

        Args:
            data_type: Type of solar data (production, consumption, etc.)

        Returns:
            Color code for the data type
        """
        return self.SOLAR_COLORS.get(data_type, '#1f77b4')

    def setup_time_series_plot(self, ax: plt.Axes, title: str,
                              ylabel: str = "Energy (kWh)",
                              xlabel: str = "Date") -> plt.Axes:
        """
        Configure axes for time series plots

        Args:
            ax: Matplotlib axes object
            title: Plot title
            ylabel: Y-axis label
            xlabel: X-axis label

        Returns:
            Configured axes object
        """
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        return ax

    def setup_financial_plot(self, ax: plt.Axes, title: str,
                           ylabel: str = "Financial Value ($)") -> plt.Axes:
        """
        Configure axes for financial plots

        Args:
            ax: Matplotlib axes object
            title: Plot title
            ylabel: Y-axis label

        Returns:
            Configured axes object
        """
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        return ax

    def setup_comparison_plot(self, ax: plt.Axes, title: str,
                            categories: List[str],
                            ylabel: str = "Value") -> plt.Axes:
        """
        Configure axes for comparison/bar plots

        Args:
            ax: Matplotlib axes object
            title: Plot title
            categories: List of category names for x-axis
            ylabel: Y-axis label

        Returns:
            Configured axes object
        """
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        return ax

    def create_dashboard_figure(self, layout: Tuple[int, int] = (2, 2),
                               figsize: Tuple[int, int] = (16, 12),
                               title: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create multi-panel dashboard figure

        Args:
            layout: Grid layout (rows, cols)
            figsize: Overall figure size
            title: Optional main title

        Returns:
            Tuple of (figure, axes_array)
        """
        fig, axes = plt.subplots(*layout, figsize=figsize)

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        # Ensure axes is always a 2D array
        if layout[0] == 1 and layout[1] == 1:
            axes = np.array([[axes]])
        elif layout[0] == 1 or layout[1] == 1:
            axes = axes.reshape(layout)

        plt.tight_layout()
        return fig, axes

    def add_solar_annotations(self, ax: plt.Axes, data: Any,
                            peak_value: bool = True,
                            average_line: bool = True) -> plt.Axes:
        """
        Add common solar data annotations

        Args:
            ax: Matplotlib axes object
            data: Data series for annotations
            peak_value: Whether to annotate peak value
            average_line: Whether to add average line

        Returns:
            Annotated axes object
        """
        if hasattr(data, 'max') and peak_value:
            peak_val = data.max()
            ax.axhline(y=peak_val, color='red', linestyle='--', alpha=0.7,
                      label=f'Peak: {peak_val:.1f}')

        if hasattr(data, 'mean') and average_line:
            avg_val = data.mean()
            ax.axhline(y=avg_val, color='green', linestyle='--', alpha=0.7,
                      label=f'Average: {avg_val:.1f}')

        if peak_value or average_line:
            ax.legend()

        return ax

    def restore_defaults(self) -> None:
        """
        Restore original matplotlib parameters
        """
        if self._original_params:
            plt.rcParams.update(self._original_params)

    def get_seasonal_colors(self, n_seasons: int = 4) -> List[str]:
        """
        Get colors for seasonal data

        Args:
            n_seasons: Number of seasons (usually 4)

        Returns:
            List of colors for seasons
        """
        return self.SEASONAL_COLORS[:n_seasons]

    def get_monthly_colors(self) -> List[str]:
        """
        Get colors for monthly data

        Returns:
            List of 12 colors for months
        """
        return [plt.colors.to_hex(c) for c in self.MONTHLY_COLORS]


# Convenience functions for common use cases

def setup_notebook_plots(figsize: Tuple[int, int] = (12, 8)) -> SolarPlotConfig:
    """
    Quick setup for notebook plotting

    Args:
        figsize: Default figure size

    Returns:
        Configured SolarPlotConfig instance
    """
    config = SolarPlotConfig()
    config.setup_solar_style(figsize=figsize)
    return config


def create_production_plot(data: Any, title: str = "Solar Production",
                          figsize: Tuple[int, int] = (12, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create standardized production plot

    Args:
        data: Production data series
        title: Plot title
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    config = SolarPlotConfig()
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data
    if hasattr(data, 'index'):  # Time series data
        ax.plot(data.index, data.values, color=config.SOLAR_COLORS['production'], linewidth=2)
        ax.tick_params(axis='x', rotation=45)
    else:  # Array data
        ax.plot(data, color=config.SOLAR_COLORS['production'], linewidth=2)

    config.setup_time_series_plot(ax, title)
    config.add_solar_annotations(ax, data)

    plt.tight_layout()
    return fig, ax


def create_financial_comparison_plot(categories: List[str], values: List[float],
                                   title: str = "Financial Comparison",
                                   figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create standardized financial comparison plot

    Args:
        categories: Category names
        values: Values for each category
        title: Plot title
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    config = SolarPlotConfig()
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    bars = ax.bar(categories, values, color=config.SOLAR_COLORS['financial'], alpha=0.7)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
               f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')

    config.setup_financial_plot(ax, title)
    plt.tight_layout()
    return fig, ax