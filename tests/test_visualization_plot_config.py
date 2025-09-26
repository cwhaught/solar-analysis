"""
Tests for the plot configuration module.

Tests cover color schemes, styling functions, and figure creation utilities.
"""

import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from unittest.mock import patch, Mock

from src.visualization.plot_config import (
    SOLAR_COLORS, SOLAR_COLOR_PALETTE, FIGURE_STYLES,
    apply_solar_style, create_styled_figure, format_solar_axes
)


class TestPlotConfig:
    """Test suite for plot configuration utilities."""

    def test_solar_colors_constants(self):
        """Test that solar color constants are properly defined."""
        expected_keys = ['production', 'consumption', 'export', 'import']

        for key in expected_keys:
            assert key in SOLAR_COLORS
            assert isinstance(SOLAR_COLORS[key], str)
            assert SOLAR_COLORS[key].startswith('#')  # Should be hex colors
            assert len(SOLAR_COLORS[key]) == 7  # #RRGGBB format

    def test_solar_color_palette(self):
        """Test solar color palette structure."""
        assert isinstance(SOLAR_COLOR_PALETTE, list)
        assert len(SOLAR_COLOR_PALETTE) > 0

        for color in SOLAR_COLOR_PALETTE:
            assert isinstance(color, str)
            assert color.startswith('#')

    def test_figure_styles_constants(self):
        """Test that figure styles are properly defined."""
        assert isinstance(FIGURE_STYLES, dict)

        # Should have at least default style
        assert 'default' in FIGURE_STYLES

        for style_name, style_config in FIGURE_STYLES.items():
            assert isinstance(style_config, dict)
            assert 'figsize' in style_config
            assert 'dpi' in style_config

    @patch('matplotlib.pyplot.style.use')
    @patch('matplotlib.pyplot.rcParams.update')
    def test_apply_solar_style_default(self, mock_rc_update, mock_style_use):
        """Test applying default solar style."""
        apply_solar_style('default')

        # Should call style.use and rcParams.update
        mock_style_use.assert_called_once()
        mock_rc_update.assert_called_once()

        # Check that rcParams were updated with solar-specific settings
        call_args = mock_rc_update.call_args[0][0]
        assert isinstance(call_args, dict)

    @patch('matplotlib.pyplot.style.use')
    @patch('matplotlib.pyplot.rcParams.update')
    def test_apply_solar_style_presentation(self, mock_rc_update, mock_style_use):
        """Test applying presentation solar style."""
        apply_solar_style('presentation')

        mock_style_use.assert_called_once()
        mock_rc_update.assert_called_once()

    @patch('matplotlib.pyplot.style.use')
    @patch('matplotlib.pyplot.rcParams.update')
    def test_apply_solar_style_dashboard(self, mock_rc_update, mock_style_use):
        """Test applying dashboard solar style."""
        apply_solar_style('dashboard')

        mock_style_use.assert_called_once()
        mock_rc_update.assert_called_once()

    @patch('matplotlib.pyplot.style.use')
    @patch('matplotlib.pyplot.rcParams.update')
    def test_apply_solar_style_invalid(self, mock_rc_update, mock_style_use):
        """Test applying invalid style falls back to default."""
        apply_solar_style('nonexistent_style')

        # Should still work and fall back to default
        mock_style_use.assert_called_once()
        mock_rc_update.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    def test_create_styled_figure_default(self, mock_figure):
        """Test creating styled figure with default parameters."""
        mock_fig = Mock()
        mock_figure.return_value = mock_fig

        result = create_styled_figure()

        assert result == mock_fig
        mock_figure.assert_called_once()

        # Check default parameters
        call_kwargs = mock_figure.call_args[1]
        assert 'figsize' in call_kwargs
        assert 'dpi' in call_kwargs

    @patch('matplotlib.pyplot.figure')
    def test_create_styled_figure_custom_params(self, mock_figure):
        """Test creating styled figure with custom parameters."""
        mock_fig = Mock()
        mock_figure.return_value = mock_fig

        custom_figsize = (12, 8)
        custom_dpi = 200

        result = create_styled_figure(figsize=custom_figsize, dpi=custom_dpi)

        assert result == mock_fig

        call_kwargs = mock_figure.call_args[1]
        assert call_kwargs['figsize'] == custom_figsize
        assert call_kwargs['dpi'] == custom_dpi

    @patch('matplotlib.pyplot.figure')
    def test_create_styled_figure_with_style(self, mock_figure):
        """Test creating styled figure with specific style."""
        mock_fig = Mock()
        mock_figure.return_value = mock_fig

        with patch('src.visualization.plot_config.apply_solar_style') as mock_apply_style:
            result = create_styled_figure(style='presentation')

            mock_apply_style.assert_called_once_with('presentation')
            assert result == mock_fig

    def test_format_solar_axes_basic(self):
        """Test basic axes formatting."""
        fig, ax = plt.subplots()

        xlabel = "Test X Label"
        ylabel = "Test Y Label"

        format_solar_axes(ax, xlabel, ylabel)

        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

        # Check that grid is enabled
        assert ax.grid

        plt.close(fig)

    def test_format_solar_axes_with_title(self):
        """Test axes formatting with title."""
        fig, ax = plt.subplots()

        xlabel = "X Label"
        ylabel = "Y Label"
        title = "Test Title"

        format_solar_axes(ax, xlabel, ylabel, title=title)

        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel
        assert ax.get_title() == title

        plt.close(fig)

    def test_format_solar_axes_with_legend(self):
        """Test axes formatting with legend."""
        fig, ax = plt.subplots()

        # Add some data to create legend entries
        ax.plot([1, 2, 3], [1, 2, 3], label='Test Line')

        format_solar_axes(ax, "X", "Y", show_legend=True)

        legend = ax.get_legend()
        assert legend is not None

        plt.close(fig)

    def test_format_solar_axes_no_legend(self):
        """Test axes formatting without legend."""
        fig, ax = plt.subplots()

        ax.plot([1, 2, 3], [1, 2, 3], label='Test Line')

        format_solar_axes(ax, "X", "Y", show_legend=False)

        legend = ax.get_legend()
        assert legend is None

        plt.close(fig)

    def test_format_solar_axes_custom_grid(self):
        """Test axes formatting with custom grid settings."""
        fig, ax = plt.subplots()

        format_solar_axes(ax, "X", "Y", grid_alpha=0.5, grid_style='--')

        # Grid should be enabled
        assert ax.grid

        plt.close(fig)

    def test_color_accessibility(self):
        """Test that colors are accessible and distinguishable."""
        colors = list(SOLAR_COLORS.values())

        # Should have at least 4 distinct colors
        assert len(set(colors)) == len(colors)

        # All colors should be valid hex colors
        for color in colors:
            assert color.startswith('#')
            assert len(color) == 7
            # Should be valid hex digits
            try:
                int(color[1:], 16)
            except ValueError:
                pytest.fail(f"Invalid hex color: {color}")

    def test_color_palette_consistency(self):
        """Test that color palette is consistent with individual colors."""
        # Main colors should be in the palette
        main_colors = list(SOLAR_COLORS.values())

        for color in main_colors:
            assert color in SOLAR_COLOR_PALETTE, f"Color {color} not found in palette"

    @patch('matplotlib.pyplot.style.available', ['default', 'seaborn', 'ggplot'])
    @patch('matplotlib.pyplot.style.use')
    @patch('matplotlib.pyplot.rcParams.update')
    def test_style_fallback_mechanism(self, mock_rc_update, mock_style_use):
        """Test that style application falls back gracefully."""
        # Test with a preferred style that might not be available
        apply_solar_style('seaborn')

        mock_style_use.assert_called_once()
        mock_rc_update.assert_called_once()

    def test_figure_styles_completeness(self):
        """Test that all figure styles have required parameters."""
        required_keys = ['figsize', 'dpi']

        for style_name, style_config in FIGURE_STYLES.items():
            for key in required_keys:
                assert key in style_config, f"Style {style_name} missing {key}"

            # Validate figsize
            figsize = style_config['figsize']
            assert isinstance(figsize, tuple)
            assert len(figsize) == 2
            assert all(isinstance(dim, (int, float)) and dim > 0 for dim in figsize)

            # Validate dpi
            dpi = style_config['dpi']
            assert isinstance(dpi, int)
            assert dpi > 0

    def test_solar_theme_integration(self):
        """Test integration of solar theme elements."""
        # Test that applying style and creating figure work together
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.style.use'), \
             patch('matplotlib.pyplot.rcParams.update'):

            mock_fig = Mock()
            mock_figure.return_value = mock_fig

            # Should work without errors
            fig = create_styled_figure(style='default')
            assert fig == mock_fig

            # Should apply consistent styling
            mock_figure.assert_called_once()

    @patch('matplotlib.pyplot.tight_layout')
    def test_format_solar_axes_layout_optimization(self, mock_tight_layout):
        """Test that axes formatting optimizes layout when needed."""
        fig, ax = plt.subplots()

        format_solar_axes(ax, "Very Long X-Axis Label That Might Cause Issues",
                         "Very Long Y-Axis Label", optimize_layout=True)

        # tight_layout should be called when optimize_layout is True
        mock_tight_layout.assert_called_once()

        plt.close(fig)

    def test_real_plotting_integration(self):
        """Test that the configuration works with actual plotting."""
        # Create a real figure to test styling
        fig = create_styled_figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Plot some test data
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]

        ax.plot(x, y, color=SOLAR_COLORS['production'], label='Production')
        ax.plot(x, [i*0.8 for i in y], color=SOLAR_COLORS['consumption'], label='Consumption')

        format_solar_axes(ax, 'Time', 'Energy (kWh)', 'Test Solar Plot', show_legend=True)

        # Verify the plot was created successfully
        assert len(ax.get_lines()) == 2
        assert ax.get_xlabel() == 'Time'
        assert ax.get_ylabel() == 'Energy (kWh)'
        assert ax.get_title() == 'Test Solar Plot'
        assert ax.get_legend() is not None

        plt.close(fig)