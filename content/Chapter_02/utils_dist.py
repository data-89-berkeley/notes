"""
Utility functions and visualization class for distribution models exploration.
Used by distribution_models_week_2.ipynb
"""

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display, clear_output
from scipy import stats
import time


def sample_distribution(dist_type, dist_category, n_samples, **params):
    """Sample from the specified distribution"""
    if dist_category == "Continuous":
        if dist_type == "Uniform":
            low = params.get('low', 0)
            high = params.get('high', 1)
            return np.random.uniform(low, high, n_samples)
        elif dist_type == "Exponential":
            scale = params.get('scale', 1)
            return np.random.exponential(scale, n_samples)
        elif dist_type == "Pareto":
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            # Pareto distribution: P(X > x) = (scale/x)^shape for x >= scale
            # Using scipy's parameterization: pareto(b, loc=0, scale=scale) gives support [scale, inf)
            return stats.pareto.rvs(shape, loc=0, scale=scale, size=n_samples)
        elif dist_type == "Beta":
            alpha = params.get('alpha', 2)
            beta = params.get('beta', 2)
            return np.random.beta(alpha, beta, n_samples)
        elif dist_type == "Gamma":
            shape = params.get('shape', 2)
            scale = params.get('scale', 1)
            return np.random.gamma(shape, scale, n_samples)
        elif dist_type == "Normal":
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            return np.random.normal(mean, std, n_samples)
        else:
            return np.random.normal(0, 1, n_samples)
    else:  # Discrete
        if dist_type == "Bernoulli":
            p = params.get('p', 0.5)
            return np.random.binomial(1, p, n_samples)
        elif dist_type == "Geometric":
            p = params.get('p', 0.5)
            return np.random.geometric(p, n_samples)
        elif dist_type == "Binomial":
            n = params.get('n', 10)
            p = params.get('p', 0.5)
            return np.random.binomial(n, p, n_samples)
        elif dist_type == "Poisson":
            lam = params.get('lam', 5)
            return np.random.poisson(lam, n_samples)
        elif dist_type == "Hypergeometric":
            ngood = params.get('ngood', 10)
            nbad = params.get('nbad', 10)
            nsample = params.get('nsample', 10)
            return np.random.hypergeometric(ngood, nbad, nsample, n_samples)
        else:
            return np.random.binomial(1, 0.5, n_samples)


def compute_pdf_pmf(x_values, dist_type, dist_category, **params):
    """Compute PDF (continuous) or PMF (discrete) for the distribution"""
    if dist_category == "Continuous":
        if dist_type == "Uniform":
            low = params.get('low', 0)
            high = params.get('high', 1)
            return stats.uniform.pdf(x_values, loc=low, scale=high-low)
        elif dist_type == "Exponential":
            scale = params.get('scale', 1)
            return stats.expon.pdf(x_values, scale=scale)
        elif dist_type == "Pareto":
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            return stats.pareto.pdf(x_values, shape, loc=0, scale=scale)
        elif dist_type == "Beta":
            alpha = params.get('alpha', 2)
            beta = params.get('beta', 2)
            return stats.beta.pdf(x_values, alpha, beta)
        elif dist_type == "Gamma":
            shape = params.get('shape', 2)
            scale = params.get('scale', 1)
            return stats.gamma.pdf(x_values, shape, scale=scale)
        elif dist_type == "Normal":
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            return stats.norm.pdf(x_values, mean, std)
        else:
            return stats.norm.pdf(x_values, 0, 1)
    else:  # Discrete
        if dist_type == "Bernoulli":
            p = params.get('p', 0.5)
            return stats.bernoulli.pmf(np.round(x_values).astype(int), p)
        elif dist_type == "Geometric":
            p = params.get('p', 0.5)
            return stats.geom.pmf(np.round(x_values).astype(int), p)
        elif dist_type == "Binomial":
            n = params.get('n', 10)
            p = params.get('p', 0.5)
            return stats.binom.pmf(np.round(x_values).astype(int), n, p)
        elif dist_type == "Poisson":
            lam = params.get('lam', 5)
            # For discrete, return PMF at integer values
            return stats.poisson.pmf(np.round(x_values).astype(int), lam)
        elif dist_type == "Hypergeometric":
            ngood = params.get('ngood', 10)
            nbad = params.get('nbad', 10)
            nsample = params.get('nsample', 10)
            return stats.hypergeom.pmf(np.round(x_values).astype(int), ngood + nbad, ngood, nsample)
        else:
            return stats.bernoulli.pmf(np.round(x_values).astype(int), 0.5)


def compute_true_probability(dist_type, dist_category, prob_type, bound1, bound2, **params):
    """Compute true probability using CDF/PMF"""
    if dist_category == "Continuous":
        if dist_type == "Uniform":
            low = params.get('low', 0)
            high = params.get('high', 1)
            dist = stats.uniform(loc=low, scale=high-low)
        elif dist_type == "Exponential":
            scale = params.get('scale', 1)
            dist = stats.expon(scale=scale)
        elif dist_type == "Pareto":
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            dist = stats.pareto(shape, loc=0, scale=scale)
        elif dist_type == "Beta":
            alpha = params.get('alpha', 2)
            beta = params.get('beta', 2)
            dist = stats.beta(alpha, beta)
        elif dist_type == "Gamma":
            shape = params.get('shape', 2)
            scale = params.get('scale', 1)
            dist = stats.gamma(shape, scale=scale)
        elif dist_type == "Normal":
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            dist = stats.norm(mean, std)
        else:
            dist = stats.norm(0, 1)
        
        if prob_type == "of outcome":
            # For continuous, P(X = x) = 0, so return 0
            return 0.0
        elif prob_type == "under upper bound":
            # P(X <= bound2) - inclusive upper bound
            prob = dist.cdf(bound2)
            # Round to 1.0 if very close (within floating point precision)
            return 1.0 if abs(prob - 1.0) < 1e-10 else prob
        elif prob_type == "above lower bound":
            # P(X >= bound1) - inclusive lower bound
            prob = 1 - dist.cdf(bound1)
            # Round to 1.0 if very close (within floating point precision)
            return 1.0 if abs(prob - 1.0) < 1e-10 else prob
        elif prob_type == "in interval":
            # For continuous distributions, P(bound1 <= X <= bound2) - inclusive bounds
            # Since P(X = x) = 0 for continuous, this equals CDF(bound2) - CDF(bound1)
            prob = dist.cdf(bound2) - dist.cdf(bound1)
            
            # Ensure result is in [0, 1]
            prob = max(0.0, min(1.0, prob))
            return 1.0 if abs(prob - 1.0) < 1e-6 else prob
    else:  # Discrete
        if dist_type == "Bernoulli":
            p = params.get('p', 0.5)
            dist = stats.bernoulli(p)
        elif dist_type == "Geometric":
            p = params.get('p', 0.5)
            dist = stats.geom(p)
        elif dist_type == "Binomial":
            n = params.get('n', 10)
            p = params.get('p', 0.5)
            dist = stats.binom(n, p)
        elif dist_type == "Poisson":
            lam = params.get('lam', 5)
            dist = stats.poisson(lam)
        elif dist_type == "Hypergeometric":
            ngood = params.get('ngood', 10)
            nbad = params.get('nbad', 10)
            nsample = params.get('nsample', 10)
            dist = stats.hypergeom(ngood + nbad, ngood, nsample)
        else:
            dist = stats.bernoulli(0.5)
        
        if prob_type == "of outcome":
            return dist.pmf(int(np.round(bound1)))
        elif prob_type == "under upper bound":
            # P(X <= bound2) - inclusive upper bound
            return dist.cdf(int(np.round(bound2)))
        elif prob_type == "above lower bound":
            # P(X >= bound1) - inclusive lower bound: 1 - CDF(bound1 - 1)
            return 1 - dist.cdf(int(np.round(bound1)) - 1)
        elif prob_type == "in interval":
            # P(bound1 <= X <= bound2) - inclusive both bounds
            # = CDF(bound2) - CDF(bound1 - 1)
            return dist.cdf(int(np.round(bound2))) - dist.cdf(int(np.round(bound1)) - 1)
    
    return 0.0


def compute_estimated_probability(samples, prob_type, bound1, bound2):
    """Compute estimated probability from samples using inclusive bounds"""
    if prob_type == "of outcome":
        # Count samples exactly equal to bound1 (for discrete, use integer comparison)
        if samples.dtype in [np.int32, np.int64] or np.all(samples == np.round(samples)):
            count = np.sum(samples == int(np.round(bound1)))
        else:
            count = np.sum(np.abs(samples - bound1) < 1e-6)
    elif prob_type == "under upper bound":
        count = np.sum(samples <= bound2)  # Inclusive upper bound
    elif prob_type == "above lower bound":
        count = np.sum(samples >= bound1)  # Inclusive lower bound
    elif prob_type == "in interval":
        count = np.sum((samples >= bound1) & (samples <= bound2))  # Inclusive both bounds
    else:
        return 0.0
    
    return count / len(samples) if len(samples) > 0 else 0.0


def determine_batch_size(sample_index):
    """
    Determine how many samples to add in this batch for animation.
    Larger batches = fewer plot updates = faster animation.
    - Samples 1-50: 5 at a time (for smooth start)
    - Samples 50-200: 20 at a time
    - Samples 200-500: 50 at a time
    - Samples 500+: 100 at a time
    """
    if sample_index < 50:
        return 5
    elif sample_index < 200:
        return 20
    elif sample_index < 500:
        return 50
    else:
        return 100


class DistributionProbabilityVisualization:
    """Interactive visualization for distribution sampling and probability calculation."""
    
    def __init__(self, initial_dist_type=None, lock_distribution=False):
        """
        Parameters
        ----------
        initial_dist_type : str or None
            Distribution to select when the visualization opens.
        lock_distribution : bool
            When True, restrict the type and distribution controls to
            ``initial_dist_type`` while leaving its parameter controls active.
        """
        self.samples = np.array([])
        self.plot_output = widgets.Output()
        self.show_pdf_flag = False  # Track whether PDF should be shown
        self.show_shaded_region_flag = False  # Track whether shaded region should be shown
        self.bounds_interacted = False  # Track whether user has interacted with bounds
        
        # Distribution options
        self.continuous_dists = ["Uniform", "Exponential", "Pareto", "Beta", "Gamma", "Normal"]
        self.discrete_dists = ["Bernoulli", "Geometric", "Binomial", "Poisson", "Hypergeometric"]
        
        self._create_widgets()
        self._set_initial_distribution(initial_dist_type, lock_distribution)
        self._setup_callbacks()

    def _set_initial_distribution(self, dist_type, lock_distribution):
        """Select and optionally lock a supported distribution before callbacks run."""
        if dist_type is None:
            return

        if dist_type in self.discrete_dists:
            category = "Discrete"
            options = self.discrete_dists
        elif dist_type in self.continuous_dists:
            category = "Continuous"
            options = self.continuous_dists
        else:
            supported = self.discrete_dists + self.continuous_dists
            raise ValueError(
                f"Unknown distribution {dist_type!r}. "
                f"Choose one of: {', '.join(supported)}."
            )

        self.category_dropdown.value = category
        self.dist_dropdown.options = options
        self.dist_dropdown.value = dist_type

        if lock_distribution:
            self.category_dropdown.options = [category]
            self.dist_dropdown.options = [dist_type]
            self.category_dropdown.disabled = True
            self.dist_dropdown.disabled = True
            # A fixed explorer is meant to start as a distribution playground,
            # so show its theoretical curve/masses without an extra click.
            self.show_pdf_flag = True
            self.show_pdf_button.description = "Hide PDF/PMF"
        
    def _create_widgets(self):
        """Create all widgets"""
        # Category dropdown
        self.category_dropdown = widgets.Dropdown(
            options=["Discrete", "Continuous"],
            value="Discrete",
            description="Type:",
            style={'description_width': 'initial'}
        )
        
        # Distribution dropdown (will be updated based on category)
        self.dist_dropdown = widgets.Dropdown(
            options=self.discrete_dists,
            value="Bernoulli",
            description="Distribution:",
            style={'description_width': 'initial'}
        )
        
        # Parameter widgets (will be shown/hidden based on distribution)
        self.param_widgets = {}
        
        # Normal parameters
        self.param_widgets['Normal'] = [
            widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Mean:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1, min=0.1, max=3, step=0.1, description='Std:', style={'description_width': 'initial'})
        ]
        
        # Exponential parameters
        self.param_widgets['Exponential'] = [
            widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Scale:', style={'description_width': 'initial'})
        ]
        
        # Beta parameters
        self.param_widgets['Beta'] = [
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description='Alpha:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description='Beta:', style={'description_width': 'initial'})
        ]
        
        # Gamma parameters
        self.param_widgets['Gamma'] = [
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description='Shape:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Scale:', style={'description_width': 'initial'})
        ]
        
        # Uniform parameters
        self.param_widgets['Uniform'] = [
            widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Low:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='High:', style={'description_width': 'initial'})
        ]
        
        # Pareto parameters
        self.param_widgets['Pareto'] = [
            widgets.FloatSlider(value=2.0, min=0.1, max=5, step=0.1, description='Shape (α):', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1.0, min=0.1, max=5, step=0.1, description='Scale (xₘ):', style={'description_width': 'initial'})
        ]
        
        # Poisson parameters
        self.param_widgets['Poisson'] = [
            widgets.FloatSlider(value=2, min=0.5, max=20, step=0.1, description='Lambda:', style={'description_width': 'initial'})
        ]
        
        # Binomial parameters
        self.param_widgets['Binomial'] = [
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='n:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='p:', style={'description_width': 'initial'})
        ]
        
        # Bernoulli parameters
        self.param_widgets['Bernoulli'] = [
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='p:', style={'description_width': 'initial'})
        ]
        
        # Geometric parameters
        self.param_widgets['Geometric'] = [
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='p:', style={'description_width': 'initial'})
        ]
        
        # Hypergeometric parameters
        self.param_widgets['Hypergeometric'] = [
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='ngood:', style={'description_width': 'initial'}),
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='nbad:', style={'description_width': 'initial'}),
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='nsample:', style={'description_width': 'initial'})
        ]
        
        # Sample size (allow small draws like 5–10 as well as large Monte Carlo samples)
        self.n_samples_slider = widgets.IntSlider(
            value=1000, min=5, max=10000, step=5,
            description="Samples:",
            style={'description_width': 'initial'}
        )

        # Number of bins for continuous sample histograms
        self.n_bins_slider = widgets.IntSlider(
            value=20, min=1, max=100, step=1,
            description="Bins:",
            style={'description_width': 'initial'},
            continuous_update=False
        )
        # Hidden for discrete (integer outcomes use one bar per value)
        self.n_bins_slider.layout.display = 'none'
        
        # Draw samples button
        self.draw_button = widgets.Button(
            description="Draw Samples",
            button_style='success'
        )
        
        # Reset all button
        self.reset_button = widgets.Button(
            description="Reset all",
            button_style='warning'
        )
        
        # Show PDF/PMF button (available before samples are drawn)
        self.show_pdf_button = widgets.Button(
            description="Show PDF/PMF",
            button_style='info',
            disabled=False
        )
        
        # Show Shaded Region button (shaded region follows probability type selection)
        self.show_shaded_region_button = widgets.Button(
            description="Display Shaded Region",
            button_style='primary',
            disabled=False
        )
        
        # Probability calculation dropdown
        self.prob_type_dropdown = widgets.Dropdown(
            options=["", "of outcome", "under upper bound", "above lower bound", "in interval"],
            value="",
            description="Find Probability:",
            style={'description_width': 'initial'}
        )
        
        # Bound sliders (will be shown/hidden based on prob_type)
        self.bound1_slider = widgets.FloatSlider(
            value=0, min=-10, max=10, step=0.1,
            description="Lower bound:",
            style={'description_width': 'initial'}
        )
        
        self.bound2_slider = widgets.FloatSlider(
            value=1, min=-10, max=10, step=0.1,
            description="Upper bound:",
            style={'description_width': 'initial'}
        )
        
        # Probability display
        self.prob_label = widgets.HTML(
            value='<div style="font-size: 18px; padding: 10px; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><b>Estimated Probability:</b> <span style="color: #0066cc; font-size: 20px; font-weight: bold;">N/A</span><br><b>True Probability:</b> <span style="color: #cc6600; font-size: 20px; font-weight: bold;">N/A</span></div>'
        )
        
        # Status display for animation progress
        self.status_html = widgets.HTML(
            value="Ready to draw samples."
        )
        
        # Parameter container (will be updated)
        self.param_container = widgets.VBox([])
        
        # Slider container (will be dynamically updated based on prob_type)
        self.slider_container = widgets.VBox([self.bound1_slider, self.bound2_slider])
        
        # Probability controls (always visible so PDF/PMF can be shown before sampling)
        self.prob_controls_container = widgets.VBox([
            widgets.HTML("<hr>"),
            self.prob_type_dropdown,
            self.slider_container,  # Dynamic slider container
            widgets.HBox([self.show_pdf_button]),  # Only PDF button (shaded region is automatic)
            widgets.HTML("<hr>"),
            self.prob_label
        ])
        
        # Initialize slider visibility based on default prob_type
        self._update_slider_visibility()
        self._update_bound_sliders()
        
    def _setup_callbacks(self):
        """Setup widget callbacks"""
        self.category_dropdown.observe(self._on_category_change, names='value')
        self.dist_dropdown.observe(self._on_dist_change, names='value')
        self.prob_type_dropdown.observe(self._on_prob_type_change, names='value')
        self.draw_button.on_click(self._on_draw_clicked)
        self.reset_button.on_click(self._on_reset_clicked)
        self.show_pdf_button.on_click(self._on_show_pdf_clicked)
        self.show_shaded_region_button.on_click(self._on_show_shaded_region_clicked)
        
        # Update plot when sliders change (but only if samples exist)
        for widgets_list in self.param_widgets.values():
            for w in widgets_list:
                w.observe(self._on_param_change, names='value')
        
        self.n_samples_slider.observe(self._on_param_change, names='value')
        self.n_bins_slider.observe(self._on_n_bins_change, names='value')
        self.bound1_slider.observe(self._on_bound_change, names='value')
        self.bound2_slider.observe(self._on_bound_change, names='value')
        
    def _on_n_bins_change(self, change):
        """Redraw histogram when bin count changes (continuous samples only)."""
        if self.category_dropdown.value == "Continuous" and len(self.samples) > 0:
            self._update_plot()
    def _on_bound_change(self, change):
        """Handle bound slider changes"""
        self.bounds_interacted = True  # User has interacted with bounds
        if len(self.samples) > 0 or self.show_pdf_flag:
            self._update_plot()
        
    def _on_param_change(self, change):
        """Handle parameter changes"""
        # Keep bound/slider ranges in sync when theoretical support changes
        if self.dist_dropdown.value in ["Poisson", "Binomial"] or self.show_pdf_flag:
            self._update_bound_sliders()
        if len(self.samples) > 0 or self.show_pdf_flag:
            self._update_plot()
        
    def _on_category_change(self, change):
        """Handle category change"""
        if change['new'] == "Continuous":
            self.dist_dropdown.options = self.continuous_dists
            self.dist_dropdown.value = "Uniform"
        else:
            self.dist_dropdown.options = self.discrete_dists
            self.dist_dropdown.value = "Bernoulli"
        self._update_param_widgets()
        # Reset probability type dropdown to empty
        self.prob_type_dropdown.value = ""
        # Clear samples; keep PDF/PMF controls available
        self.samples = np.array([])
        self.show_pdf_flag = False
        self.show_pdf_button.disabled = False
        self.show_pdf_button.description = "Show PDF/PMF"
        self.show_shaded_region_flag = False
        self.show_shaded_region_button.disabled = False
        self.show_shaded_region_button.description = "Display Shaded Region"
        # Reset bounds interaction flag - histogram will be all blue until user interacts
        self.bounds_interacted = False
        # Reset status
        self.status_html.value = "Ready to draw samples."
        self._update_bound_sliders()
        self._show_blank_plot()
        
    def _on_dist_change(self, change):
        """Handle distribution change"""
        self._update_param_widgets()
        # Reset probability type dropdown to empty
        self.prob_type_dropdown.value = ""
        # Clear samples; keep PDF/PMF controls available
        self.samples = np.array([])
        self.show_pdf_flag = False
        self.show_pdf_button.disabled = False
        self.show_pdf_button.description = "Show PDF/PMF"
        self.show_shaded_region_flag = False
        self.show_shaded_region_button.disabled = False
        self.show_shaded_region_button.description = "Display Shaded Region"
        # Reset bounds interaction flag - histogram will be all blue until user interacts
        self.bounds_interacted = False
        # Reset status
        self.status_html.value = "Ready to draw samples."
        self._update_bound_sliders()
        self._show_blank_plot()
        
    def _on_reset_clicked(self, button):
        """Handle Reset all button click - reset everything to initial state"""
        # Clear samples
        self.samples = np.array([])
        
        # Reset PDF flag and button (still available without samples)
        self.show_pdf_flag = False
        self.show_pdf_button.disabled = False
        self.show_pdf_button.description = "Show PDF/PMF"
        
        # Reset shaded region flag and button
        self.show_shaded_region_flag = False
        self.show_shaded_region_button.disabled = False
        self.show_shaded_region_button.description = "Display Shaded Region"
        
        # Reset bounds interaction flag
        self.bounds_interacted = False
        
        # Reset probability type
        self.prob_type_dropdown.value = ""
        self._update_slider_visibility()
        self._update_bound_sliders()
        
        # Reset probability label
        self.prob_label.value = '<div style="font-size: 18px; padding: 10px; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><b>Estimated Probability:</b> <span style="color: #0066cc; font-size: 20px; font-weight: bold;">N/A</span><br><b>True Probability:</b> <span style="color: #cc6600; font-size: 20px; font-weight: bold;">N/A</span></div>'
        
        # Reset status
        self.status_html.value = "Ready to draw samples."
        
        # Show blank plot
        self._show_blank_plot()
        
    def _on_show_pdf_clicked(self, button):
        """Handle Show PDF/PMF button click (works with or without samples)"""
        self.show_pdf_flag = not self.show_pdf_flag
        if self.show_pdf_flag:
            self.show_pdf_button.description = "Hide PDF/PMF"
            self._update_bound_sliders()
            self._update_plot()
        else:
            self.show_pdf_button.description = "Show PDF/PMF"
            if len(self.samples) > 0:
                self._update_plot()
            else:
                self._show_blank_plot()
    
    def _on_show_shaded_region_clicked(self, button):
        """Handle Show Shaded Region button click"""
        if len(self.samples) > 0 or self.show_pdf_flag:
            self.show_shaded_region_flag = not self.show_shaded_region_flag
            if self.show_shaded_region_flag:
                self.show_shaded_region_button.description = "Hide Shaded Region"
            else:
                self.show_shaded_region_button.description = "Display Shaded Region"
            self._update_plot()
        
    def _update_param_widgets(self):
        """Update parameter widgets based on current distribution"""
        dist = self.dist_dropdown.value
        if dist in self.param_widgets:
            self.param_container.children = tuple(self.param_widgets[dist])
        # Bin count only applies to continuous histograms
        if self.category_dropdown.value == "Continuous":
            self.n_bins_slider.layout.display = None
        else:
            self.n_bins_slider.layout.display = 'none'
        # Don't update plot here to avoid double updates
        
    def _update_slider_visibility(self):
        """Update which sliders are visible based on probability type"""
        prob_type = self.prob_type_dropdown.value
        
        if prob_type == "":
            # Empty selection - hide sliders
            self.slider_container.children = ()
        elif prob_type == "of outcome":
            # Only show bound1 (the outcome value)
            self.slider_container.children = (self.bound1_slider,)
            self.bound1_slider.description = "Outcome:"
        elif prob_type == "under upper bound":
            # Only show bound2 (upper bound)
            self.slider_container.children = (self.bound2_slider,)
            self.bound2_slider.description = "Upper bound:"
        elif prob_type == "above lower bound":
            # Only show bound1 (lower bound)
            self.slider_container.children = (self.bound1_slider,)
            self.bound1_slider.description = "Lower bound:"
        elif prob_type == "in interval":
            # Show both bounds
            self.slider_container.children = (self.bound1_slider, self.bound2_slider)
            self.bound1_slider.description = "Lower bound:"
            self.bound2_slider.description = "Upper bound:"
    
    def _on_prob_type_change(self, change):
        """Handle probability type change"""
        self._update_slider_visibility()  # Update which sliders are shown
        # If empty, set bounds off-screen and hide shaded region
        if self.prob_type_dropdown.value == "":
            # Set bounds to very negative and very positive values to keep them off-screen
            self.bound1_slider.value = -1e10
            self.bound2_slider.value = 1e10
            # Hide shaded region when no prob_type is selected
            self.show_shaded_region_flag = False
        else:
            # When changing to a non-empty type, automatically show shaded region
            self.show_shaded_region_flag = True
            # When changing to a non-empty type, reset bounds to default range
            # Check if previous value was empty (changing from empty to non-empty)
            if change.get('old') == "":
                # Reset bounds_interacted so bounds get set to default range
                self.bounds_interacted = False
            # Update bound sliders to set proper ranges and default values
            # This will set bounds to full range (x_min to x_max) for the distribution
            self._update_bound_sliders(reset_to_full_range=True)
        if len(self.samples) > 0 or self.show_pdf_flag:
            self._update_plot()
        
    def _on_draw_clicked(self, button):
        """Handle draw samples button with progressive animation"""
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        n_total = self.n_samples_slider.value
        
        # Get parameters
        params = self._get_params_dict()
        
        # Generate all samples at once
        all_samples = sample_distribution(dist_type, dist_category, n_total, **params)
        
        # Progressive visualization
        self.status_html.value = "Generating samples..."
        
        # Preserve PDF visibility if the student already turned it on
        keep_pdf = self.show_pdf_flag
        
        sample_index = 0
        batch_count = 0
        while sample_index < n_total:
            batch_size = determine_batch_size(sample_index)
            end_index = min(sample_index + batch_size, n_total)
            
            # Get samples up to current index
            self.samples = all_samples[:end_index]
            
            # Update bound sliders on first batch
            if sample_index == 0:
                # If prob_type is not empty, reset bounds to full range for the distribution
                if self.prob_type_dropdown.value != "":
                    self.bounds_interacted = False
                    self._update_bound_sliders(reset_to_full_range=True)
                else:
                    self._update_bound_sliders()
                self.show_pdf_button.disabled = False
                self.show_pdf_flag = keep_pdf
                self.show_pdf_button.description = "Hide PDF/PMF" if keep_pdf else "Show PDF/PMF"
                # Automatically show shaded region if prob_type is selected
                if self.prob_type_dropdown.value != "":
                    self.show_shaded_region_flag = True
                else:
                    self.show_shaded_region_flag = False
                # Update slider visibility based on current prob_type
                self._update_slider_visibility()
            
            # Update plot less frequently to speed up animation
            # Update every batch for first 100 samples, then every 2 batches
            should_update_plot = (sample_index < 100) or (batch_count % 2 == 0)
            
            if should_update_plot:
                self._update_plot()
            
            # Always update status
            self.status_html.value = f"Generated {end_index} / {n_total} samples"
            
            # Constant speed: every 1000 samples takes 2 seconds
            # Delay is proportional to batch size
            delay = batch_size / 500.0
            time.sleep(delay)
            
            sample_index = end_index
            batch_count += 1
        
        # Final update with all samples
        self.samples = all_samples
        self._update_bound_sliders()  # Update sliders to match full sample range
        self._update_plot()
        self.status_html.value = f"Complete! Generated {n_total} samples."
    
    def _get_theoretical_x_range(self, params=None):
        """Return a reasonable x-range covering most of the current distribution's mass."""
        dist_type = self.dist_dropdown.value
        if params is None:
            params = self._get_params_dict()

        if dist_type == "Uniform":
            low = params.get('low', 0)
            high = params.get('high', 1)
            return float(min(low, high)), float(max(low, high))
        if dist_type == "Exponential":
            scale = params.get('scale', 1)
            return 0.0, float(max(scale * 5, 1))
        if dist_type == "Pareto":
            scale = params.get('scale', 1.0)
            return float(scale), float(scale + 5.0)
        if dist_type == "Beta":
            return 0.0, 1.0
        if dist_type == "Gamma":
            shape = params.get('shape', 2)
            scale = params.get('scale', 1)
            return 0.0, float(max(shape * scale * 5, 1))
        if dist_type == "Normal":
            mean = params.get('mean', 0)
            std = max(params.get('std', 1), 1e-6)
            return float(mean - 4 * std), float(mean + 4 * std)
        if dist_type == "Bernoulli":
            # Leave one outcome-width on either side of 0 and 1 so the
            # histogram and PMF bars are fully visible.
            return -1, 2
        if dist_type == "Geometric":
            p = max(params.get('p', 0.5), 0.01)
            return 1, int(max(15, stats.geom.ppf(0.99, p)))
        if dist_type == "Binomial":
            return 0, int(params.get('n', 10))
        if dist_type == "Poisson":
            lam = params.get('lam', 5)
            return 0, int(max(lam * 3, lam + 4 * np.sqrt(lam) + 1, 5))
        if dist_type == "Hypergeometric":
            ngood = int(params.get('ngood', 10))
            nsample = int(params.get('nsample', 10))
            return 0, min(nsample, ngood)
        return -5.0, 5.0

    def _get_x_axis_range(self):
        """Get the x-axis range for the plot (used by both plot and sliders).

        When PDF/PMF is shown, the range is the union of the sample extent and the
        current theoretical support so neither leaves the window after parameter changes.
        """
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        params = self._get_params_dict()
        theo_min, theo_max = self._get_theoretical_x_range(params)

        if len(self.samples) > 0:
            sample_min = float(np.min(self.samples))
            sample_max = float(np.max(self.samples))
            if self.show_pdf_flag:
                # Fit both samples and the updated theoretical PDF/PMF
                pad = 1.0 if dist_category == "Discrete" else 0.5
                x_min = min(sample_min, theo_min) - pad
                x_max = max(sample_max, theo_max) + pad
            elif dist_type == "Pareto":
                x_min = theo_min
                x_max = theo_max
            else:
                pad = 1.0 if dist_category == "Discrete" else 1.0
                x_min = sample_min - pad
                x_max = sample_max + pad
        else:
            x_min, x_max = theo_min, theo_max

        if dist_category == "Discrete":
            x_min = max(int(np.floor(x_min)), 0) if dist_type != "Bernoulli" else 0
            if dist_type == "Bernoulli":
                x_min, x_max = -1, 2
            elif dist_type == "Binomial":
                x_min = 0
                x_max = int(max(x_max, params.get('n', 10)))
            else:
                x_max = int(np.ceil(x_max))
                if not self.show_pdf_flag and dist_type not in ["Poisson", "Bernoulli", "Binomial"]:
                    range_padding = max(int((x_max - x_min) * 0.2), 3)
                    x_max = x_max + range_padding

        # Exponential support starts at zero; do not pad into negative values.
        if dist_category == "Continuous" and dist_type == "Exponential":
            x_min = 0.0

        if x_min >= x_max:
            x_max = x_min + 1

        return x_min, x_max

    def _continuous_histogram_span(self, params=None):
        """Return (low, high) used for equal-width continuous histogram bins."""
        if params is None:
            params = self._get_params_dict()
        dist_type = self.dist_dropdown.value

        if dist_type == "Uniform":
            low = params.get('low', 0)
            high = params.get('high', 1)
            a, b = (float(low), float(high)) if low < high else (float(high), float(low))
            if a >= b:
                b = a + 1e-6
            return a, b

        if dist_type == "Beta":
            return 0.0, 1.0

        smin = float(np.min(self.samples))
        smax = float(np.max(self.samples))
        if smin >= smax:
            smax = smin + 1e-6
        return smin, smax

    def _continuous_histogram_bin_edges(self, params=None):
        """Equal-width bin edges from the bins slider."""
        a, b = self._continuous_histogram_span(params)
        n_bins = max(1, int(self.n_bins_slider.value))
        return np.linspace(a, b, n_bins + 1)

    def _continuous_histogram(self, params=None):
        """Density histogram with equal-width bins and correct outer edges."""
        bin_edges = self._continuous_histogram_bin_edges(params)
        counts, bin_edges = np.histogram(self.samples, bins=bin_edges, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = float(bin_edges[1] - bin_edges[0])
        return counts, bin_edges, bin_centers, bin_width
        
    def _update_bound_sliders(self, reset_to_full_range=False):
        """Update bound slider ranges to match the plot's x-axis range"""
        # If prob_type is empty, set bounds off-screen and return early
        if self.prob_type_dropdown.value == "":
            # Set bounds to very negative and very positive values to keep them off-screen
            # Expand range first to avoid conflicts
            wide_min = min(self.bound1_slider.min, self.bound2_slider.min, -1e10)
            wide_max = max(self.bound1_slider.max, self.bound2_slider.max, 1e10)
            self.bound1_slider.max = wide_max
            self.bound2_slider.max = wide_max
            self.bound1_slider.min = wide_min
            self.bound2_slider.min = wide_min
            self.bound1_slider.value = -1e10
            self.bound2_slider.value = 1e10
            return
        
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        
        x_min, x_max = self._get_x_axis_range()
        
        # Ensure valid range
        if x_min >= x_max:
            x_max = x_min + 1
        
        # Determine step size
        if dist_category == "Discrete":
            step = 1
        else:
            step = 0.1
        
        # If bounds haven't been interacted with or reset requested, set to full range
        # To avoid TraitError when updating slider min/max, we need to handle the order carefully
        # First, expand the range to accommodate all possible values, then set the final range
        
        # Temporarily expand the range to a very wide interval to avoid conflicts
        wide_min = min(x_min, self.bound1_slider.min, self.bound2_slider.min, -1e10)
        wide_max = max(x_max, self.bound1_slider.max, self.bound2_slider.max, 1e10)
        
        # Set to wide range first (order: max first, then min to avoid min > max)
        self.bound1_slider.max = wide_max
        self.bound2_slider.max = wide_max
        self.bound1_slider.min = wide_min
        self.bound2_slider.min = wide_min
        
        if reset_to_full_range or not self.bounds_interacted:
            # Set to full range so entire histogram is blue (all selected)
            self.bound1_slider.value = x_min
            self.bound2_slider.value = x_max
        else:
            # Clamp slider values to new range
            self.bound1_slider.value = max(x_min, min(x_max, self.bound1_slider.value))
            self.bound2_slider.value = max(x_min, min(x_max, self.bound2_slider.value))
        
        # Now set the actual min, max, and step (order: min first since we're shrinking from wide range)
        self.bound1_slider.min = x_min
        self.bound1_slider.max = x_max
        self.bound1_slider.step = step
        
        self.bound2_slider.min = x_min
        self.bound2_slider.max = x_max
        self.bound2_slider.step = step
        
    def _draw_samples(self):
        """Draw new samples"""
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        n_samples = self.n_samples_slider.value
        
        # Get parameters
        params = {}
        if dist_type in self.param_widgets:
            widgets_list = self.param_widgets[dist_type]
            if dist_type == "Uniform":
                params['low'] = widgets_list[0].value
                params['high'] = widgets_list[1].value
            elif dist_type == "Exponential":
                params['scale'] = widgets_list[0].value
            elif dist_type == "Pareto":
                params['shape'] = widgets_list[0].value
                params['scale'] = widgets_list[1].value
            elif dist_type == "Beta":
                params['alpha'] = widgets_list[0].value
                params['beta'] = widgets_list[1].value
            elif dist_type == "Gamma":
                params['shape'] = widgets_list[0].value
                params['scale'] = widgets_list[1].value
            elif dist_type == "Normal":
                params['mean'] = widgets_list[0].value
                params['std'] = widgets_list[1].value
            elif dist_type == "Bernoulli":
                params['p'] = widgets_list[0].value
            elif dist_type == "Geometric":
                params['p'] = widgets_list[0].value
            elif dist_type == "Binomial":
                params['n'] = widgets_list[0].value
                params['p'] = widgets_list[1].value
            elif dist_type == "Poisson":
                params['lam'] = widgets_list[0].value
            elif dist_type == "Hypergeometric":
                params['ngood'] = widgets_list[0].value
                params['nbad'] = widgets_list[1].value
                params['nsample'] = widgets_list[2].value
        
        self.samples = sample_distribution(dist_type, dist_category, n_samples, **params)
        
    def _get_params_dict(self):
        """Get current parameters as dictionary"""
        dist_type = self.dist_dropdown.value
        params = {}
        if dist_type in self.param_widgets:
            widgets_list = self.param_widgets[dist_type]
            if dist_type == "Uniform":
                params['low'] = widgets_list[0].value
                params['high'] = widgets_list[1].value
            elif dist_type == "Exponential":
                params['scale'] = widgets_list[0].value
            elif dist_type == "Pareto":
                params['shape'] = widgets_list[0].value
                params['scale'] = widgets_list[1].value
            elif dist_type == "Beta":
                params['alpha'] = widgets_list[0].value
                params['beta'] = widgets_list[1].value
            elif dist_type == "Gamma":
                params['shape'] = widgets_list[0].value
                params['scale'] = widgets_list[1].value
            elif dist_type == "Normal":
                params['mean'] = widgets_list[0].value
                params['std'] = widgets_list[1].value
            elif dist_type == "Bernoulli":
                params['p'] = widgets_list[0].value
            elif dist_type == "Geometric":
                params['p'] = widgets_list[0].value
            elif dist_type == "Binomial":
                params['n'] = widgets_list[0].value
                params['p'] = widgets_list[1].value
            elif dist_type == "Poisson":
                params['lam'] = widgets_list[0].value
            elif dist_type == "Hypergeometric":
                params['ngood'] = widgets_list[0].value
                params['nbad'] = widgets_list[1].value
                params['nsample'] = widgets_list[2].value
        return params
        
    def _show_blank_plot(self):
        """Show blank plot when distribution changes"""
        with self.plot_output:
            clear_output(wait=True)
            # Create empty figure (single plot)
            fig = go.Figure()
            # Set default axis ranges for blank plot from current distribution support
            y_title = "P(X = x)" if self.category_dropdown.value == "Discrete" else "Density"
            x_min, x_max = self._get_theoretical_x_range()
            if x_min >= x_max:
                x_max = x_min + 1
            fig.update_xaxes(title_text="x", range=[x_min, x_max])
            fig.update_yaxes(title_text=y_title, range=[0, 1])
            fig.update_layout(height=600, showlegend=True, title="Select Show PDF/PMF or Draw Samples")
            fig.show()
            
            # Reset probability label
            self.prob_label.value = '<div style="font-size: 18px; padding: 10px; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><b>Estimated Probability:</b> <span style="color: #0066cc; font-size: 20px; font-weight: bold;">N/A</span><br><b>True Probability:</b> <span style="color: #cc6600; font-size: 20px; font-weight: bold;">N/A</span></div>'
    
    def _update_plot(self, change=None):
        """Update the plot with histogram and/or overlaid PDF/PMF"""
        # Allow PDF/PMF-only view before any samples are drawn
        if len(self.samples) == 0 and not self.show_pdf_flag:
            self._show_blank_plot()
            return
        
        # Use instance flags for display options
        show_pdf = self.show_pdf_flag
        show_shaded_region = self.show_shaded_region_flag
            
        with self.plot_output:
            clear_output(wait=True)
            
            dist_type = self.dist_dropdown.value
            dist_category = self.category_dropdown.value
            prob_type = self.prob_type_dropdown.value
            params = self._get_params_dict()
            
            # Determine bounds based on probability type
            bound1 = self.bound1_slider.value
            bound2 = self.bound2_slider.value
            
            # Create single figure (no subplots)
            fig = go.Figure()
            
            # Determine x range (use same method as sliders to ensure they match)
            x_min, x_max = self._get_x_axis_range()
            
            # For discrete, ensure we cover integer values
            if dist_category == "Discrete":
                x_range = np.arange(int(x_min), int(x_max) + 1)
            else:
                x_range = np.linspace(x_min, x_max, 500)
            
            # Compute PDF/PMF if needed (will overlay on histogram)
            pdf_pmf_values = None
            if show_pdf:
                pdf_pmf_values = compute_pdf_pmf(x_range, dist_type, dist_category, **params)
            
            # Create histogram first (as base layer)
            if len(self.samples) > 0:
                if dist_category == "Discrete":
                    # For discrete, use integer bins
                    unique_vals, counts = np.unique(self.samples, return_counts=True)
                    counts = counts / len(self.samples)  # Normalize to probability
                    
                    # Only show red highlighting when shaded region is shown and prob_type is not empty
                    if show_shaded_region and prob_type != "":
                        # Determine which values are in the selected region (inclusive bounds)
                        if prob_type == "of outcome":
                            selected_mask = unique_vals == int(np.round(bound1))
                        elif prob_type == "under upper bound":
                            selected_mask = unique_vals <= bound2  # Inclusive upper bound
                        elif prob_type == "above lower bound":
                            selected_mask = unique_vals >= bound1  # Inclusive lower bound
                        elif prob_type == "in interval":
                            selected_mask = (unique_vals >= bound1) & (unique_vals <= bound2)  # Inclusive both bounds
                        else:
                            selected_mask = np.zeros(len(unique_vals), dtype=bool)
                        
                        # Create color array: red for selected, blue for not selected
                        hist_colors = ['rgba(255,0,0,0.7)' if sel else 'rgba(70,130,180,0.6)' for sel in selected_mask]
                        hist_line_colors = ['darkred' if sel else 'navy' for sel in selected_mask]
                        hist_line_widths = [2 if sel else 1 for sel in selected_mask]
                    else:
                        # All blue when shaded region is not shown
                        hist_colors = 'rgba(70,130,180,0.6)'
                        hist_line_colors = 'navy'
                        hist_line_widths = 1
                    
                    # Plot histogram with color-coded bars (single trace with array of colors)
                    fig.add_trace(go.Bar(
                        x=unique_vals,
                        y=counts,
                        name='Histogram of Samples',
                        marker=dict(
                            color=hist_colors,
                            line=dict(color=hist_line_colors, width=hist_line_widths)
                        ),
                        showlegend=True,
                        opacity=0.7
                    ))
                else:
                    # Equal-width bins on the distribution support (not the padded axis range),
                    # so Uniform endpoints are not artificially shorter.
                    counts, bin_edges, bin_centers, bin_width = self._continuous_histogram(params)
                    fig.add_trace(go.Bar(
                        x=bin_centers,
                        y=counts,
                        name='Histogram of Samples',
                        marker=dict(color='rgba(70,130,180,0.6)', line=dict(color='navy', width=1)),
                        showlegend=True,
                        width=bin_width * 0.9,
                        opacity=0.7
                    ))
                    
                    # Only shade histogram for probability region when shaded region is shown and prob_type is not empty (inclusive bounds)
                    if show_shaded_region and prob_type != "":
                        if prob_type == "under upper bound":
                            mask = bin_centers <= bound2  # Inclusive upper bound
                            if np.any(mask):
                                fig.add_trace(go.Bar(
                                    x=bin_centers[mask],
                                    y=counts[mask],
                                    name='Selected Samples',
                                    marker=dict(color='rgba(255,0,0,0.5)', line=dict(color='red', width=2)),
                                    showlegend=False,
                                    width=bin_width * 0.9,
                                    opacity=0.7
                                ))
                        elif prob_type == "above lower bound":
                            mask = bin_centers >= bound1  # Inclusive lower bound
                            if np.any(mask):
                                fig.add_trace(go.Bar(
                                    x=bin_centers[mask],
                                    y=counts[mask],
                                    name='Selected Samples',
                                    marker=dict(color='rgba(255,0,0,0.5)', line=dict(color='red', width=2)),
                                    showlegend=False,
                                    width=bin_width * 0.9,
                                    opacity=0.7
                                ))
                        elif prob_type == "in interval":
                            mask = (bin_centers >= bound1) & (bin_centers <= bound2)  # Inclusive both bounds
                            if np.any(mask):
                                fig.add_trace(go.Bar(
                                    x=bin_centers[mask],
                                    y=counts[mask],
                                    name='Selected Samples',
                                    marker=dict(color='rgba(255,0,0,0.5)', line=dict(color='red', width=2)),
                                    showlegend=False,
                                    width=bin_width * 0.9,
                                    opacity=0.7
                                ))
            
            # Overlay PDF/PMF on top if show_pdf is True
            if show_pdf and pdf_pmf_values is not None:
                # Plot PDF/PMF overlay
                if dist_category == "Discrete":
                    # Determine which PMF values are in the selected region (inclusive bounds) - only if shaded region is shown and prob_type is not empty
                    if show_shaded_region and prob_type != "":
                        if prob_type == "of outcome":
                            pmf_selected_mask = x_range == int(np.round(bound1))
                        elif prob_type == "under upper bound":
                            pmf_selected_mask = x_range <= bound2  # Inclusive upper bound
                        elif prob_type == "above lower bound":
                            pmf_selected_mask = x_range >= bound1  # Inclusive lower bound
                        elif prob_type == "in interval":
                            pmf_selected_mask = (x_range >= bound1) & (x_range <= bound2)  # Inclusive both bounds
                        else:
                            pmf_selected_mask = np.zeros(len(x_range), dtype=bool)
                        
                        # Create color array: red for selected, orange for not selected
                        pmf_colors = ['rgba(255,0,0,0.8)' if sel else 'rgba(255,165,0,0.8)' for sel in pmf_selected_mask]
                        pmf_line_colors = ['darkred' if sel else 'orange' for sel in pmf_selected_mask]
                        pmf_line_widths = [3 if sel else 2 for sel in pmf_selected_mask]
                    else:
                        # All orange when shaded region is not shown
                        pmf_colors = 'rgba(255,165,0,0.8)'
                        pmf_line_colors = 'orange'
                        pmf_line_widths = 2
                    
                    # For discrete, plot as bars with color-coding (red for selected region, orange otherwise)
                    fig.add_trace(go.Bar(
                        x=x_range,
                        y=pdf_pmf_values,
                        name='PMF',
                        marker=dict(
                            color=pmf_colors,
                            line=dict(color=pmf_line_colors, width=pmf_line_widths)
                        ),
                        showlegend=True,
                        opacity=0.8
                    ))
                else:
                    # For continuous, plot as line (orange)
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=pdf_pmf_values,
                        mode='lines',
                        name='PDF',
                        line=dict(color='orange', width=3),
                        showlegend=True
                    ))
            
            # Add shaded area and vertical lines only if shaded region is shown and prob_type is not empty
            if show_shaded_region and prob_type != "":
                # Determine which region to shade based on prob_type (inclusive bounds)
                if prob_type == "of outcome":
                    shade_x = [bound1, bound1]
                    if pdf_pmf_values is not None:
                        shade_y = [0, np.max(pdf_pmf_values) * 1.1]
                    else:
                        shade_y = [0, 1]
                elif prob_type == "under upper bound":
                    mask = x_range <= bound2  # Inclusive upper bound
                    shade_x = x_range[mask]
                    if pdf_pmf_values is not None:
                        shade_y = pdf_pmf_values[mask]
                    else:
                        shade_y = None
                elif prob_type == "above lower bound":
                    mask = x_range >= bound1  # Inclusive lower bound
                    shade_x = x_range[mask]
                    if pdf_pmf_values is not None:
                        shade_y = pdf_pmf_values[mask]
                    else:
                        shade_y = None
                elif prob_type == "in interval":
                    mask = (x_range >= bound1) & (x_range <= bound2)  # Inclusive both bounds
                    shade_x = x_range[mask]
                    if pdf_pmf_values is not None:
                        shade_y = pdf_pmf_values[mask]
                    else:
                        shade_y = None
                
                # Calculate max y value for vertical lines (from both histogram and PDF/PMF)
                max_hist = 0
                if len(self.samples) > 0:
                    if dist_category == "Discrete":
                        unique_vals, hist_counts = np.unique(self.samples, return_counts=True)
                        hist_counts = hist_counts / len(self.samples)
                        max_hist = np.max(hist_counts) if len(hist_counts) > 0 else 0
                    else:
                        hist_counts, _, _, _ = self._continuous_histogram(params)
                        max_hist = np.max(hist_counts) if len(hist_counts) > 0 else 0
                max_pdf = np.max(pdf_pmf_values) if pdf_pmf_values is not None and len(pdf_pmf_values) > 0 else 0
                max_y = max(max_hist, max_pdf, 0.1) * 1.1
                
                # Add vertical lines for bounds and shaded area for continuous PDF
                if prob_type == "of outcome":
                    # Vertical line for outcome
                    fig.add_trace(go.Scatter(
                        x=[bound1, bound1],
                        y=[0, max_y],
                        mode='lines',
                        name='Bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
                elif dist_category == "Continuous" and show_pdf and shade_y is not None and len(shade_x) > 0:
                    # For continuous PDF, add filled area under PDF curve
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([[shade_x[0]], shade_x, [shade_x[-1]]]),
                        y=np.concatenate([[0], shade_y, [0]]),
                        fill='tozeroy',
                        mode='lines',
                        name='Selected PDF Region',
                        line=dict(color='rgba(255,0,0,0.4)', width=2),
                        fillcolor='rgba(255,0,0,0.3)',
                        showlegend=False
                    ))
                
                # Add vertical lines for bounds
                if prob_type == "under upper bound":
                    fig.add_trace(go.Scatter(
                        x=[bound2, bound2],
                        y=[0, max_y],
                        mode='lines',
                        name='Upper bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
                elif prob_type == "above lower bound":
                    fig.add_trace(go.Scatter(
                        x=[bound1, bound1],
                        y=[0, max_y],
                        mode='lines',
                        name='Lower bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
                elif prob_type == "in interval":
                    fig.add_trace(go.Scatter(
                        x=[bound1, bound1],
                        y=[0, max_y],
                        mode='lines',
                        name='Lower bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=[bound2, bound2],
                        y=[0, max_y],
                        mode='lines',
                        name='Upper bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
            
            # Update layout (single plot): fit both samples and PDF/PMF in the window
            y_title = "P(X = x)" if dist_category == "Discrete" else "Density"

            max_hist = 0
            if len(self.samples) > 0:
                if dist_category == "Discrete":
                    _, hist_counts = np.unique(self.samples, return_counts=True)
                    hist_counts = hist_counts / len(self.samples)
                    max_hist = float(np.max(hist_counts)) if len(hist_counts) > 0 else 0
                else:
                    hist_counts, _, _, _ = self._continuous_histogram(params)
                    max_hist = float(np.max(hist_counts)) if len(hist_counts) > 0 else 0
            max_pdf = float(np.max(pdf_pmf_values)) if pdf_pmf_values is not None and len(pdf_pmf_values) > 0 else 0
            y_max = max(max_hist, max_pdf, 0.1) * 1.15

            fig.update_xaxes(title_text="x", range=[x_min, x_max])
            fig.update_yaxes(title_text=y_title, range=[0, y_max])
            if len(self.samples) > 0 and show_pdf:
                plot_title = "Histogram of Samples and PDF/PMF"
            elif len(self.samples) > 0:
                plot_title = "Histogram of Samples"
            else:
                plot_title = "PDF/PMF"
            fig.update_layout(height=600, showlegend=True, title=plot_title)
            
            # Compute probabilities (only if prob_type is not empty)
            if prob_type == "":
                # Show N/A when no interval type is selected
                self.prob_label.value = (
                    '<div style="font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                    '<b>Estimated Probability:</b> <span style="color: #999; font-size: 16px;">N/A (select an interval type above)</span><br><br>'
                    '<b>True Probability:</b> <span style="color: #999; font-size: 16px;">N/A (select an interval type above)</span>'
                    '</div>'
                )
            else:
                has_samples = len(self.samples) > 0
                est_text = (
                    f'{compute_estimated_probability(self.samples, prob_type, bound1, bound2):.4f}'
                    if has_samples else 'N/A (draw samples to estimate)'
                )
                est_style = (
                    'color: #0066cc; font-size: 22px; font-weight: bold; background-color: white; padding: 4px 8px; border-radius: 4px;'
                    if has_samples else 'color: #999; font-size: 16px;'
                )
                if show_pdf:
                    true_prob = compute_true_probability(dist_type, dist_category, prob_type, bound1, bound2, **params)
                    self.prob_label.value = (
                        f'<div style="font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                        f'<b>Estimated Probability (from samples):</b> <span style="{est_style}">{est_text}</span><br><br>'
                        f'<b>True Probability (from {("CDF" if dist_category == "Continuous" else "PMF")}):</b> <span style="color: #cc6600; font-size: 22px; font-weight: bold; background-color: white; padding: 4px 8px; border-radius: 4px;">{true_prob:.4f}</span>'
                        f'</div>'
                    )
                else:
                    self.prob_label.value = (
                        f'<div style="font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                        f'<b>Estimated Probability (from samples):</b> <span style="{est_style}">{est_text}</span><br><br>'
                        f'<b>True Probability:</b> <span style="color: #999; font-size: 16px;">N/A (click "Show PDF/PMF" to see comparison)</span>'
                        f'</div>'
                    )
            
            fig.show()
            
    def display(self):
        """Display the complete interface"""
        # Update parameter widgets initially
        self._update_param_widgets()
        
        # A distribution-specific explorer starts with its fixed PDF/PMF;
        # the general explorer preserves its blank initial plot.
        if self.show_pdf_flag:
            self._update_plot()
        else:
            self._show_blank_plot()
        
        # Create main layout
        controls = widgets.VBox([
            self.category_dropdown,
            self.dist_dropdown,
            self.param_container,
            self.n_samples_slider,
            self.n_bins_slider,
            widgets.HBox([self.draw_button, self.reset_button]),  # Buttons side by side
            self.status_html,  # Status display for animation progress
            self.prob_controls_container  # Probability controls (available before sampling)
        ])
        
        display(widgets.HBox([controls, self.plot_output]))


def run_distribution_explorer(dist_type=None):
    """
    Create and display the interactive distribution visualization.

    Pass a supported distribution name (for example, ``"Bernoulli"`` or
    ``"Normal"``) to lock the PDF/PMF explorer to that distribution while
    retaining its parameter, sampling, and probability controls.

    Parameters
    ----------
    dist_type : str or None
        Optional distribution name. When supplied, the type and distribution
        dropdowns are locked to that distribution.
    """
    viz = DistributionProbabilityVisualization(
        initial_dist_type=dist_type,
        lock_distribution=dist_type is not None,
    )
    viz.display()
    return viz
