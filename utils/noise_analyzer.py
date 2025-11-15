import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import glob
import time
from scipy import ndimage


class NoiseAnalyzer:
    def __init__(self, img1_path, img2_path):
        """Initialize the noise analyzer with two images."""
        # Load source images
        self.img1 = self._load_grayscale_image(img1_path)
        self.img2 = self._load_grayscale_image(img2_path)
        
        # Extract noise patterns
        self.noise1 = self._extract_noise(self.img1)
        self.noise2 = self._extract_noise(self.img2)
        
        # Performance optimization for 144fps
        self._zoomed_padded_cache = None
        self._last_event_time = 0.0
        self._target_fps = 144
        self._frame_time = 1.0 / self._target_fps
        self._last_mouse_pos = (0, 0)

        # Dynamic overlay state
        self._current_axis = None
        self._current_circles = None
        self._current_zoom_im = None
        self._current_noise = None
        self._zoom_components = None
        self.backgrounds = {}

        # Precompute reusable arrays
        self.zoom_radius = 150
        self.zoom_factor = 3
        self.extract_radius = self.zoom_radius // self.zoom_factor
        self.display_size = self.zoom_radius * 2
        
        # Precompute circular mask once
        self._circular_mask = self._create_circular_mask(self.zoom_radius)
        
        # Precompute zoom grid for fast interpolation
        self._zoom_grid_y, self._zoom_grid_x = np.mgrid[
            -self.extract_radius:self.extract_radius:complex(0, self.display_size),
            -self.extract_radius:self.extract_radius:complex(0, self.display_size)
        ]
        
        print(f"Image 1: {img1_path} ({self.img1.shape})")
        print(f"Image 2: {img2_path} ({self.img2.shape})")
        print(f"Target FPS: {self._target_fps}")
    
    def _load_grayscale_image(self, path):
        """Load an image as grayscale numpy array."""
        return np.array(Image.open(path).convert('L'))
    
    def _extract_noise(self, img):
        """Extract high-frequency noise using Gaussian filter."""
        lowpass = ndimage.gaussian_filter(img.astype(float), sigma=2)
        return img.astype(float) - lowpass
    
    def _setup_plot(self):
        """Set up the matplotlib figure and axes."""
        # Colormap with transparent NaNs for circular zoom
        cmap = plt.cm.seismic.copy()
        cmap.set_bad(alpha=0.0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle('Forensic Noise Analysis', fontsize=16)
        
        return fig, ax1, ax2, cmap
    
    def _create_zoom_circles(self, ax):
        """Create zoom indicator circles for an axis."""
        outer_circle = Circle(
            (0, 0), self.zoom_radius, 
            fill=False, color='white', linewidth=3, 
            visible=False, zorder=12
        )
        inner_circle = Circle(
            (0, 0), self.zoom_radius,
            fill=False, color='black', linewidth=1,
            visible=False, zorder=13
        )
        
        ax.add_patch(outer_circle)
        ax.add_patch(inner_circle)
        
        return outer_circle, inner_circle
    
    def _create_zoom_overlay(self, ax, cmap):
        """Create the zoom image overlay for an axis."""
        size = self.display_size
        return ax.imshow(
            np.full((size, size), np.nan),
            extent=[0, 1, 0, 1],
            cmap=cmap,
            vmin=-50,
            vmax=50,
            visible=False,
            zorder=11,
            interpolation='nearest',
            alpha=0.9
        )
    
    def _create_circular_mask(self, radius):
        """Create a circular boolean mask for zoom overlay."""
        y_grid, x_grid = np.ogrid[-radius:radius, -radius:radius]
        return x_grid**2 + y_grid**2 <= radius**2
    
    def _setup_event_handlers(self, fig, ax1, ax2, components):
        """Set up optimized mouse event handlers for 144fps."""
        circle1_outer, circle1_inner, zoom_im1 = components['ax1']
        circle2_outer, circle2_inner, zoom_im2 = components['ax2']
        
        # Track which axis is currently active
        self._current_axis = None
        self._current_circles = None
        self._current_zoom_im = None
        self._current_noise = None
        
        def on_motion(event):
            """High-performance mouse movement handler."""
            current_time = time.time()
            
            # Frame rate limiting - only process if enough time has passed
            if current_time - self._last_event_time < self._frame_time:
                return
            
            self._last_event_time = current_time
            
            # Quick exit if no valid mouse position
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                self._hide_all_zoom_elements()
                return
            
            x, y = int(event.xdata), int(event.ydata)
            
            # Only update if mouse actually moved significantly
            if abs(x - self._last_mouse_pos[0]) < 2 and abs(y - self._last_mouse_pos[1]) < 2:
                return
            
            self._last_mouse_pos = (x, y)
            
            # Determine which axis and corresponding noise to use
            if event.inaxes == ax1:
                if self._current_axis != 'ax1':
                    self._hide_all_zoom_elements()
                    self._current_axis = 'ax1'
                    self._current_circles = (circle1_outer, circle1_inner)
                    self._current_zoom_im = zoom_im1
                    self._current_noise = self.noise1
            elif event.inaxes == ax2:
                if self._current_axis != 'ax2':
                    self._hide_all_zoom_elements()
                    self._current_axis = 'ax2'
                    self._current_circles = (circle2_outer, circle2_inner)
                    self._current_zoom_im = zoom_im2
                    self._current_noise = self.noise2
            else:
                self._hide_all_zoom_elements()
                return
            
            # Update the zoom elements for current axis
            self._update_zoom_elements_optimized(x, y)
            
            # Use blitting for maximum performance
            fig.canvas.restore_region(self.backgrounds[self._current_axis])
            
            # Draw only the dynamic elements
            outer_circle, inner_circle = self._current_circles
            ax = ax1 if self._current_axis == 'ax1' else ax2
            ax.draw_artist(outer_circle)
            ax.draw_artist(inner_circle)
            ax.draw_artist(self._current_zoom_im)
            
            # Blit only the affected axis
            fig.canvas.blit(ax.bbox)
        
        def on_leave(event):
            """Hide zoom elements when mouse leaves figure."""
            self._hide_all_zoom_elements()
            fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('figure_leave_event', on_leave)
    
    def _hide_all_zoom_elements(self):
        """Hide all zoom elements and restore a clean background."""
        # Hide all dynamic artists if they've been created
        zoom_components = getattr(self, "_zoom_components", None)
        if zoom_components is not None:
            for outer, inner, zoom_im in zoom_components.values():
                outer.set_visible(False)
                inner.set_visible(False)
                zoom_im.set_visible(False)

        # Reset current state
        self._current_axis = None
        self._current_circles = None
        self._current_zoom_im = None
        self._current_noise = None

        # If we have cached backgrounds, restore both axes
        if hasattr(self, "backgrounds") and hasattr(self, "fig"):
            canvas = self.fig.canvas
            for name, ax in (("ax1", self.ax1), ("ax2", self.ax2)):
                if name in self.backgrounds:
                    canvas.restore_region(self.backgrounds[name])
                    canvas.blit(ax.bbox)
    def _update_zoom_elements_optimized(self, x, y):
        """Ultra-fast update of zoom elements."""
        if self._current_circles is None or self._current_zoom_im is None:
            return
            
        outer_circle, inner_circle = self._current_circles
        
        # Update circle positions
        outer_circle.center = (x, y)
        inner_circle.center = (x, y)
        outer_circle.set_visible(True)
        inner_circle.set_visible(True)
        
        # Update zoom overlay with optimized method
        self._update_zoom_circle_ultrafast(self._current_zoom_im, self._current_noise, x, y)
        self._current_zoom_im.set_visible(True)
    
    def _update_zoom_circle_ultrafast(self, zoom_im, noise, x, y):
        """Ultra-fast zoom circle update using precomputed grids and vectorized operations."""
        height, width = noise.shape
        
        # Boundary checks (most efficient form)
        y_min = max(0, y - self.extract_radius)
        y_max = min(height, y + self.extract_radius)
        x_min = max(0, x - self.extract_radius)
        x_max = min(width, x + self.extract_radius)
        
        if y_min >= y_max or x_min >= x_max:
            return
        
        # Extract region (this is the fastest way)
        region = noise[y_min:y_max, x_min:x_max]
        region_height, region_width = region.shape
        
        # Calculate offsets for centering
        offset_y = (self.display_size - region_height * self.zoom_factor) // 2
        offset_x = (self.display_size - region_width * self.zoom_factor) // 2
        
        # Initialize or reuse display array
        if self._zoomed_padded_cache is None:
            self._zoomed_padded_cache = np.full((self.display_size, self.display_size), np.nan)
        else:
            # Faster than fill() for large arrays
            self._zoomed_padded_cache[:] = np.nan
        
        # Vectorized zoom using ndimage.zoom for highest performance
        zoomed = ndimage.zoom(region, self.zoom_factor, order=1)
        
        # Bounds for placement
        end_y = offset_y + zoomed.shape[0]
        end_x = offset_x + zoomed.shape[1]
        
        if end_y <= self.display_size and end_x <= self.display_size:
            self._zoomed_padded_cache[offset_y:end_y, offset_x:end_x] = zoomed
        
        # Apply circular mask
        self._zoomed_padded_cache[~self._circular_mask] = np.nan
        
        # Update image data
        zoom_im.set_data(self._zoomed_padded_cache)
        zoom_im.set_extent([x - self.zoom_radius, x + self.zoom_radius, 
                           y + self.zoom_radius, y - self.zoom_radius])
    
    def _cache_backgrounds(self, fig, ax1, ax2):
        """Cache the background for blitting optimization."""
        # Draw everything once to initialize
        fig.canvas.draw()
        
        # Cache backgrounds for both axes
        self.backgrounds = {
            'ax1': fig.canvas.copy_from_bbox(ax1.bbox),
            'ax2': fig.canvas.copy_from_bbox(ax2.bbox)
        }
        
        print("Backgrounds cached for 144fps blitting optimization")
    
    def analyze(self):
        """Run the complete noise analysis with 144fps-optimized visualization."""
        # Setup plot components
        self.fig, self.ax1, self.ax2, cmap = self._setup_plot()
        
        # Display noise maps
        im1 = self.ax1.imshow(self.noise1, cmap=cmap, vmin=-50, vmax=50)
        self.ax1.set_title('Image 1 - Noise Pattern')
        self.ax1.axis('off')
        plt.colorbar(im1, ax=self.ax1, fraction=0.046)
        
        im2 = self.ax2.imshow(self.noise2, cmap=cmap, vmin=-50, vmax=50)
        self.ax2.set_title('Image 2 - Noise Pattern')
        self.ax2.axis('off')
        plt.colorbar(im2, ax=self.ax2, fraction=0.046)
        
        # Create zoom components
        circle1_outer, circle1_inner = self._create_zoom_circles(self.ax1)
        circle2_outer, circle2_inner = self._create_zoom_circles(self.ax2)
        
        zoom_im1 = self._create_zoom_overlay(self.ax1, cmap)
        zoom_im2 = self._create_zoom_overlay(self.ax2, cmap)
        
        # Freeze axis limits to prevent auto-scaling
        for ax, im in [(self.ax1, im1), (self.ax2, im2)]:
            extent = im.get_extent()
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_autoscale_on(False)
        
        # Setup interactive components
        components = {
            'ax1': (circle1_outer, circle1_inner, zoom_im1),
            'ax2': (circle2_outer, circle2_inner, zoom_im2)
        }

        # Keep a reference to all zoom components for quick hiding
        self._zoom_components = components

        # Finalize layout BEFORE caching backgrounds so bboxes are correct
        plt.tight_layout()
        
        # Cache backgrounds AFTER all static elements and layout are finalized
        self._cache_backgrounds(self.fig, self.ax1, self.ax2)
        
        # Setup high-performance event handlers
        self._setup_event_handlers(self.fig, self.ax1, self.ax2, components)
        
        plt.show()


def find_image_files():
    """Find up to 2 image files in current directory."""
    image_extensions = [
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"
    ]
    
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(ext))
        images.extend(glob.glob(ext.upper()))
    
    return sorted(set(images))[:2]


if __name__ == "__main__":
    images = find_image_files()
    
    if len(images) < 2:
        print("Error: Need at least 2 images in current directory")
        exit(1)
    
    print(f"Analyzing images: {images[0]} vs {images[1]}")
    analyzer = NoiseAnalyzer(images[0], images[1])
    analyzer.analyze()
