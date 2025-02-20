import lightkurve as lk
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import multiprocessing as mp
from functools import partial
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib.widgets import RectangleSelector


def centroid_analysis(tpf, clipped_lc, ui):
    # Extract the flux-weighted centroid positions
    centroid_col, centroid_row = tpf.estimate_centroids()

    # Convert to numpy arrays and handle any potential NaN values
    centroid_col = np.nan_to_num(getattr(centroid_col, 'value', centroid_col))
    centroid_row = np.nan_to_num(getattr(centroid_row, 'value', centroid_row))
    
    # Remove extreme outliers (more than 5 sigma from median)
    for centroid in [centroid_col, centroid_row]:
        median = np.median(centroid)
        mad = np.median(np.abs(centroid - median))
        mask = np.abs(centroid - median) < 5 * mad * 1.4826  # 1.4826 converts MAD to sigma
        centroid[~mask] = median

    # Get time and flux arrays
    clipped_lc_time = clipped_lc.time.value
    clipped_lc_flux = clipped_lc.flux.value

    # Ensure all arrays are the same length by truncating to shortest
    min_length = min(len(clipped_lc_time), len(centroid_col), len(centroid_row))
    centroid_col = centroid_col[:min_length]
    centroid_row = centroid_row[:min_length]
    clipped_lc_time = clipped_lc_time[:min_length]
    clipped_lc_flux = clipped_lc_flux[:min_length]

    # Plot centroid
    ui.plot_centroid(clipped_lc_time, centroid_col, centroid_row)
    plt.close()  # Close the figure to prevent memory issues

    # Compare centroid movement during transit vs non-transit times
    median_flux = np.median(clipped_lc_flux)
    std_flux = np.std(clipped_lc_flux)
    in_transit = clipped_lc_flux < (median_flux - 2 * std_flux)

    if np.sum(in_transit) > 0:
        in_transit_centroid_col = centroid_col[in_transit]
        in_transit_centroid_row = centroid_row[in_transit]

        # Calculate shifts during transit, handling potential NaN values
        shift_col = np.nanstd(in_transit_centroid_col)
        shift_row = np.nanstd(in_transit_centroid_row)

        if not (np.isnan(shift_col) or np.isnan(shift_row)):
            print(f"Potential centroid shift during transit: Std Dev Col = {shift_col:.4f} pixels, Std Dev Row = {shift_row:.4f} pixels")
            if shift_col > 1.0 or shift_row > 1.0:  # More than 1 pixel shift is suspicious
                print("Warning: Unusually large centroid shift detected - possible outliers")
        else:
            print("Warning: Unable to calculate centroid shift due to invalid values")


def calculate_snr(folded_lc, planet_period):
    try:
        # Use a wider window for transit detection
        transit_duration = 0.2 * planet_period
        in_transit = (np.abs(folded_lc.time.value) < transit_duration)
        out_of_transit = ~in_transit

        # Require minimum number of points for reliable calculation
        min_points = 20
        if np.sum(in_transit) < min_points or np.sum(out_of_transit) < min_points:
            print(f"Warning: Insufficient points for reliable SNR calculation (minimum {min_points} required)")
            return None

        # Handle masked arrays explicitly to avoid partition warning
        in_transit_flux = np.ma.filled(folded_lc.flux[in_transit], fill_value=np.nan)
        out_transit_flux = np.ma.filled(folded_lc.flux[out_of_transit], fill_value=np.nan)
        
        # Remove NaN values before calculations
        in_transit_flux = in_transit_flux[~np.isnan(in_transit_flux)]
        out_transit_flux = out_transit_flux[~np.isnan(out_transit_flux)]
        
        # Use robust statistics for better noise estimation
        transit_depth = np.median(in_transit_flux)
        baseline = np.median(out_transit_flux)
        
        # Use median absolute deviation for noise estimation
        noise = np.median(np.abs(out_transit_flux - baseline)) * 1.4826  # Scale factor for MAD
        
        # Calculate SNR using relative depth
        depth = np.abs(baseline - transit_depth)
        snr = depth / noise if noise > 0 else 0
        
        # Print detailed diagnostics
        print(f"\nTransit Analysis Results:")
        print(f"Number of in-transit points: {len(in_transit_flux)}")
        print(f"Number of out-of-transit points: {len(out_transit_flux)}")
        print(f"Transit Depth: {depth:.6f}")
        print(f"Baseline Flux: {baseline:.6f}")
        print(f"Noise Level: {noise:.6f}")
        print(f"Signal-to-Noise Ratio (SNR): {snr:.2f}")

        if snr < 7:
            print("Warning: Low SNR, transit signal may not be reliable.")
            if snr < 1:
                print("Signal is likely noise - consider skipping this period.")
        
        return snr

    except Exception as e:
        print(f"Error calculating SNR: {e}")
        return None


# Function to search for and download all TPF files using TIC ID
def search_and_download_tpf(tic_id):
    try:
        search_result = lk.search_targetpixelfile(f"TIC {tic_id}")
        if len(search_result) == 0:
            print(f"No TPF files found for TIC {tic_id}.")
            return None

        # Initialize list to store TPF objects
        tpfs = []
        
        # Get the default download path (usually ~/.lightkurve-cache)
        download_dir = os.path.join(os.path.expanduser('~'), '.lightkurve\cache')
        
        # Create directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        # Process each search result
        for idx, result in enumerate(search_result):
            # Get mission info
            mission_info = result.mission[0]  # Get the first mission info
            
            # Extract sector from the table
            sector = result.table['sequence_number'][0]  # TESS sector number
            
            # Construct expected local path
            filename = f"tess-s{int(sector):04d}-{tic_id}-tpf.fits"
            expected_path = os.path.join(download_dir, 'mastDownload', 'TESS', filename)
            
            print(f"\nFile {idx + 1}/{len(search_result)}: {filename}")
            print(f"Sector: {sector}, Mission: {mission_info}")
            
            if os.path.exists(expected_path):
                print(f"Using existing local file: {expected_path}")
                tpf = lk.TessTargetPixelFile(expected_path)
                tpfs.append(tpf)
            else:
                print(f"Downloading: {filename}")
                try:
                    # Download with progress bar
                    tpf = result.download(
                        cutout_size=None,
                        quality_bitmask='default',
                        download_dir=download_dir,
                        verbose=True
                    )
                    if tpf is not None:
                        print(f"Saved to: {tpf.path}")
                        tpfs.append(tpf)
                    else:
                        print(f"Failed to download: {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    continue

        if len(tpfs) == 0:
            print(f"No files downloaded or found locally for TIC {tic_id}.")
            return None

        print(f"\nTotal files processed: {len(tpfs)}")
        
        # Print summary of all files
        print("\nSummary of all files:")
        for tpf in tpfs:
            print(f"- {tpf.path}")

        return tpfs

    except Exception as e:
        print(f"Error searching for TIC {tic_id}: {e}")
        return None


class TESSAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TESS Data Analyzer")
        
        # Set minimum window size
        self.root.minsize(800, 600)
        
        # Initialize dragging state
        self.is_dragging = False
        self.rect_artist = None
        
        # Create main container
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create top frame for input controls
        self.input_frame = ttk.Frame(self.main_container)
        self.input_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Add file input controls
        self.setup_file_input()
        
        # Add TIC search controls
        self.setup_tic_search()
        
        # Create status label
        self.status_label = ttk.Label(self.main_container, text="Ready")
        self.status_label.pack(fill=tk.X, pady=(0, 5))
        
        # Create notebook with fixed height ratio
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Initialize tabs
        self.setup_tabs()
        
        # Initialize canvases and outputs
        self.initialize_canvases()

    def setup_tabs(self):
        # Create tabs with proper frame management
        self.lightcurve_tab = ttk.Frame(self.notebook)
        self.ttv_tab = ttk.Frame(self.notebook)
        self.centroid_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.lightcurve_tab, text="Light Curve")
        self.notebook.add(self.ttv_tab, text="TTV Analysis")
        self.notebook.add(self.centroid_tab, text="Centroid Analysis")
        
        # Setup each tab with proper frame management
        self.setup_tab(self.lightcurve_tab, "lightcurve")
        self.setup_tab(self.ttv_tab, "ttv")
        self.setup_tab(self.centroid_tab, "centroid")

    def setup_tab(self, tab, name):
        # Create main container frame for the tab
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create plot frame that will expand
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        setattr(self, f'{name}_frame', plot_frame)
        
        # Create navigation frame with fixed height at bottom
        nav_frame = ttk.Frame(main_frame, height=40)  # Reduced fixed height
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))
        nav_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Create button frame with grid for centering
        button_frame = ttk.Frame(nav_frame)
        button_frame.place(relx=0.5, rely=0.5, anchor='center')  # Center in nav_frame
        
        # New ordering: Previous, Next, Output label, Reset View.
        prev_button = ttk.Button(button_frame, text="Previous", 
                                 command=lambda: self.show_previous_output(name))
        prev_button.pack(side=tk.LEFT, padx=5)
        setattr(self, f'{name}_prev_button', prev_button)
        
        next_button = ttk.Button(button_frame, text="Next", 
                                 command=lambda: self.show_next_output(name))
        next_button.pack(side=tk.LEFT, padx=5)
        setattr(self, f'{name}_next_button', next_button)
        
        output_label = ttk.Label(button_frame, text="Output 0 of 0")
        output_label.pack(side=tk.LEFT, padx=10)
        setattr(self, f'{name}_nav_label', output_label)
        
        reset_button = ttk.Button(button_frame, text="Reset View",
                                  command=self.reset_view)
        reset_button.pack(side=tk.LEFT, padx=5)
        setattr(self, f'{name}_reset_button', reset_button)

    def setup_file_input(self):
        # Add file load controls
        ttk.Label(self.input_frame, text="Load Local File:").grid(row=0, column=0, padx=5, pady=5)
        self.file_entry = ttk.Entry(self.input_frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)
        self.browse_button = ttk.Button(self.input_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        self.load_button = ttk.Button(self.input_frame, text="Load", command=self.load_local_file)
        self.load_button.grid(row=0, column=3, padx=5, pady=5)

    def setup_tic_search(self):
        # Add TIC ID controls
        ttk.Label(self.input_frame, text="Search by TIC ID:").grid(row=1, column=0, padx=5, pady=5)
        self.tic_entry = ttk.Entry(self.input_frame, width=40)
        self.tic_entry.grid(row=1, column=1, padx=5, pady=5)
        self.search_button = ttk.Button(self.input_frame, text="Search & Download", command=self.search_tic)
        self.search_button.grid(row=1, column=2, columnspan=2, padx=5, pady=5)

    def browse_file(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select TESS Target Pixel File",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def load_local_file(self):
        file_path = self.file_entry.get()
        if not file_path:
            self.status_label.config(text="Please select a file first", foreground="red")
            return
        
        # Clear previous plots before loading new data
        self.clear_all_plots()
        
        self.status_label.config(text="Processing file...", foreground="blue")
        try:
            tpf = lk.TessTargetPixelFile(file_path)
            result = process_single_tpf_path(tpf.path)
            if result is not None:
                self.plot_results(result)
                self.status_label.config(text="File processed successfully", foreground="green")
            else:
                self.status_label.config(text="File processing failed", foreground="red")
        except Exception as e:
            self.status_label.config(text=f"Error processing file: {str(e)}", foreground="red")

    def search_tic(self):
        tic_id = self.tic_entry.get()
        if not tic_id:
            self.status_label.config(text="Please enter a TIC ID", foreground="red")
            return
        
        # Clear previous plots before loading new data
        self.clear_all_plots()
        
        self.status_label.config(text="Searching and downloading data...", foreground="blue")
        try:
            tpfs = search_and_download_tpf(tic_id)
            if tpfs is not None:
                # Process files in parallel
                results = process_tess_files_parallel(tpfs)
                
                # Plot results in the UI thread
                for result in results:
                    self.plot_results(result)
                
                self.status_label.config(
                    text=f"Processed {len(results)} files successfully", 
                    foreground="green"
                )
            else:
                self.status_label.config(text="No files found for this TIC ID", foreground="red")
        except Exception as e:
            self.status_label.config(text=f"Error processing TIC ID: {str(e)}", foreground="red")

    def show_previous_output(self, tab_name):
        if self.current_output_index[tab_name] > 0:
            self.current_output_index[tab_name] -= 1
            self.update_output_display(tab_name)

    def show_next_output(self, tab_name):
        if self.current_output_index[tab_name] < len(self.current_outputs[tab_name]) - 1:
            self.current_output_index[tab_name] += 1
            self.update_output_display(tab_name)

    def update_output_display(self, tab_name):
        index = self.current_output_index[tab_name]
        outputs = self.current_outputs[tab_name]
        
        if outputs:
            # Get the current figure and display it
            fig = outputs[index]
            canvas = getattr(self, f'{tab_name}_canvas')
            
            if canvas:
                canvas.get_tk_widget().destroy()
            
            new_canvas = FigureCanvasTkAgg(fig, master=getattr(self, f'{tab_name}_frame'))
            new_canvas.draw()
            new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            setattr(self, f'{tab_name}_canvas', new_canvas)
            
            # Update navigation label
            label = getattr(self, f'{tab_name}_nav_label')
            label.config(text=f"Output {index + 1} of {len(outputs)}")
            
            # Update button states
            prev_button = getattr(self, f'{tab_name}_prev_button')
            next_button = getattr(self, f'{tab_name}_next_button')
            prev_button.config(state='normal' if index > 0 else 'disabled')
            next_button.config(state='normal' if index < len(outputs) - 1 else 'disabled')

    def plot_lightcurve(self, folded_lc, moving_avg):
        if self.lightcurve_canvas:
            self.lightcurve_canvas.get_tk_widget().destroy()
        
        # Create figure with adjusted size
        self.current_fig = plt.figure(figsize=(10, 6))
        ax = self.current_fig.add_subplot(111)
        
        # Plot data
        folded_lc.scatter(ax=ax, c='black', s=2)
        ax.plot(folded_lc.time.value, moving_avg, color='red', label='Smoothed Moving Average')
        
        # Set title and labels
        ax.set_title('Phase Folded Light Curve')
        ax.set_xlabel('Phase [days]')
        ax.set_ylabel('Normalized Flux')
        ax.legend()
        
        # Store original view limits
        self.original_xlim = ax.get_xlim()
        self.original_ylim = ax.get_ylim()
        
        # Create and pack canvas
        self.lightcurve_canvas = FigureCanvasTkAgg(self.current_fig, master=self.lightcurve_frame)
        self.lightcurve_canvas.draw()
        self.lightcurve_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Store the data and enable zoom
        self.current_folded_lc = folded_lc
        self.enable_zoom_selection(folded_lc)
        
        # Connect mouse events
        self.lightcurve_canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.lightcurve_canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.lightcurve_canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        
        # Update display
        self.current_outputs['lightcurve'].append(self.current_fig)
        self.update_output_display('lightcurve')

    def enable_zoom_selection(self, folded_lc):
        """Enable interactive zoom selection"""
        # Create the selector with updated parameters
        ax = self.current_fig.gca()
        self.selector = RectangleSelector(
            ax, self.zoom_selection_callback,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor='red', edgecolor='red', alpha=0.2, fill=True)
        )
        
        # Store the light curve data for the callback
        self.zoom_folded_lc = folded_lc
        
        # Make sure the selector is active
        self.selector.set_active(True)

    def on_mouse_press(self, event):
        if event.inaxes:
            self.is_dragging = True
            self.drag_start = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.inaxes and self.is_dragging:
            self.is_dragging = False
            if hasattr(self, 'drag_start'):
                end_point = (event.xdata, event.ydata)
                # Clear any existing rectangle
                if hasattr(self, 'rect_artist') and self.rect_artist is not None:
                    self.rect_artist.remove()
                    self.rect_artist = None
                # Perform zoom and recalculation
                self.zoom_selection_callback(self.drag_start, end_point)

    def on_mouse_motion(self, event):
        if event.inaxes and self.is_dragging:
            ax = event.inaxes
            
            # Remove existing rectangle if it exists
            if hasattr(self, 'rect_artist') and self.rect_artist is not None:
                self.rect_artist.remove()
            
            # Draw new rectangle
            x0, y0 = self.drag_start
            x1, y1 = event.xdata, event.ydata
            self.rect_artist = plt.Rectangle(
                (min(x0, x1), min(y0, y1)), 
                abs(x1 - x0), abs(y1 - y0),
                fill=True, fc='red', ec='red', alpha=0.2
            )
            ax.add_patch(self.rect_artist)
            self.lightcurve_canvas.draw()

    def zoom_selection_callback(self, eclick, erelease):
        """Handle zoom selection"""
        try:
            if isinstance(eclick, tuple):
                x1, y1 = eclick
                x2, y2 = erelease
            else:
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
            
            # Get the selected region bounds
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            
            # Filter data to selected region
            mask = (self.zoom_folded_lc.time.value >= xmin) & \
                   (self.zoom_folded_lc.time.value <= xmax) & \
                   (self.zoom_folded_lc.flux.value >= ymin) & \
                   (self.zoom_folded_lc.flux.value <= ymax)
            
            zoomed_lc = self.zoom_folded_lc[mask]
            
            # Check if we have enough points
            if len(zoomed_lc) < 20:
                self.status_label.config(text="Selected region too small. Need at least 20 points.", foreground="red")
                return
            
            # Create new figure and axis
            plt.close(self.current_fig)
            self.current_fig = plt.figure(figsize=(10, 6))
            ax = self.current_fig.add_subplot(111)
            
            # Recalculate analysis for zoomed region
            window_size = max(20, len(zoomed_lc) // 10)  # Adjust window size for zoomed data
            df = pd.DataFrame({'flux': zoomed_lc.flux})
            moving_avg = df['flux'].rolling(window=window_size, center=True).mean()
            
            # Plot zoomed data with new calculations
            zoomed_lc.scatter(ax=ax, c='black', s=2, label='Data')
            ax.plot(zoomed_lc.time.value, moving_avg, color='red', label='Smoothed Moving Average')
            
            # Update labels and title
            ax.set_title('Phase Folded Light Curve (Zoomed)')
            ax.set_xlabel('Phase [days]')
            ax.set_ylabel('Normalized Flux')
            ax.legend()
            
            # Set the zoom limits with a small margin
            margin = 0.1  # 10% margin
            x_range = xmax - xmin
            y_range = ymax - ymin
            ax.set_xlim(xmin - margin * x_range, xmax + margin * x_range)
            ax.set_ylim(ymin - margin * y_range, ymax + margin * y_range)
            
            # Recalculate SNR for zoomed region
            snr = calculate_snr(zoomed_lc, self.zoom_folded_lc.period.value)
            if snr is not None:
                ax.set_title(f'Phase Folded Light Curve (Zoomed) - SNR: {snr:.2f}')
                self.status_label.config(text=f"Zoomed region analyzed. SNR: {snr:.2f}", foreground="green")
            
            # Update the canvas while preserving button frame
            if self.lightcurve_canvas:
                self.lightcurve_canvas.get_tk_widget().destroy()
            
            self.lightcurve_canvas = FigureCanvasTkAgg(self.current_fig, master=self.lightcurve_frame)
            self.lightcurve_canvas.draw()
            self.lightcurve_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Store the zoomed light curve for further zooming
            self.zoom_folded_lc = zoomed_lc
            
            # Re-enable zoom selection for further zooming
            self.enable_zoom_selection(zoomed_lc)
            
            # Connect the canvas to mouse events
            self.lightcurve_canvas.mpl_connect('button_press_event', self.on_mouse_press)
            self.lightcurve_canvas.mpl_connect('button_release_event', self.on_mouse_release)
            self.lightcurve_canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
            
        except Exception as e:
            print(f"Error in zoom selection: {e}")
            self.status_label.config(text=f"Error in zoom selection: {str(e)}", foreground="red")

    def reset_view(self):
        """Reset to original view"""
        if hasattr(self, 'original_xlim') and hasattr(self, 'original_ylim'):
            ax = self.current_fig.gca()
            ax.set_xlim(self.original_xlim)
            ax.set_ylim(self.original_ylim)
            
            # Replot original data
            ax.clear()
            self.current_folded_lc.scatter(ax=ax, c='black', s=2)
            
            # Recalculate moving average for full dataset
            df = pd.DataFrame({'flux': self.current_folded_lc.flux})
            moving_avg = df['flux'].rolling(window=200, center=True).mean()
            ax.plot(self.current_folded_lc.time.value, moving_avg, color='red', 
                    label='Smoothed Moving Average')
            
            # Reset title and labels
            ax.set_title('Phase Folded Light Curve')
            ax.set_xlabel('Phase [days]')
            ax.set_ylabel('Normalized Flux')
            ax.legend()
            
            self.lightcurve_canvas.draw()
            self.status_label.config(text="View reset to original", foreground="green")

    def plot_ttv(self, folded_lc, period):
        if self.ttv_canvas:
            self.ttv_canvas.get_tk_widget().pack_forget()
        
        # Create figure with adjusted size and margins
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        
        # Plot data
        ax.plot(folded_lc.time.value, folded_lc.flux.value, 'o-', color='blue', markerfacecolor='red')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title('Transit Light Curve')
        ax.grid(True)
        
        # Adjust layout to fit in window
        fig.tight_layout(pad=1.5)
        
        # Create and pack canvas
        self.ttv_canvas = FigureCanvasTkAgg(fig, master=self.ttv_frame)
        self.ttv_canvas.draw()
        self.ttv_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Store the figure and update display
        self.current_outputs['ttv'].append(fig)
        self.update_output_display('ttv')
        plt.close(fig)

    def plot_centroid(self, tpf, folded_lc):
        if self.centroid_canvas:
            self.centroid_canvas.get_tk_widget().pack_forget()
        
        # Create figure with adjusted size and margins
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        
        # Plot data
        ax.plot(folded_lc.time.value, folded_lc.flux.value, label='Transit Light Curve')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title('Transit Light Curve')
        ax.legend()
        
        # Adjust layout to fit in window
        fig.tight_layout(pad=1.5)
        
        # Create and pack canvas
        self.centroid_canvas = FigureCanvasTkAgg(fig, master=self.centroid_frame)
        self.centroid_canvas.draw()
        self.centroid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Store the figure and update display
        self.current_outputs['centroid'].append(fig)
        self.update_output_display('centroid')
        plt.close(fig)

    def plot_results(self, result):
        if result is not None:
            # Load TPF for plotting when needed
            tpf = lk.TessTargetPixelFile(result['tpf_path'])
            
            self.plot_lightcurve(result['folded_lc'], result['moving_avg'])
            self.plot_ttv(result['folded_lc'], result['period'])
            self.plot_centroid(tpf, result['folded_lc'])

    def clear_all_plots(self):
        # Clear all stored outputs
        self.current_outputs = {
            'lightcurve': [],
            'ttv': [],
            'centroid': []
        }
        self.current_output_index = {
            'lightcurve': 0,
            'ttv': 0,
            'centroid': 0
        }
        
        # Clear all canvases
        if self.lightcurve_canvas:
            self.lightcurve_canvas.get_tk_widget().destroy()
            self.lightcurve_canvas = None
        if self.ttv_canvas:
            self.ttv_canvas.get_tk_widget().destroy()
            self.ttv_canvas = None
        if self.centroid_canvas:
            self.centroid_canvas.get_tk_widget().destroy()
            self.centroid_canvas = None
            
        # Reset navigation labels
        for tab_name in ['lightcurve', 'ttv', 'centroid']:
            label = getattr(self, f'{tab_name}_nav_label')
            label.config(text="Output 0 of 0")
            
            # Reset button states
            prev_button = getattr(self, f'{tab_name}_prev_button')
            next_button = getattr(self, f'{tab_name}_next_button')
            prev_button.config(state='disabled')
            next_button.config(state='disabled')

    def add_analysis_features(self):
        """
        Add new analysis features to UI
        """
        # Add new analysis tab
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Advanced Analysis")
        
        # Add parameter display
        self.param_frame = ttk.LabelFrame(self.analysis_tab, text="Transit Parameters")
        self.param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add false positive indicators
        self.fp_frame = ttk.LabelFrame(self.analysis_tab, text="False Positive Checks")
        self.fp_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add statistical results
        self.stats_frame = ttk.LabelFrame(self.analysis_tab, text="Statistical Analysis")
        self.stats_frame.pack(fill=tk.X, padx=5, pady=5)

    def initialize_canvases(self):
        """Initialize canvas attributes and output tracking"""
        # Initialize canvas objects
        self.lightcurve_canvas = None
        self.ttv_canvas = None
        self.centroid_canvas = None
        
        # Initialize output tracking
        self.current_outputs = {
            'lightcurve': [],
            'ttv': [],
            'centroid': []
        }
        self.current_output_index = {
            'lightcurve': 0,
            'ttv': 0,
            'centroid': 0
        }


def process_tess_files_parallel(tpfs, use_tls=True):
    try:
        # Reduce number of cores for more stable processing
        num_cores = min(8, mp.cpu_count() - 16)
        print(f"\nUsing {num_cores} cores for processing")
        print(f"Total files to process: {len(tpfs)}")
        
        # Convert TPFs to paths for multiprocessing
        tpf_paths = [tpf.path for tpf in tpfs]
        
        # Pass the use_tls flag to each processing call
        with mp.Pool(num_cores) as pool:
            results = pool.map(partial(process_single_tpf_path, use_tls=use_tls), tpf_paths)
            
        successful = [r for r in results if r is not None]
        failed = len(results) - len(successful)
        
        print(f"\nProcessing Summary:")
        print(f"Total files: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {failed}")
        
        return successful
        
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        return []

def process_single_tpf_path(tpf_path, use_tls=True):
    try:
        print(f"\nProcessing: {tpf_path}")
        # Load TPF from path
        tpf = lk.TessTargetPixelFile(tpf_path)
        plt.close('all')
        
        # Generate the light curve using the pipeline mask
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
        print("Generated light curve")

        # Flatten the light curve to remove long-term trends
        flat_lc = lc.flatten()
        print("Flattened light curve")

        # Perform sigma clipping to remove outliers
        clipped_lc = flat_lc[sigma_clip(flat_lc.flux, sigma=5).mask == False]
        print("Performed sigma clipping")

        # Compute moving average
        window_size = 200
        df = pd.DataFrame({'flux': clipped_lc.flux})
        moving_avg = df['flux'].rolling(window=window_size, center=True).mean()
        print("Computed moving average")

        # Search for transits using BLS
        period = np.linspace(0.5, 10, 10000)
        bls = clipped_lc.to_periodogram(method='bls', period=period, frequency_factor=500)
        print("Completed BLS search")

        # Extract period information
        planet_x_period = bls.period_at_max_power
        planet_x_t0 = bls.transit_time_at_max_power.value

        # Fold the light curve
        folded_lc = clipped_lc.fold(period=planet_x_period, epoch_time=planet_x_t0)
        print("Folded light curve")
        
        # Check SNR before continuing
        snr = calculate_snr(folded_lc, planet_x_period.value)
        print(f"SNR calculated: {snr}")
        
        # Remove SNR filtering temporarily to see all results
        # if snr is None or snr < 1.0:
        #     print(f"Skipping {tpf_path} due to low SNR")
        #     return None

        # Return the processed data without the TPF object
        return {
            'folded_lc': folded_lc,
            'moving_avg': moving_avg,
            'period': float(planet_x_period.value),
            'epoch': float(planet_x_t0),
            'tpf_path': tpf_path,
            'snr': snr
        }

    except Exception as e:
        print(f"Error processing {tpf_path}: {str(e)}")
        return None
    finally:
        plt.close('all')


def enhanced_transit_detection(folded_lc, period):
    """
    Improved transit detection with multiple methods
    """
    # BLS with multiple period ranges
    period_ranges = [
        (0.5, 5),    # Short period planets
        (5, 15),     # Medium period planets
        (15, 30)     # Longer period planets
    ]
    
    results = []
    for p_min, p_max in period_ranges:
        period = np.linspace(p_min, p_max, 10000)
        bls = folded_lc.to_periodogram(method='bls', period=period, frequency_factor=500)
        results.append({
            'period': bls.period_at_max_power,
            'power': bls.max_power,
            'range': (p_min, p_max)
        })
    
    return results


if __name__ == "__main__":
    root = tk.Tk()
    ui = TESSAnalyzerUI(root)
    root.mainloop()
