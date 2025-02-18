import lightkurve as lk
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to process the TESS Target Pixel File (TPF) for a given TIC ID and plot the phase-folded light curve
def process_tess_file(tpf_path, ui):
    try:
        # Load the TESS Target Pixel File (TPF)
        tpf = lk.TessTargetPixelFile(tpf_path)

        # Generate the light curve using the pipeline mask
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)

        # Flatten the light curve to remove long-term trends
        flat_lc = lc.flatten()

        # Perform sigma clipping to remove outliers (3-sigma is typical)
        clipped_lc = flat_lc[sigma_clip(flat_lc.flux, sigma=5).mask == False]

        # Compute a moving average of the flux values using pandas' rolling method
        window_size = 200  # Larger window size for smoother average (adjust this value for desired smoothness)
        df = pd.DataFrame({'flux': clipped_lc.flux})
        moving_avg = df['flux'].rolling(window=window_size, center=True).mean()

        # Search for transits using BLS with a fixed period range
        period = np.linspace(0.5, 10, 10000)  # Define a fixed period range (e.g., 0.5 to 10 days)
        bls = clipped_lc.to_periodogram(method='bls', period=period, frequency_factor=500)

        # Extract the period of the most prominent transit
        planet_x_period = bls.period_at_max_power
        print(f"Period at max power: {planet_x_period}")
        planet_x_t0 = bls.transit_time_at_max_power.value

        # Call before phase-folding
        centroid_analysis(tpf, clipped_lc, ui)

        # Phase-fold the light curve
        folded_lc = clipped_lc.fold(period=planet_x_period, epoch_time=planet_x_t0)

        # Call after phase-folding the light curve
        odd_even_transit_analysis(folded_lc, planet_x_period.value)

        # Call after phase-folding the light curve
        calculate_snr(folded_lc, planet_x_period.value)

        # Call after identifying the transit period and folding the light curve
        transit_timing_variation_analysis(folded_lc, planet_x_period.value, ui)

        # Instead of plt.show(), use the UI to plot
        ui.plot_lightcurve(folded_lc, moving_avg)

        return folded_lc

    except Exception as e:
        print(f"Error processing {tpf_path}: {e}")

def odd_even_transit_analysis(folded_lc, period):
    odd_transits = folded_lc[(folded_lc.time.value % (2 * period)) < period]
    even_transits = folded_lc[(folded_lc.time.value % (2 * period)) >= period]

    # Calculate median depth of odd and even transits
    odd_depth = np.median(odd_transits.flux)
    even_depth = np.median(even_transits.flux)

    print(f"Odd Transit Depth: {odd_depth}")
    print(f"Even Transit Depth: {even_depth}")

    if np.abs(odd_depth - even_depth) > 0.01:  # Example threshold
        print("Warning: Significant difference between odd and even transit depths. Possible false positive.")


def centroid_analysis(tpf, clipped_lc, ui):
    # Estimate centroid positions using the flux-weighted column and row for each frame in the TPF
    # Extract the flux-weighted centroid column and row positions
    centroid_col, centroid_row = tpf.estimate_centroids()

    # Convert centroid positions to unitless values
    if hasattr(centroid_col, 'value'):
        centroid_col = centroid_col.value
    if hasattr(centroid_row, 'value'):
        centroid_row = centroid_row.value

    # Convert time and flux of clipped light curve to plain (unitless) values
    clipped_lc_time = clipped_lc.time.value
    clipped_lc_flux = clipped_lc.flux.value

    # Ensure all relevant arrays have the same length
    if len(clipped_lc_time) != len(centroid_col):
        print("Warning: Mismatch in centroid and clipped light curve lengths.")
        min_length = min(len(clipped_lc_time), len(centroid_col))
        centroid_col = centroid_col[:min_length]
        centroid_row = centroid_row[:min_length]
        clipped_lc_time = clipped_lc_time[:min_length]
        clipped_lc_flux = clipped_lc_flux[:min_length]

    # Instead of plt.show(), use the UI to plot
    ui.plot_centroid(clipped_lc_time, centroid_col, centroid_row)

    # Compare centroid movement during transit vs non-transit times
    # Convert in-transit condition to a dimensionless comparison
    median_flux = np.median(clipped_lc_flux)
    std_flux = np.std(clipped_lc_flux)
    in_transit = clipped_lc_flux < (median_flux - 2 * std_flux)

    # Align in-transit mask length with centroid data length
    in_transit = in_transit[:len(centroid_col)]

    if np.sum(in_transit) > 0:
        # Determine the centroid positions during the transit
        in_transit_centroid_col = centroid_col[in_transit]
        in_transit_centroid_row = centroid_row[in_transit]

        # Calculate shifts during transit
        shift_col = np.std(in_transit_centroid_col)
        shift_row = np.std(in_transit_centroid_row)

        print(
            f"Potential centroid shift during transit: Std Dev Col = {shift_col:.4f} pixels, Std Dev Row = {shift_row:.4f} pixels")
        if shift_col > 0.1 or shift_row > 0.1:  # Threshold value for potential shift
            print(
                "Warning: Significant centroid shift detected during transit. Investigate further to rule out contamination.")


def transit_timing_variation_analysis(folded_lc, planet_period, ui):
    """
    Analyze Transit Timing Variations (TTVs) by extracting individual transit times
    and calculating their deviations from a linear ephemeris.

    Parameters:
    - folded_lc: lightkurve.LightCurve object that has been folded on the planet's period.
    - planet_period: float, the orbital period of the planet in the same units as folded_lc.time.

    Returns:
    - None. Displays a plot of TTVs.
    """
    # Validate inputs
    if not hasattr(folded_lc, 'time') or not hasattr(folded_lc, 'flux'):
        raise AttributeError("folded_lc must have 'time' and 'flux' attributes.")

    if not isinstance(planet_period, (int, float)):
        raise ValueError("planet_period must be a numerical value (int or float).")

    # Extract time and flux as unitless arrays
    time = folded_lc.time.value  # Assuming time is in days; adjust if different
    flux = folded_lc.flux.value  # Flux as unitless

    # Calculate epochs using floor division to assign each time point to an epoch
    epochs = time // planet_period
    unique_epochs = np.unique(epochs)

    transit_times = []
    for epoch in unique_epochs:
        transit_mask = (epochs == epoch)
        if np.any(transit_mask):
            # Extract the times corresponding to the current epoch
            current_times = time[transit_mask]
            # Calculate the median transit time for this epoch
            transit_time = np.median(current_times)
            transit_times.append(transit_time)

    # Convert transit_times to a NumPy array for numerical operations
    transit_times = np.array(transit_times)

    # Check if at least two transits are present to calculate deviations
    if len(transit_times) < 2:
        print("Insufficient number of transits to calculate timing variations.")
        return

    # Calculate expected transit times based on a linear ephemeris
    expected_times = transit_times[0] + np.arange(len(transit_times)) * planet_period

    # Calculate deviations (TTVs)
    deviations = transit_times - expected_times

    # Instead of plt.show(), use the UI to plot
    ui.plot_ttv(unique_epochs, deviations)


def calculate_snr(folded_lc, planet_period):
    try:
        in_transit = (np.abs(folded_lc.time.value) < (0.05 * planet_period))  # Assume 5% of the period for transit duration
        out_of_transit = ~in_transit

        transit_depth = np.median(folded_lc.flux[in_transit])
        out_transit_flux = folded_lc.flux[out_of_transit]
        noise = np.std(out_transit_flux)

        snr = np.abs(1 - transit_depth) / noise
        print(f"Signal-to-Noise Ratio (SNR): {snr}")

        if snr < 7:
            print("Warning: Low SNR, transit signal may not be reliable.")
    except Exception as e:
        print(f"Error calculating SNR: {e}")


# Function to search for and download all TPF files using TIC ID
def search_and_download_tpf(tic_id):
    try:
        search_result = lk.search_targetpixelfile(f"TIC {tic_id}")
        if len(search_result) == 0:
            print(f"No TPF files found for TIC {tic_id}.")
            return None

        # Download all matching TPF files
        tpfs = search_result.download_all()

        if tpfs is None or len(tpfs) == 0:
            print(f"No files downloaded for TIC {tic_id}.")
            return None

        print(f"Downloaded {len(tpfs)} TPF files for TIC {tic_id}.")

        # Store paths to all downloaded files
        downloaded_paths = []
        for tpf in tpfs:
            downloaded_paths.append(tpf.path)
            print(f"Saved TPF file to: {tpf.path}")

        # Optionally print a summary of all paths
        print("\nSummary of all downloaded files:")
        for path in downloaded_paths:
            print(path)

        return tpfs

    except Exception as e:
        print(f"Error searching for TIC {tic_id}: {e}")
        return None


class TESSAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TESS Data Analyzer")
        
        # Create control panel frame
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add file load controls
        ttk.Label(self.control_frame, text="Load Local File:").grid(row=0, column=0, padx=5, pady=5)
        self.file_entry = ttk.Entry(self.control_frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)
        self.browse_button = ttk.Button(self.control_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        self.load_button = ttk.Button(self.control_frame, text="Load", command=self.load_local_file)
        self.load_button.grid(row=0, column=3, padx=5, pady=5)
        
        # Add TIC ID controls
        ttk.Label(self.control_frame, text="Search by TIC ID:").grid(row=1, column=0, padx=5, pady=5)
        self.tic_entry = ttk.Entry(self.control_frame, width=40)
        self.tic_entry.grid(row=1, column=1, padx=5, pady=5)
        self.search_button = ttk.Button(self.control_frame, text="Search & Download", command=self.search_tic)
        self.search_button.grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        
        # Create status label
        self.status_label = ttk.Label(self.control_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=2, column=0, columnspan=4, pady=5)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.lightcurve_tab = ttk.Frame(self.notebook)
        self.ttv_tab = ttk.Frame(self.notebook)
        self.centroid_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.lightcurve_tab, text="Light Curve")
        self.notebook.add(self.ttv_tab, text="TTV Analysis")
        self.notebook.add(self.centroid_tab, text="Centroid Analysis")
        
        # Add canvas for each tab
        self.lightcurve_canvas = None
        self.ttv_canvas = None
        self.centroid_canvas = None
        
        # Add output selection controls
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
        
        # Add navigation controls to each tab
        self.add_navigation_controls(self.lightcurve_tab, 'lightcurve')
        self.add_navigation_controls(self.ttv_tab, 'ttv')
        self.add_navigation_controls(self.centroid_tab, 'centroid')
        
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
        
        self.status_label.config(text="Processing file...", foreground="blue")
        try:
            process_tess_file(file_path, self)
            self.status_label.config(text="File processed successfully", foreground="green")
        except Exception as e:
            self.status_label.config(text=f"Error processing file: {str(e)}", foreground="red")

    def search_tic(self):
        tic_id = self.tic_entry.get()
        if not tic_id:
            self.status_label.config(text="Please enter a TIC ID", foreground="red")
            return
        
        self.status_label.config(text="Searching and downloading data...", foreground="blue")
        try:
            tpfs = search_and_download_tpf(tic_id)
            if tpfs is not None:
                for tpf in tpfs:
                    process_tess_file(tpf.path, self)
                self.status_label.config(text=f"Downloaded and processed {len(tpfs)} files", foreground="green")
            else:
                self.status_label.config(text="No files found for this TIC ID", foreground="red")
        except Exception as e:
            self.status_label.config(text=f"Error processing TIC ID: {str(e)}", foreground="red")

    def add_navigation_controls(self, parent, tab_name):
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.prev_button = ttk.Button(nav_frame, text="Previous", 
                                    command=lambda: self.show_previous_output(tab_name))
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(nav_frame, text="Next", 
                                    command=lambda: self.show_next_output(tab_name))
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.output_label = ttk.Label(nav_frame, text="Output 1 of 1")
        self.output_label.pack(side=tk.LEFT, padx=5)
        
        setattr(self, f'{tab_name}_nav_label', self.output_label)

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
            
            new_canvas = FigureCanvasTkAgg(fig, master=getattr(self, f'{tab_name}_tab'))
            new_canvas.draw()
            new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            setattr(self, f'{tab_name}_canvas', new_canvas)
            
            # Update navigation label
            label = getattr(self, f'{tab_name}_nav_label')
            label.config(text=f"Output {index + 1} of {len(outputs)}")

    def plot_lightcurve(self, folded_lc, moving_avg):
        if self.lightcurve_canvas:
            self.lightcurve_canvas.get_tk_widget().destroy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        folded_lc.scatter(ax=ax, c='black', s=2)
        ax.plot(folded_lc.time.value, moving_avg, color='red', label='Smoothed Moving Average')
        
        # Calculate dynamic limits based on data
        time_values = folded_lc.time.value
        flux_values = folded_lc.flux.value
        
        # Set x-axis limits to show full phase range
        ax.set_xlim(min(time_values), max(time_values))
        
        # Set y-axis limits based on data range with some padding
        y_min = min(flux_values) - 0.1 * (max(flux_values) - min(flux_values))
        y_max = max(flux_values) + 0.1 * (max(flux_values) - min(flux_values))
        ax.set_ylim(y_min, y_max)
        
        # Adjust layout to use available space
        fig.tight_layout()
        
        ax.set_title('Phase Folded Light Curve')
        ax.set_xlabel('Phase [days]')
        ax.set_ylabel('Normalized Flux')
        ax.legend()
        
        self.lightcurve_canvas = FigureCanvasTkAgg(fig, master=self.lightcurve_tab)
        self.lightcurve_canvas.draw()
        self.lightcurve_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_ttv(self, unique_epochs, deviations):
        if self.ttv_canvas:
            self.ttv_canvas.get_tk_widget().destroy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(unique_epochs, deviations, 'o-', color='blue', markerfacecolor='red')
        
        # Calculate dynamic limits
        x_min = min(unique_epochs) - 0.5
        x_max = max(unique_epochs) + 0.5
        y_min = min(deviations) - 0.1 * (max(deviations) - min(deviations))
        y_max = max(deviations) + 0.1 * (max(deviations) - min(deviations))
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Adjust layout
        fig.tight_layout()
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Transit Timing Deviation (days)')
        ax.set_title('Transit Timing Variations (TTVs)')
        ax.grid(True)
        
        self.ttv_canvas = FigureCanvasTkAgg(fig, master=self.ttv_tab)
        self.ttv_canvas.draw()
        self.ttv_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_centroid(self, clipped_lc_time, centroid_col, centroid_row):
        if self.centroid_canvas:
            self.centroid_canvas.get_tk_widget().destroy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(clipped_lc_time, centroid_col, label='Centroid Column')
        ax.plot(clipped_lc_time, centroid_row, label='Centroid Row')
        
        # Calculate dynamic limits
        x_min = min(clipped_lc_time) - 0.1 * (max(clipped_lc_time) - min(clipped_lc_time))
        x_max = max(clipped_lc_time) + 0.1 * (max(clipped_lc_time) - min(clipped_lc_time))
        
        y_min = min(min(centroid_col), min(centroid_row)) - 0.1 * (max(max(centroid_col), max(centroid_row)) - min(min(centroid_col), min(centroid_row)))
        y_max = max(max(centroid_col), max(centroid_row)) + 0.1 * (max(max(centroid_col), max(centroid_row)) - min(min(centroid_col), min(centroid_row)))
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Adjust layout
        fig.tight_layout()
        
        ax.set_xlabel('Time (BTJD)')
        ax.set_ylabel('Centroid Position (pixels)')
        ax.set_title('Centroid Motion Over Time')
        ax.legend()
        
        self.centroid_canvas = FigureCanvasTkAgg(fig, master=self.centroid_tab)
        self.centroid_canvas.draw()
        self.centroid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    ui = TESSAnalyzerUI(root)
    root.mainloop()
