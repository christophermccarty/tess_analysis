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
        plt.close('all')  # Close any existing figures at start
        
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
        plt.close('all')
        print(f"Error processing {tpf_path}: {e}")
    finally:
        plt.close('all')  # Ensure all figures are closed

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
        # Use a wider window for transit detection
        transit_duration = 0.2 * planet_period  # Increased from 0.1
        in_transit = (np.abs(folded_lc.time.value) < transit_duration)
        out_of_transit = ~in_transit

        # Require minimum number of points for reliable calculation
        if np.sum(in_transit) < 10 or np.sum(out_of_transit) < 10:
            print("Warning: Insufficient points for reliable SNR calculation")
            return None

        # Calculate transit depth using robust statistics
        in_transit_flux = folded_lc.flux[in_transit]
        out_transit_flux = folded_lc.flux[out_of_transit]
        
        # Remove outliers before calculating depth
        in_transit_flux = sigma_clip(in_transit_flux, sigma=3)
        out_transit_flux = sigma_clip(out_transit_flux, sigma=3)
        
        transit_depth = np.nanmedian(in_transit_flux)
        baseline = np.nanmedian(out_transit_flux)
        
        # Calculate noise using out-of-transit scatter
        noise = np.nanstd(out_transit_flux)
        
        # Calculate SNR using relative depth
        depth = np.abs(baseline - transit_depth)
        snr = depth / noise if noise > 0 else 0
        
        print(f"Transit Depth: {depth:.6f}")
        print(f"Baseline Flux: {baseline:.6f}")
        print(f"Noise Level: {noise:.6f}")
        print(f"Signal-to-Noise Ratio (SNR): {snr:.6f}")

        if snr < 7:
            print("Warning: Low SNR, transit signal may not be reliable.")
            
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
        
        # Set fixed window size
        self.root.geometry("1024x768")  # Width x Height
        self.root.resizable(False, False)  # Disable window resizing
        
        # Create control panel frame with fixed height
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
        
        # Create notebook with specific size to fill remaining space
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create main frames for each tab
        self.lightcurve_tab = ttk.Frame(self.notebook)
        self.ttv_tab = ttk.Frame(self.notebook)
        self.centroid_tab = ttk.Frame(self.notebook)
        
        # Add the main frames to the notebook
        self.notebook.add(self.lightcurve_tab, text="Light Curve")
        self.notebook.add(self.ttv_tab, text="TTV Analysis")
        self.notebook.add(self.centroid_tab, text="Centroid Analysis")
        
        # Create plot frames and navigation controls for each tab
        self.setup_tab(self.lightcurve_tab, 'lightcurve')
        self.setup_tab(self.ttv_tab, 'ttv')
        self.setup_tab(self.centroid_tab, 'centroid')
        
        # Initialize canvases
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
        
    def setup_tab(self, tab, name):
        # Create frame for plot
        plot_frame = ttk.Frame(tab)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        setattr(self, f'{name}_frame', plot_frame)
        
        # Create frame for navigation at bottom
        nav_frame = ttk.Frame(tab)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add navigation controls
        self.add_navigation_controls(nav_frame, name)

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
        # Center the buttons
        parent.grid_columnconfigure(0, weight=1)
        
        # Create a frame for the buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=0, column=0, pady=5)
        
        # Add Previous button
        prev_button = ttk.Button(button_frame, text="Previous", 
                               command=lambda: self.show_previous_output(tab_name))
        prev_button.pack(side=tk.LEFT, padx=5)
        setattr(self, f'{tab_name}_prev_button', prev_button)
        
        # Add output label
        output_label = ttk.Label(button_frame, text="Output 0 of 0")
        output_label.pack(side=tk.LEFT, padx=10)
        setattr(self, f'{tab_name}_nav_label', output_label)
        
        # Add Next button
        next_button = ttk.Button(button_frame, text="Next", 
                               command=lambda: self.show_next_output(tab_name))
        next_button.pack(side=tk.LEFT, padx=5)
        setattr(self, f'{tab_name}_next_button', next_button)

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
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot data
        folded_lc.scatter(ax=ax, c='black', s=2)
        ax.plot(folded_lc.time.value, moving_avg, color='red', label='Smoothed Moving Average')
        
        # Set title and labels
        ax.set_title('Phase Folded Light Curve')
        ax.set_xlabel('Phase [days]')
        ax.set_ylabel('Normalized Flux')
        ax.legend()
        
        # Adjust layout
        fig.tight_layout(pad=1.0)
        
        # Create and pack canvas
        self.lightcurve_canvas = FigureCanvasTkAgg(fig, master=self.lightcurve_frame)
        self.lightcurve_canvas.draw()
        self.lightcurve_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Store the figure and update display
        self.current_outputs['lightcurve'].append(fig)
        self.update_output_display('lightcurve')
        plt.close(fig)

    def plot_ttv(self, unique_epochs, deviations):
        if self.ttv_canvas:
            self.ttv_canvas.get_tk_widget().pack_forget()
        
        # Create figure with adjusted size and margins
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        
        # Plot data
        ax.plot(unique_epochs, deviations, 'o-', color='blue', markerfacecolor='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Transit Timing Deviation (days)')
        ax.set_title('Transit Timing Variations (TTVs)')
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

    def plot_centroid(self, clipped_lc_time, centroid_col, centroid_row):
        if self.centroid_canvas:
            self.centroid_canvas.get_tk_widget().pack_forget()
        
        # Create figure with adjusted size and margins
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        
        # Plot data
        ax.plot(clipped_lc_time, centroid_col, label='Centroid Column')
        ax.plot(clipped_lc_time, centroid_row, label='Centroid Row')
        ax.set_xlabel('Time (BTJD)')
        ax.set_ylabel('Centroid Position (pixels)')
        ax.set_title('Centroid Motion Over Time')
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



if __name__ == "__main__":
    root = tk.Tk()
    ui = TESSAnalyzerUI(root)
    root.mainloop()
