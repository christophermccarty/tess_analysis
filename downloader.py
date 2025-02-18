import json
import requests
import sys
from urllib.parse import quote as urlencode
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

# Global list to accumulate obsids for download
obsid_downloads = []

# Helper functions for querying MAST
def mast_query(request):
    request_url = 'https://mast.stsci.edu/api/v0/invoke'
    version = ".".join(map(str, sys.version_info[:3]))
    headers = {
        "Content-type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
        "User-agent": "python-requests/" + version
    }
    req_string = json.dumps(request)
    req_string = urlencode(req_string)
    resp = requests.post(request_url, data="request=" + req_string, headers=headers)
    return resp.headers, resp.content.decode('utf-8')


# Helper functions to build filters
def set_filters(parameters):
    return [{"paramName": p, "values": v} for p, v in parameters.items()]


def set_min_max(min_value, max_value):
    return [{'min': min_value, 'max': max_value}]


# TIC advanced search function
def tic_advanced_search():
    # Filters based on the given URL parameters
    filts = set_filters({
        "Teff": set_min_max(2000, 4000),
        "d": set_min_max(1, 5)  # Distance in parsecs
    })

    # Define the MAST query for TIC search
    request = {
        "service": "Mast.Catalogs.Filtered.Ctl.Rows",
        "format": "json",
        "params": {
            "columns": "*",
            "filters": filts
        }
    }

    # Perform the query
    headers, out_string = mast_query(request)
    out_data = json.loads(out_string)

    # Check if the data is found
    if 'data' in out_data:
        results = out_data['data']
        
        # Build a set of unique TIC IDs by checking the key exactly "ID"
        # This will not pick up "objID" because dictionary key matching is exact.
        unique_ids = {entry['ID'] for entry in results if 'ID' in entry}
        print(f"Total TIC search results found (unique 'ID'): {len(unique_ids)}")
        
        proceed = input("Do you want to proceed with processing these results? (Yes/No): ")
        if proceed.strip().lower() not in ['yes', 'y']:
            print("Operation cancelled by user.")
            return

        # Sort results by distance (assuming the column 'd' is distance)
        sorted_results = sorted(results, key=lambda x: x.get('d'))

        # Display top sorted results and run debug for each cone search
        for i, entry in enumerate(sorted_results[:10]):
            tic_id = entry.get('ID')
            ra = entry.get('ra')
            dec = entry.get('dec')
            dist = entry.get('d')
            print(f"{i + 1}. TIC ID: {tic_id}, RA: {ra}, Dec: {dec}, Distance: {dist}")
            # Now search for the Product Group ID (obsid) using the new function with RA and Dec
            search_by_coordinates(tic_id, ra, dec)
        
        # After processing all entries, prompt the user to download:
        if obsid_downloads:
            print(f"\nTotal timeseries products found: {len(obsid_downloads)}")
            user_input = input("Do you want to proceed with downloading these results? (Yes/No): ")
            if user_input.strip().lower() in ['yes', 'y']:
                for idx, obsid in enumerate(obsid_downloads, 1):
                    print(f"Downloading result {idx}/{len(obsid_downloads)} with obsid {obsid}...")
                    download_data(obsid)
            else:
                print("Download cancelled by user.")
        else:
            print("No timeseries products available for download.")
    else:
        print("No data found in TIC search response.")


# New function to search by RA/Dec and obtain Product Group ID (obsid)
def search_by_coordinates(tic_id, ra, dec):
    # Define the MAST query for searching by RA and Dec, filtering for timeseries results.
    request = {
        "service": "Mast.Caom.Cone",
        "format": "json",
        "params": {
            "ra": ra,
            "dec": dec,
            "radius": 0.01,  # You might consider increasing this value if needed
            "dataproduct_type": "timeseries"  # Only return results with timeseries data
            # Alternatively, if required by the API, use "dataProductType": "timeseries"
        }
    }

    # Perform the query
    headers, out_string = mast_query(request)
    out_data = json.loads(out_string)

    # Extract and print Product Group ID (obsid) for timeseries results
    if 'data' in out_data:
        found = False
        for entry in out_data['data']:
            # Check the product type using keys from MastPy examples:
            product_type = (entry.get('dataproduct_type') or entry.get('dataProductType') or '').lower()
            if product_type == 'timeseries':
                obsid = entry.get('obsid')
                obsid_downloads.append(obsid)
                found = True
        if not found:
            print(f"No timeseries Product Group ID found for TIC {tic_id}")
    else:
        print(f"No Product Group ID found for TIC {tic_id}")


# Function to download and unzip data
def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
    print(f"Data extracted to {extract_to}")


# Function to handle downloading based on obsid
def download_data(obsid, destination='D:/dev/lightkurve/data/unconfirmed'):
    # Build the download URL for the data bundle
    url = f'https://mast.stsci.edu/api/v0.1/Download/bundle.zip?previews=false&obsid={obsid}'
    print(f"Downloading from {url}...")

    # Download and unzip the file
    download_and_unzip(url, destination)


# Main function to execute search
if __name__ == "__main__":
    tic_advanced_search()
