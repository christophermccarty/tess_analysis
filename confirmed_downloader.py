import json
import requests
import pandas as pd
from urllib.parse import quote as urlencode
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


# Helper functions for querying MAST
def mast_query(request):
    request_url = 'https://mast.stsci.edu/api/v0/invoke'
    headers = {
        "Content-type": "application/x-www-form-urlencoded",
        "Accept": "text/plain"
    }

    req_string = json.dumps(request)
    req_string = urlencode(req_string)

    try:
        resp = requests.post(request_url, data="request=" + req_string, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.headers, resp.content.decode('utf-8')
        else:
            print(f"Error: Received status code {resp.status_code}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None, None


# Function to search by TIC ID for timeseries data using RA/Dec
def search_by_coordinates(tic_id, ra, dec):
    request = {
        "service": "Mast.Caom.Cone",
        "format": "json",
        "params": {
            "ra": ra,
            "dec": dec,
            "radius": 0.02,  # A small search radius around the coordinates
            "dataproduct_type": "timeseries",
            "observation_type": "science",
            "filters": "TESS"
        }
    }

    headers, out_string = mast_query(request)
    if out_string:
        out_data = json.loads(out_string)
        if 'data' in out_data and len(out_data['data']) > 0:
            for entry in out_data['data']:
                if entry.get('dataproduct_type') == 'timeseries':
                    obsid = entry.get('obsid')
                    print(f"Found obsid {obsid} for TIC {tic_id}")
                    return obsid
    return None


# Function to download and unzip data using `obsid`
def download_data(obsid, destination='D:/dev/lightkurve/data/confirmed'):
    url = f'https://mast.stsci.edu/api/v0.1/Download/bundle.zip?previews=false&obsid={obsid}'
    print(f"Downloading from {url}...")

    try:
        http_response = urlopen(url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=destination)
        print(f"Data extracted to {destination}")
    except Exception as e:
        print(f"Error during download: {e}")


# Updated TIC search function to read from a CSV file
def tic_advanced_search_from_csv(csv_file):
    # Read the CSV file to get TIC ID, RA, and DEC
    df = pd.read_csv(csv_file)

    #for index, row in df.iterrows():
    for index, row in df.head(5).iterrows():
        tic_id = row['tic_id']
        ra = row['ra']
        dec = row['dec']

        print(f"Processing TIC ID: {tic_id}, RA: {ra}, Dec: {dec}")

        # Search by coordinates for each row
        obsid = search_by_coordinates(tic_id, ra, dec)

        # If obsid found, proceed to download
        if obsid:
            print(f"Retrieved obsid for TIC {tic_id}: {obsid}")
            download_data(obsid)
        else:
            print(f"No obsid found for TIC {tic_id}")


if __name__ == "__main__":
    confirmed_csv_file = r"D:\dev\lightkurve\testing\PS_2025.02.16_21.54.53.csv"  # List of confirmed exoplanets
    tic_advanced_search_from_csv(confirmed_csv_file)
