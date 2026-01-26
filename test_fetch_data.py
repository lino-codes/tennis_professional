from pathlib import Path
import datetime
import requests

atp_url = 'http://www.tennis-data.co.uk/2026/2026.xlsx'
wta_url = 'http://www.tennis-data.co.uk/2026w/2026.xlsx'

target_folder = './data'

def download_file(url: str, dest_folder: str, filename: str | None = None) -> Path:
    """
    Download a file from `url` into `dest_folder`.
    If `filename` is None, use the basename from the URL.

    Returns the full Path to the downloaded file.
    """
    # Basic guard so we only accept http/https URLs
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Unsupported URL scheme in {url!r}")

    dest_path = Path(dest_folder)
    dest_path.mkdir(parents=True, exist_ok=True)  # create folder if needed [web:11][web:14]

    if filename is None:
        filename = url.split("/")[-1] or "downloaded_file"

    file_path = dest_path / filename

    # Stream to avoid loading large files fully into memory [web:7][web:19]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

    return file_path

# NOTE: download atp file
today = datetime.date.today().strftime('%Y%m%d')
atp_downloaded = download_file(atp_url, target_folder, f'atp_tennis-data_{today}.xlsx')
print(f"Saved to: {atp_downloaded}")
wta_downloaded = download_file(wta_url, target_folder, f'wta_tennis-data_{today}.xlsx')
print(f"Saved to: {wta_downloaded}")
