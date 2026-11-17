import argparse
import hashlib
import json
import os
import sys
import urllib.request
ARTICLE = 'https://api.figshare.com/v2/articles/28196177'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, 'data', 'external', 'idsse')
UA = {'User-Agent': 'tiki-data-downloader'}

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, path, size):
    tmp = path + '.part'
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r, open(tmp, 'wb') as f:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print(f'\r  {done / 1000000.0:8.1f} / {size / 1000000.0:.1f} MB', end='', flush=True)
    print()
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-positions', action='store_true', help='no bajar el tracking (pesado)')
    args = ap.parse_args()
    os.makedirs(DEST, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(ARTICLE, headers=UA)) as r:
        files = json.load(r)['files']
    bad = []
    for f in sorted(files, key=lambda f: f['name']):
        if args.no_positions and 'positions' in f['name']:
            continue
        path = os.path.join(DEST, f['name'])
        if os.path.exists(path) and md5(path) == f['supplied_md5']:
            print(f"ok   {f['name']}")
            continue
        print(f"get  {f['name']}")
        fetch(f['download_url'], path, f['size'])
        if md5(path) != f['supplied_md5']:
            bad.append(f['name'])
            print(f"  MD5 NO COINCIDE: {f['name']}")
    if bad:
        sys.exit(f'{len(bad)} ficheros corruptos; vuelve a ejecutar el script')
    print(f'listo: {DEST}')
if __name__ == '__main__':
    main()
