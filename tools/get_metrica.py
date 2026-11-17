import hashlib
import json
import os
import sys
import urllib.request
REPO = 'metrica-sports/sample-data'
COMMIT = 'e706dd506b360d69d9d123d5b8026e7294b13996'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, 'data', 'external', 'metrica')
UA = {'User-Agent': 'tiki-data-downloader'}
WANT = ('data/', 'documentation/')

def git_blob_sha(path):
    h = hashlib.sha1()
    h.update(b'blob %d\x00' % os.path.getsize(path))
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
            print('\r  {:8.1f} / {:.1f} MB'.format(done / 1000000.0, size / 1000000.0), end='', flush=True)
    print()
    os.replace(tmp, path)

def main():
    url = 'https://api.github.com/repos/{}/git/trees/{}?recursive=1'.format(REPO, COMMIT)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        tree = json.load(r)['tree']
    files = [t for t in tree if t['type'] == 'blob' and t['path'].startswith(WANT)]
    bad = []
    for t in sorted(files, key=lambda t: t['path']):
        rel = t['path'].split('/', 1)[1] if t['path'].startswith('data/') else t['path']
        path = os.path.join(DEST, *rel.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and git_blob_sha(path) == t['sha']:
            print('ok   ' + rel)
            continue
        print('get  ' + rel)
        fetch('https://raw.githubusercontent.com/{}/{}/{}'.format(REPO, COMMIT, t['path']), path, t['size'])
        if git_blob_sha(path) != t['sha']:
            bad.append(rel)
            print('  SHA NO COINCIDE: ' + rel)
    if bad:
        sys.exit('{} ficheros corruptos; vuelve a ejecutar el script'.format(len(bad)))
    print('listo: ' + DEST)
if __name__ == '__main__':
    main()
