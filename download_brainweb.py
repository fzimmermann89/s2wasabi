import concurrent.futures
import gzip
import hashlib
import io
import pickle
import re
import urllib.request
import warnings
from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm import tqdm

import argparse

def download(outdir, cachedir, workers=4):
    URL = "http://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html"
    page = requests.get(URL)
    subjects = re.findall('option value=(\d*)>', page.text)
    
    classesPD = {
        'gry':  0.86,
        'wht':0.77,
        'csf': 1,
        'mrw': .77,
        'dura': 1,
        'fat': 1,
        'fat2': .77,
        'mus': 1,
        'm-s': 1,
        'ves': 0.9,
    }


    def one_subject(subject, outfilename, cachedir, classesPD):
        def load_url(url, cachedir, timeout=60):
            h = hashlib.sha256(url.encode("utf-8")).hexdigest()
            try:
                res = pickle.load(open(Path(cachedir) / h, 'rb'))
            except (FileNotFoundError, EOFError):
                with urllib.request.urlopen(url, timeout=timeout) as conn:
                    res = conn.read()
                try:
                    pickle.dump(res, open(Path(cachedir) / h, 'wb'))
                except Exception as e:
                    warnings.warn('could not cache',e)
            return res

        def unpack(data, dtype, shape):
            return np.frombuffer(gzip.open(io.BytesIO(data)).read(), dtype=dtype).reshape(shape)


        def toPD(x,pd):
            return (np.clip(x-np.min(x[50]),0,4096)/4096*pd).astype(np.float32)

        def norm(input):
            out=np.zeros(input[0].shape+(len(input),),dtype=np.uint8)
            with np.errstate(divide='ignore', invalid='ignore'):
                norm=np.nan_to_num(255/sum(input))
            for i,v in enumerate(input):
                out[...,i]=v*norm
            s=np.array(out.shape[:-1])
            return out[tuple((slice(s0,-s1)) for s0,s1 in zip(s%16//2,s%16-s%16//2))+(slice(None),)]

        values=norm([toPD(unpack(
                        load_url(
                            f"http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject{subject}_{c}&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D",
                            cachedir=cachedir,
                        ),
                        shape=(362, 434, 362),
                        dtype=np.uint16,),pd) for c,pd in classesPD.items()])

        with h5py.File(outfilename, 'w') as f:
            f.create_dataset('classes', values.shape, dtype=values.dtype, data=values, chunks=(16,16,16,values.shape[-1]))
            bg = 255-np.sum(values,-1)
            f.create_dataset('background', bg.shape, dtype=values.dtype, data=bg, chunks=(32,32,32))
            with np.errstate(divide='ignore', invalid='ignore'):
                norm=(1/np.sum(values,-1)).astype(np.float32)
            f.create_dataset('norm', norm.shape, dtype=np.float32, data=norm,chunks=(32,32,32))
            f.attrs['classnames']=list(classesPD.keys())
            f.attrs['subject']=int(subject)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one_subject, subject, Path(outdir) / f's{subject}.h5', cachedir, classesPD): subject for subject in subjects}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            s = futures[future]
            try:
                fn = future.result()
            except Exception as e:
                print('%s generated an exception: %s' % (s, e))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'download brainweb')
    parser.add_argument('path', help='outputpath')
    parser.add_argument("--cache", help="cache, default /tmp", default='/tmp/')
    args = parser.parse_args()
    print('downloading...')
    download(args.path,args.cache)
    print('done!')
