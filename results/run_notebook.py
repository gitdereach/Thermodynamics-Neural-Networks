import json,sys,nbformat
from nbclient import NotebookClient
src=sys.argv[1]; nb=nbformat.read(src,as_version=4)
for c in nb.cells:
    if c.cell_type=='code': c.source=c.source.replace('niters=5000','niters=5000')
NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':sys.argv[2]}}).execute()
nbformat.write(nb,sys.argv[3])
for c in nb.cells:
    for o in c.get('outputs',[]):
        if o.get('output_type')=='stream': print(o['text'][-600:])
        if o.get('output_type')=='error': print('ERR',o['ename'],o['evalue'])
