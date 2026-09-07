import struct,sys,os
T={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rd(f,n): return f.read(n)
def u32(f): return struct.unpack('<I',f.read(4))[0]
def u64(f): return struct.unpack('<Q',f.read(8))[0]
def val(f,t):
    if t==8:
        n=u64(f); return f.read(n).decode('utf-8','replace')
    if t==9:
        et=u32(f); n=u64(f); return [val(f,et) for _ in range(min(n,0))] or (f.seek(0,1),'<arr len=%d>'%n)[1] if False else _skiparr(f,et,n)
    fmt=T[t]; sz=struct.calcsize('<'+fmt); return struct.unpack('<'+fmt,f.read(sz))[0]
def _skiparr(f,et,n):
    if et==8:
        for _ in range(n):
            l=u64(f); f.seek(l,1)
        return '<str[%d]>'%n
    if et==9: raise SystemExit('nested')
    sz=struct.calcsize('<'+T[et]); f.seek(sz*n,1); return '<arr[%d]>'%n
for path in sys.argv[1:]:
    with open(path,'rb') as f:
        assert f.read(4)==b'GGUF'
        ver=u32(f); ntensor=u64(f); nkv=u64(f)
        kv={}
        for _ in range(nkv):
            kl=u64(f); k=f.read(kl).decode(); t=u32(f); kv[k]=val(f,t)
        keys=[k for k in kv if any(s in k for s in ('block_count','expert','attention.head','context_length','embedding_length','architecture','name','rope','key_length','value_length','ssm','feed_forward'))]
        print('##',os.path.basename(path),'tensors=%d'%ntensor)
        for k in sorted(keys): print('   ',k,'=',kv[k])
