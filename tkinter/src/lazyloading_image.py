from sbNative.runtimetools import get_path
import pickle
import time
import os
import atexit
import gc
import weakref


class LazyImage:
    cacheindex = 0
    lazy_images = []
    def __init__(self, rgb, name):
        self.name = name
        self._rgb = rgb
        self.loaded = True
        self.cacheindex = LazyImage.cacheindex
        LazyImage.cacheindex += 1
        self.cachepath = get_path() / "imagecache" / f"cached_image_{self.cacheindex}.pkl"
        self.own_weakref = weakref.ref(self)
        LazyImage.lazy_images.append(self.own_weakref)
        self.last_access = time.time()
        self.unload_block = False

    def cache(self):
        if self.unload_block or not self.loaded:return
        
        if not self.cachepath.parent.exists():
            self.cachepath.parent.mkdir(parents=True)
        
        with open(str(self.cachepath), "wb") as f:
            f.write(pickle.dumps(self._rgb))
        self.loaded = False
        try:
            del self._rgb
            gc.collect()
        except:pass
        # print("cached", self.name)


    def load(self):
        if self.loaded:
            return self._rgb
        # print("loading", self.name)
        self.unload_block = True
        with open(str(self.cachepath), "rb") as f:

            self._rgb = pickle.loads(f.read())
            self.loaded = True
            self.unload_block = False
            return self._rgb

    @property
    def rgb(self):
        self.last_access = time.time()
        if not self.loaded:
            self.load()
        return self._rgb
    
    def __del__(self):
        LazyImage.lazy_images.remove(self.own_weakref)
        ## delete the cache file if present
        if self.cachepath.exists():
            os.remove(self.cachepath)


def unload_caller(): ## TODO: one image is always retained for some reason. If you delete all images, the last one that you clicked "remove image" on will be retained. As soon as new images are loaded, the previously, falsly retained gets recycled, so no real memory leak, but something to consider fixing
    while True:
        for img_weakref in LazyImage.lazy_images:
            img = img_weakref()
            if img is None:
                continue
            if time.time() - img.last_access > 1:
                img.cache()
        time.sleep(1)


def remove_all_cache():
    for img_weakref in LazyImage.lazy_images:
        img = img_weakref()
        if img is None:
            continue
        if img.cachepath.exists():
            os.remove(img.cachepath)


import threading
unload_thread = threading.Thread(target=unload_caller, daemon=True)
unload_thread.start()
atexit.register(remove_all_cache)
