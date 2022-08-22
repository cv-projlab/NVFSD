import numpy as np
cimport numpy as np
import cv2
import random

DTYPE = np.uint8

cpdef transformation(np.ndarray clip, int angle, float value, int deltax, int deltay):
    cdef int batchno
    cdef int frameno
    cdef int h
    cdef int w
    cdef int ang
    cdef float val
    cdef int dx
    cdef int dy

    cdef np.ndarray M
    cdef np.ndarray translation_matrix

    cdef np.ndarray img = np.zeros((224, 224), dtype=DTYPE)
    cdef np.ndarray newclip = np.zeros_like(clip, dtype=DTYPE)

    for batchno in range(clip.shape[0]):
        ang = int(random.uniform(-angle, angle))
        val = random.uniform(value, 1+(value))
        dx = int(random.uniform(-deltax, deltax))
        dy = int(random.uniform(-deltay, deltay))

        M = cv2.getRotationMatrix2D((int(w/2), int(h/2+40)), ang, val)

        translation_matrix = np.array([[1, 0, dx], [0, 1, -30+dy]], dtype=np.float32)

        for frameno in range(clip.shape[1]):
            img = clip[batchno, frameno, :, :]
            h, w = img.shape[:2]

            M = cv2.getRotationMatrix2D((int(w/2), int(h/2+40)), ang, val)

            translation_matrix = np.array([[1, 0, dx], [0, 1, -30+dy]], dtype=np.float32)

            img = cv2.warpAffine(img, M, (w, h))
            img = cv2.warpAffine(img, translation_matrix, (w, h))

            newclip[batchno, frameno, :, :] = img


    return newclip


cpdef random_erase(np.ndarray clip):
    cdef int batchno
    cdef int frameno
    cdef float val

    cdef np.ndarray img = np.zeros((224, 224), dtype=DTYPE)
    cdef np.ndarray newclip = np.zeros_like(clip, dtype=DTYPE)

    for batchno in range(clip.shape[0]):
        val = random.uniform(0, 0.65)

        for frameno in range(clip.shape[1]):
            img = clip[batchno, frameno, :, :]
            h, w = img.shape[:2]

            img[:int(h*val),:] = 0*img[:int(h*val),:]

            newclip[batchno, frameno, :, :] = img


    return newclip



cpdef affine(np.ndarray clip, int angle, float scale, int deltax, int deltay):
    cdef int batchno
    cdef int channelno
    cdef int frameno
    cdef int h
    cdef int w
    cdef int ang
    cdef float scl
    cdef int dx
    cdef int dy
    
    cdef np.ndarray M


    batchno = clip.shape[0] if clip.ndim == 5 else 1

    channelno = clip.shape[clip.ndim-4]
    frameno = clip.shape[clip.ndim-3]
    h = clip.shape[clip.ndim-2]
    w = clip.shape[clip.ndim-1]

    
    cdef np.ndarray newclip = np.copy(clip)
    
    for batchno_ in range(batchno):
        ang = int(random.uniform(-angle, angle))
        scl = random.uniform(scale, 1+(scale))
        dx = int(random.uniform(-deltax, deltax))
        dy = int(random.uniform(-deltay, deltay))
        
        M = cv2.getRotationMatrix2D((w//2, h//2), ang, scl) # Rotation
        M[:, ~0] += np.array([dx, dy]) # Translation
        
        # Apply mtx
        for channelno_ in range(channelno):
            for frameno_ in range(frameno):
                if clip.ndim > 4:
                    img = clip[batchno_, channelno_, frameno_, :, :]
                    img = cv2.warpAffine(img, M, (w, h))
                    newclip[batchno_, channelno_, frameno_, :, :] = img
                else:
                    img = clip[channelno_, frameno_, :, :]
                    img = cv2.warpAffine(img, M, (w, h))
                    newclip[channelno_, frameno_, :, :] = img

    return newclip


cpdef random_erase_box(np.ndarray clip, float prob):
    cdef int batchno
    cdef int batchno_
    cdef int frameno
    cdef int channelno
    cdef int h
    cdef int w
    cdef float rnd
    cdef int x
    cdef int y

    
    batchno = clip.shape[0] if clip.ndim == 5 else 1

    channelno = clip.shape[clip.ndim-4]
    frameno = clip.shape[clip.ndim-3]
    h = clip.shape[clip.ndim-2]
    w = clip.shape[clip.ndim-1]


    cdef np.ndarray newclip = np.copy(clip)
    

    for batchno_ in range(batchno):      
        rnd = np.random.rand()

        if rnd <= prob :
            #Make box
            hbox = np.random.randint(h//10,h/1.1) # box dimensions
            wbox = np.random.randint(w//10,w/1.1) # box dimensions

            box = np.zeros((hbox,wbox), dtype=DTYPE)

            #Generate random coordinates
            x = np.random.randint(0,h-hbox) # box coords
            y = np.random.randint(0,w-wbox) # box coords

            if clip.ndim > 4:
                newclip[batchno_, :,:, x:x + hbox, y:y + wbox] = box
            else:
                newclip[ :,:, x:x + hbox, y:y + wbox] = box


    return newclip


cpdef rotate180(np.ndarray clip):
    # Only for testing
    cdef int frameno
    cdef np.ndarray img = np.zeros((224, 224), dtype=DTYPE)
    cdef np.ndarray newclip = np.zeros_like(clip, dtype=DTYPE)

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), 180, 1)


    for frameno in range(clip.shape[0]):

        img = clip[frameno, :, :]

        img = cv2.warpAffine(img, M, (w, h))

        newclip[frameno, :, :] = img


    return newclip


cpdef appearance_only(np.ndarray clip):
    # Only for testing
    cdef int batchno
    cdef int frameno
    cdef np.ndarray img = np.zeros((224, 224), dtype=DTYPE)
    cdef np.ndarray newclip = np.zeros_like(clip, dtype=DTYPE)

    cdef np.ndarray sums = np.zeros((clip.shape[1],),dtype = np.int_)

    sums = sums*0
    for frameno in range(clip.shape[0]):
        img = clip[frameno, :, :]
        sums[frameno] = np.sum(img).astype(np.int_)

    maxidx = np.argmax(sums)

    for frameno in range(clip.shape[0]):
        newclip[frameno, :, :] = clip[maxidx, :, :]

    return newclip
