import numpy as np
cimport numpy as np

DTYPE = np.uint8

cpdef OR_image_scale(cv2_img, factor):
    '''
    :param cv2_img: open cv image
    :param factor: scale factor. Resize to half the size: factor = 1/2
    :return: rescale_img: cv2_img rescaled by the factor of factor
    '''
	

    if factor >= 1:
        raise RuntimeError('Factor has to be lower than 1!!')
	
    cdef float fact
    fact = 1 / factor
    origSize = cv2_img.shape

    #    resized_img = np.zeros((origSize[0] // int(fact), origSize[1] // int(fact))).astype(np.uint8)
    cdef np.ndarray resized_img = np.zeros((origSize[0] // int(fact), origSize[1] // int(fact)), dtype=DTYPE)
    assert resized_img.dtype == DTYPE

    cdef int row
    cdef int col
    cdef int idx
    for row in range(resized_img.shape[0]):

        for col in range(resized_img.shape[1]):

            str_r = int(fact * row + 1 + (1 - fact))
            end_r = int(str_r + fact - 1)
            str_c = int(fact * col + 1 + (1 - fact))
            end_c = int(str_c + fact - 1)

            vec = np.array(cv2_img[str_r-1:end_r-1, str_c-1:end_c-1]).reshape(-1, )



            for idx in range(len(vec)):

                resized_img[row, col] = resized_img[row, col] | vec[idx]
    
    
    return resized_img