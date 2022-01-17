import numpy as np

class NdlBinary:
    def __init__(self, file_name, mode="r"):
        self._file = open(file_name, mode=mode+"b")
        
    def close(self):
        self._file.close()

    def init(self):
        self._file.seek(0)
        
    def _readSegment(self):
        # read header
        seg = self._file.read(4)
        if len(seg) == 0: # eof
            return None
        header_size = np.frombuffer(seg, dtype=np.int32)[0]
        seg = self._file.read(4)
        header_type = seg.decode()
        
        dtype=None
        if header_type == "INTE":
            dtype=np.int32
        elif header_type == "REAL":
            dtype=np.float32
        elif header_type == "DOUB":
            dtype=np.float64
            
        # read data
        seg = self._file.read(header_size)
        data = np.frombuffer(seg, dtype=dtype)

        # check header integrity
        seg = self._file.read(4)
        if header_size != np.frombuffer(seg, dtype=np.int32)[0]:
            raise ValueError
        seg = self._file.read(4)
        if header_type != seg.decode():
            raise ValueError

        return data

    def read(self):
        shape = self._readSegment()
        if shape is None: # eof
            return None
        data = self._readSegment()
        return np.reshape(data, shape)
    
    def write(self, ndarray):
        """
            write ndarray binary segment
        """
        # write ndarray binary as below
        # 1. check data type
        if ndarray.dtype == "int32":
            nd_dtype = "INTE"
        elif ndarray.dtype == "float32":
            nd_dtype = "REAL"
        elif ndarray.dtype == "float64":
            nd_dtype = "DOUB"
        else:
            raise ValueError("ndarray datatype should be int32, float32 or float64")
        
        # 2. dimension
        shape = np.array(ndarray.shape, dtype=np.int32)
        self._file.write(addHeader(shape.tobytes(), "INTE"))
        
        # 3. ndarray flatten binary
        self._file.write(addHeader(ndarray.flatten().tobytes(), nd_dtype))
        
def addHeader(byte_arr, dtype):
    length = len(byte_arr)
    header_len = np.array([length], dtype=np.int32).tobytes()
    header_dtype = dtype.encode()
    header = header_len + header_dtype
    return header + byte_arr + header
