
from array import array

output_file = open('file', 'wb')
float_array = array('f', [3.14, 2.7, 0.0, -1.0, 1.1])
float_array.tofile(output_file)
output_file.close()