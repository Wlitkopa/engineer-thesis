

def convert_measurement(output) -> []:
    output_array = output.split(' ')
    for i in range(len(output_array)):
        output_array[i] = int(output_array[i], 2)

    return output_array