

def convert_to_matrix_row(output_data):
    result = ""
    for i in range(len(output_data)):
        result += f"\nvector with {output_data[i][2]} shots: "
        temp_result = "["
        for j in range(len(output_data[i][0])):
            if j == 0:
                temp_result += f"{output_data[i][0][j]}"
            else:
                temp_result += f", {output_data[i][0][j]}"
        temp_result += "],"
        result += temp_result
    return result