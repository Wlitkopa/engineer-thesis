

def convert_to_order_list(all_orders):
    result = ""
    for i in range(len(all_orders)):
        result += f"\norder with {all_orders[i][1]} shots: {all_orders[i][0]}"

    return result
