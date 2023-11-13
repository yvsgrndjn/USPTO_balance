import sys

# Read the input file for this job
input_file = f"{sys.argv[1]}"

# Read the list of elements from which we want to retrieve indices of our data
with open('/data/home/yves/USPTO_balance/USPTO_templates_tagged_rxns.txt', 'r') as f:
    bigger_list = f.read().splitlines()

# Read the elements from the current input file
with open(input_file, 'r') as f:
    input_elements = f.read().splitlines()

# Initialize a list to store the indices of matching elements
indices = []

# Find indices of elements in the bigger list
for element in input_elements:
    try:
        index = bigger_list.index(element)
        indices.append(index)
    except ValueError:
        # Element not found in the bigger list
        indices.append(-1)

# Save the indices to an output file
output_file = f"output_indices{sys.argv[1]}.txt"
with open(output_file, 'w') as f:
    f.write('\n'.join(map(str, indices)))
