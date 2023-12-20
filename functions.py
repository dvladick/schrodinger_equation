def get_chunk(total, size, rank):
    
    chunk_size = total // size
    remainder = total % size
    
    if rank < remainder:
        chunk_size += 1
        
    return chunk_size


def get_rows_info(total, size, rank):
    
    nrows = get_chunk(total, size, rank)
    
    # print(f'step = {size}')
    # rows_list = [i for i in range(0, n, size)]

    rows_list = [i * size + rank for i in range(nrows)]

    return nrows, rows_list

# n = 14
# size = 3
# for i in range(3):

#     print(f'p_{i}, nrows = {get_chunk(n, size, i)}')
#     print(f'p_{i}, list of rows = {get_rows_info(n, size, i)}')
#     print('_' * 50)