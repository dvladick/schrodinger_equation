import numpy as np
from functions import get_rows_info
from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


n = 5

nrows, rows = get_rows_info(n, size, rank)
print(f'p #{rank}: {rows}')
if rank == 0: print('='*50)


if rank == 0: #rank 0 load and share
    
    mtx_global = np.load('matrix_with_b.npy').astype(np.float64)
    print(f'dtype: {mtx_global.dtype}')
    print(f'root:\n{mtx_global}\n')
    comm.Bcast([mtx_global, MPI.DOUBLE], root = 0)

else: # the rest of ranks acquire
    mtx_global = np.empty((n, n + 1), dtype = np.float64)
    comm.Bcast([mtx_global, MPI.DOUBLE], root = 0)



# take required rows
local_rows = np.empty((nrows, n + 1), dtype = np.float64)
for i, row_idx in enumerate(rows):

    local_rows[i, :] = mtx_global[row_idx, :]


comm.Barrier()

#clear memory except root process
if rank != 0: del mtx_global


# GAUSSIAN ELINIMATION
    

tmp = np.zeros(n + 1, dtype = np.float64)


# show_ranks = []

row = 0
for i in range(n - 1):

    if row < nrows:
        # print(f'rank = {rank}, row = {row}; rows: {rows}; Cond (i == rows[row]): ({i} == {rows[row]}) = {i == rows[row]}')
        # print(f'rank = {rank}, row = {row}; rows: {rows};')
    
        if i == rows[row]:
            
            # print(f'rank = {rank}, row_idx = {row_idx}, local = {rows.index(row_idx)}')
            # print(f'local_rows:\n{local_rows}')
            comm.Bcast([local_rows[row, :], MPI.DOUBLE], root = rank)
            tmp = local_rows[row, :].copy()
            row += 1
        else:

            comm.Bcast([tmp, MPI.DOUBLE], root = i % size)
            # print(f'rank = {rank}, tmp: {tmp}')
            # print('_'*50)

        # if rank in show_ranks: 
        #     print(f'rank = {rank}, i = {i}, range({row}, {nrows}) = {list(range(row, nrows))}')
        #     print(f'rank = {rank}, tmp: {tmp}')
        for j in range(row, nrows):
            
            # if rank in show_ranks: print(f'rank = {rank}; local_rows (start of step):\n {np.round(local_rows, 3)}')
            scaling = local_rows[j, i] / tmp[i]
            # if rank in show_ranks:  print(f'rank = {rank}, scaling =  local_rows[{j}, {i}] / tmp[{i}] =  {local_rows[j, i]} / {tmp[i]} =  {local_rows[j, i] / tmp[i]}')
            local_rows[j, :] -= scaling * tmp
            # if rank in show_ranks:  print(f'rank = {rank}; local_rows (finish of step):\n {np.round(local_rows, 3)}')


    # if rank == 0: print('_' * 50 + '\n')
# print(f'rank = {rank}, Final row = {row}')
            
comm.Barrier()
print(f'\nrank = {rank}; local_rows:\n {np.round(local_rows, 3)}')


MPI.Finalize()