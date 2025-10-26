#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

typedef struct {
    int p;              // number of processes
    MPI_Comm comm;      // communicator for entire grid
    MPI_Comm row_comm;  // communicator of my row
    MPI_Comm col_comm;  // communicator of my column 
    int q;              // order of grid  
    int my_row;         // my row number 
    int my_col;         // my column number
    int my_rank;        // my rank in the grid communicator
} GRID_INFO_TYPE;

void Setup_grid(GRID_INFO_TYPE* grid) {
    int old_rank, dims[2], periods[2], coords[2], varying_coords[2];
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);
    grid->q = (int)sqrt((double)grid->p);
    dims[0] = dims[1] = grid->q;
    periods[0] = periods[1] = 1; // periodic (cyclic) grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &(grid->comm));
    MPI_Comm_rank(grid->comm, &(grid->my_rank));
    MPI_Cart_coords(grid->comm, grid->my_rank, 2, coords);
    grid->my_row = coords[0];
    grid->my_col = coords[1];
    /* row_comm: all processes with same row (varying column) */
    varying_coords[0] = 0; varying_coords[1] = 1;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->row_comm));
    /* col_comm: all processes with same column (varying row) */
    varying_coords[0] = 1; varying_coords[1] = 0;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->col_comm));
}

/* Multiply two local blocks (square, size = b x b). All arrays are row-major */
static void local_mat_mult(int *A, int *B, int *C, int b) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < b; ++j) {
            int sum = 0;
            for (int k = 0; k < b; ++k) {
                sum += A[i*b + k] * B[k*b + j];
            }
            C[i*b + j] += sum;
        }
    }
}

/* Fox algorithm implementation using contiguous int blocks.
 * b: local block size (block is b x b)
 */
static void Fox(int b, GRID_INFO_TYPE *grid, int *local_A, int *local_B, int *local_C) {
    int q = grid->q;
    int *tmpA = (int*)malloc(b * b * sizeof(int));
    MPI_Status status;
    int row_rank, col_rank;
    MPI_Comm_rank(grid->row_comm, &row_rank); /* rank within the row communicator (0..q-1) */
    MPI_Comm_rank(grid->col_comm, &col_rank); /* rank within the col communicator (0..q-1) */

    /* initialize local_C to zero */
    for (int i = 0; i < b*b; ++i) local_C[i] = 0;

    for (int step = 0; step < q; ++step) {
        int bcast_root_col = (grid->my_row + step) % q; /* column index that holds the A block to broadcast */

        if (bcast_root_col == grid->my_col) {
            /* I'm the broadcaster for this step */
            MPI_Bcast(local_A, b*b, MPI_INT, bcast_root_col, grid->row_comm);
            local_mat_mult(local_A, local_B, local_C, b);
        } else {
            MPI_Bcast(tmpA, b*b, MPI_INT, bcast_root_col, grid->row_comm);
            local_mat_mult(tmpA, local_B, local_C, b);
        }

        /* circular shift of B blocks upwards in column communicator */
        int dest = (col_rank - 1 + q) % q; /* send to previous row in this column */
        int src  = (col_rank + 1) % q;     /* receive from next row in this column */
        MPI_Sendrecv_replace(local_B, b*b, MPI_INT, dest, 0, src, 0, grid->col_comm, &status);
    }

    free(tmpA);
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    GRID_INFO_TYPE grid;
    Setup_grid(&grid);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = 0;
    int *A = NULL, *B = NULL;

    /* Only world rank 0 reads input from stdin */
    if (world_rank == 0) {
        if (scanf("%d", &n) != 1) {
            fprintf(stderr, "Failed to read matrix size\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* allocate and read two n x n matrices: A then B */
        A = (int*)calloc((size_t)n * n, sizeof(int));
        B = (int*)calloc((size_t)n * n, sizeof(int));
        if (!A || !B) {
            fprintf(stderr, "Allocation failed on root\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (scanf("%d", &A[i*n + j]) != 1) { A[i*n + j] = 0; }
            }
        }
        /* Try to read second matrix B. If there aren't enough values, assume
           the input contained only one matrix and use B = A. */
        int read_ok = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (scanf("%d", &B[i*n + j]) != 1) { read_ok = 0; break; }
            }
            if (!read_ok) break;
        }
        if (!read_ok) {
            /* copy A into B */
            for (int i = 0; i < n*n; ++i) B[i] = A[i];
        }
    }

    /* broadcast original n to all processes */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n <= 0) {
        if (world_rank == 0) fprintf(stderr, "Invalid matrix size n=%d\n", n);
        MPI_Finalize();
        return 0;
    }

    /* grid.q must divide n exactly for this assignment's expected behavior.
       If n is not divisible by q, the expected output in the examples is
       an error message. */
    int q = grid.q;
    if (q * q != grid.p) { /* sanity */ }
    if (n % q != 0) {
        if (world_rank == 0) {
            /* Print exact expected error message (no trailing newline in examples) */
            printf("ERROR: Invalid configuration!");
        }
        MPI_Finalize();
        return 0;
    }
    int n_padded = n; /* no padding allowed; n must be divisible by q */
    int b = n_padded / q; /* local block size */

    /* root (world_rank==0) will distribute blocks using grid.comm */
    int root_grid_rank;
    MPI_Comm_rank(grid.comm, &root_grid_rank); /* this gives each process its grid rank; for world_rank==0 it's the sender rank */
    /* but we need the root's grid rank value on all processes: */
    int sender_grid_rank = -1;
    if (world_rank == 0) sender_grid_rank = root_grid_rank;
    MPI_Bcast(&sender_grid_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* allocate local blocks for each process */
    int block_elems = b * b;
    int *local_A = (int*)calloc((size_t)block_elems, sizeof(int));
    int *local_B = (int*)calloc((size_t)block_elems, sizeof(int));
    int *local_C = (int*)calloc((size_t)block_elems, sizeof(int));
    if (!local_A || !local_B || !local_C) {
        fprintf(stderr, "[%d] Local allocation failed\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    /* Root prepares padded matrices and sends blocks to appropriate grid ranks */
    if (world_rank == 0) {
        int *A_pad = (int*)calloc((size_t)n_padded * n_padded, sizeof(int));
        int *B_pad = (int*)calloc((size_t)n_padded * n_padded, sizeof(int));
        if (!A_pad || !B_pad) { fprintf(stderr, "Allocation failed for padded mats\n"); MPI_Abort(MPI_COMM_WORLD, 3); }

        /* copy original into padded arrays */
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            A_pad[i*n_padded + j] = A[i*n + j];
            B_pad[i*n_padded + j] = B[i*n + j];
        }

        /* send each block to its destination (in grid communicator ranks) */
        int coords[2];
        int dest_rank;
        int *tmp = (int*)malloc((size_t)block_elems * sizeof(int));
        for (int bi = 0; bi < q; ++bi) {
            for (int bj = 0; bj < q; ++bj) {
                /* build block tmp for A */
                for (int ii = 0; ii < b; ++ii) {
                    for (int jj = 0; jj < b; ++jj) {
                        int gi = bi*b + ii;
                        int gj = bj*b + jj;
                        tmp[ii*b + jj] = A_pad[gi*n_padded + gj];
                    }
                }
                coords[0] = bi; coords[1] = bj;
                MPI_Cart_rank(grid.comm, coords, &dest_rank);
                if (dest_rank == sender_grid_rank) {
                    memcpy(local_A, tmp, (size_t)block_elems * sizeof(int));
                } else {
                    MPI_Send(tmp, block_elems, MPI_INT, dest_rank, 100, grid.comm);
                }

                /* build block tmp for B and send */
                for (int ii = 0; ii < b; ++ii) {
                    for (int jj = 0; jj < b; ++jj) {
                        int gi = bi*b + ii;
                        int gj = bj*b + jj;
                        tmp[ii*b + jj] = B_pad[gi*n_padded + gj];
                    }
                }
                if (dest_rank == sender_grid_rank) {
                    memcpy(local_B, tmp, (size_t)block_elems * sizeof(int));
                } else {
                    MPI_Send(tmp, block_elems, MPI_INT, dest_rank, 200, grid.comm);
                }
            }
        }
        free(tmp);
        free(A_pad);
        free(B_pad);
    } else {
        /* non-root processes receive their blocks from sender_grid_rank (on grid.comm) */
        MPI_Status st;
        MPI_Recv(local_A, block_elems, MPI_INT, sender_grid_rank, 100, grid.comm, &st);
        MPI_Recv(local_B, block_elems, MPI_INT, sender_grid_rank, 200, grid.comm, &st);
    }

    /* All processes now have local_A and local_B. Run Fox algorithm on local blocks. */
    /* Temporary debug output: if CP_DEBUG is set, print a couple values of the
       received local blocks so we can tell whether distribution succeeded. */
    char *dbg = getenv("CP_DEBUG");
    if (dbg) {
        /* print to stderr so we don't interfere with normal stdout matrix output */
        int grid_rank;
        MPI_Comm_rank(grid.comm, &grid_rank);
        fprintf(stderr, "[world %d grid %d coords %d,%d] local_A[0]=%d local_B[0]=%d\n",
                world_rank, grid_rank, grid.my_row, grid.my_col,
                local_A[0], local_B[0]);
        fflush(stderr);
    }

    Fox(b, &grid, local_A, local_B, local_C);

    /* Gather results: root collects all blocks into C_pad and prints top-left n x n area */
    if (world_rank == 0) {
        int *C_pad = (int*)calloc((size_t)n_padded * n_padded, sizeof(int));
        if (!C_pad) { fprintf(stderr, "Allocation failed for C_pad\n"); MPI_Abort(MPI_COMM_WORLD, 4); }

        /* place own block */
        int my_coords[2];
        MPI_Cart_coords(grid.comm, sender_grid_rank, 2, my_coords);
        int bi = my_coords[0], bj = my_coords[1];
        for (int ii = 0; ii < b; ++ii)
            for (int jj = 0; jj < b; ++jj)
                C_pad[(bi*b + ii)*n_padded + (bj*b + jj)] = local_C[ii*b + jj];

        /* receive from other ranks */
        for (int pi = 0; pi < grid.p; ++pi) {
            if (pi == sender_grid_rank) continue;
            int coords2[2];
            MPI_Cart_coords(grid.comm, pi, 2, coords2);
            int bi2 = coords2[0], bj2 = coords2[1];
            int *tmpC = (int*)malloc((size_t)block_elems * sizeof(int));
            MPI_Status st;
            MPI_Recv(tmpC, block_elems, MPI_INT, pi, 300, grid.comm, &st);
            for (int ii = 0; ii < b; ++ii)
                for (int jj = 0; jj < b; ++jj)
                    C_pad[(bi2*b + ii)*n_padded + (bj2*b + jj)] = tmpC[ii*b + jj];
            free(tmpC);
        }

        /* print the resulting n x n matrix (top-left) in row-major form */
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                printf("%d", C_pad[i*n_padded + j]);
                if (j + 1 < n) printf(" ");
            }
            printf("\n");
        }
        free(C_pad);
    } else {
        /* non-root send their computed local_C back to root (using grid.comm ranks) */
        MPI_Send(local_C, block_elems, MPI_INT, sender_grid_rank, 300, grid.comm);
    }

    free(local_A); free(local_B); free(local_C);
    if (A) free(A); if (B) free(B);

    MPI_Finalize();
    return 0;
}