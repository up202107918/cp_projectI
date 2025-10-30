#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF 1000000000

typedef struct {
    int p;              // total processes
    int q;              // grid order (sqrt(p))
    int my_rank;        // rank in grid communicator
    int my_row, my_col; // coordinates in grid
    MPI_Comm grid_comm; // Cartesian grid communicator
    MPI_Comm row_comm;  // row subcommunicator
    MPI_Comm col_comm;  // column subcommunicator
} Grid;

static void grid_setup(Grid* g) {
    int dims[2], periods[2] = {0, 0}, coords[2];
    MPI_Comm_size(MPI_COMM_WORLD, &g->p);
    g->q = (int)lround(sqrt((double)g->p));
    dims[0] = dims[1] = g->q;
    // If p is not a perfect square, we won't create a proper grid
    if (g->q * g->q != g->p) {
        g->grid_comm = MPI_COMM_NULL;
        g->row_comm = MPI_COMM_NULL;
        g->col_comm = MPI_COMM_NULL;
        g->my_rank = -1;
        g->my_row = g->my_col = -1;
        return;
    }
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &g->grid_comm);
    MPI_Comm_rank(g->grid_comm, &g->my_rank);
    MPI_Cart_coords(g->grid_comm, g->my_rank, 2, coords);
    g->my_row = coords[0];
    g->my_col = coords[1];
    int remain_dims_row[2] = {0, 1};
    int remain_dims_col[2] = {1, 0};
    MPI_Cart_sub(g->grid_comm, remain_dims_row, &g->row_comm);
    MPI_Cart_sub(g->grid_comm, remain_dims_col, &g->col_comm);
}

// Local C += A(min-plus)B for (bs x bs) blocks
static inline void local_minplus_mm(const int* A, const int* B, int* C, int bs) {
    for (int i = 0; i < bs; ++i) {
        int ioffA = i * bs;
        int ioffC = i * bs;
        for (int k = 0; k < bs; ++k) {
            int a = A[ioffA + k];
            if (a >= INF) continue; // skip INF contributions
            int koffB = k * bs;
            for (int j = 0; j < bs; ++j) {
                int b = B[koffB + j];
                if (b >= INF) continue; // avoid overflow, no path via k
                int cand = a + b;
                if (cand < C[ioffC + j]) C[ioffC + j] = cand;
            }
        }
    }
}

// Fox algorithm over a q x q grid: C = A(min-plus)B, all blocks are bs x bs
static void fox_minplus(const Grid* g, const int* A_local_in, int* B_local_io, int* C_local, int bs) {
    // working buffers
    int* A_bcast = (int*)malloc((size_t)bs * bs * sizeof(int));
    if (!A_bcast) {
        fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // init C with INF
    for (int i = 0; i < bs * bs; ++i) C_local[i] = INF;

    for (int stage = 0; stage < g->q; ++stage) {
        int root_col = (g->my_row + stage) % g->q; // root within the row communicator
        // Broadcast A block across the row
        if (g->my_col == root_col) {
            // root sends its A
            MPI_Bcast((void*)A_local_in, bs * bs, MPI_INT, root_col, g->row_comm);
            local_minplus_mm(A_local_in, B_local_io, C_local, bs);
        } else {
            MPI_Bcast(A_bcast, bs * bs, MPI_INT, root_col, g->row_comm);
            local_minplus_mm(A_bcast, B_local_io, C_local, bs);
        }
        // Rotate B blocks up by one within the column
        int src = (g->my_row + 1) % g->q;
        int dst = (g->my_row + g->q - 1) % g->q;
        MPI_Sendrecv_replace(B_local_io, bs * bs, MPI_INT, dst, 0, src, 0, g->col_comm, MPI_STATUS_IGNORE);
    }

    free(A_bcast);
}

// Copy out a block (bs x bs) from a padded matrix (Npad x Npad) starting at (r0, c0)
static void copy_block_out(const int* M, int N, int r0, int c0, int bs, int* out) {
    for (int i = 0; i < bs; ++i) {
        memcpy(out + i * bs, M + (r0 + i) * N + c0, (size_t)bs * sizeof(int));
    }
}

// Copy in a block (bs x bs) into a matrix (N x N) at (r0, c0)
static void copy_block_in(int* M, int N, int r0, int c0, int bs, const int* in) {
    for (int i = 0; i < bs; ++i) {
        memcpy(M + (r0 + i) * N + c0, in + i * bs, (size_t)bs * sizeof(int));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Grid grid;
    grid_setup(&grid);

    // If p is not a perfect square, print error and exit
    if (grid.q * grid.q != world_size) {
        if (world_rank == 0) {
            printf("ERROR: Invalid configuration!\n");
            fflush(stdout);
        }
        MPI_Finalize();
        return 0;
    }

    // Rank 0 reads input matrix
    int N = 0;
    int* M_global = NULL; // NxN
    if (world_rank == 0) {
        if (scanf("%d", &N) != 1) {
            // invalid input (anything that is not an integer)
            printf("ERROR: Invalid configuration!\n");
            fflush(stdout);
            MPI_Finalize();
            return 0;
        }
        if (N <= 0) {
            printf("ERROR: Invalid configuration!\n");
            fflush(stdout);
            MPI_Finalize();
            return 0;
        }
        M_global = (int*)malloc((size_t)N * N * sizeof(int));
        if (!M_global) {
            fprintf(stderr, "Allocation failed at root\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int x;
                if (scanf("%d", &x) != 1) x = 0;
                if (i == j) {
                    M_global[(size_t)i * N + j] = 0;
                } else {
                    M_global[(size_t)i * N + j] = (x == 0 ? INF : x);
                }
            }
        }
    }

    // Broadcast N to all
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Requirement: fail for any uneven (odd) N regardless of number of processes
    if ((N % 2) != 0) {
        if (world_rank == 0) {
            printf("ERROR: Invalid configuration!\n");
            fflush(stdout);
        }
        MPI_Finalize();
        return 0;
    }

    // Require perfect block decomposition without padding
    if (N % grid.q != 0) {
        if (world_rank == 0) {
            printf("ERROR: Invalid configuration!\n");
            fflush(stdout);
        }
        MPI_Finalize();
        return 0;
    }

    // Block size and global matrix distribution without padding
    int bs = N / grid.q;

    // Broadcast the full N x N matrix to all ranks
    if (world_rank != 0) {
        M_global = (int*)malloc((size_t)N * N * sizeof(int));
        if (!M_global) {
            fprintf(stderr, "Allocation failed at worker global\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(M_global, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Extract my local block
    int* A_local = (int*)malloc((size_t)bs * bs * sizeof(int));
    int* B_local = (int*)malloc((size_t)bs * bs * sizeof(int));
    int* C_local = (int*)malloc((size_t)bs * bs * sizeof(int));
    if (!A_local || !B_local || !C_local) {
        fprintf(stderr, "Allocation failed local\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int r0 = grid.my_row * bs;
    int c0 = grid.my_col * bs;
    copy_block_out(M_global, N, r0, c0, bs, A_local);
    memcpy(B_local, A_local, (size_t)bs * bs * sizeof(int));

    // Repeated squaring: compute T = D; then T = T(minplus)T, k times until 2^k >= N-1
    int needed = (N <= 1) ? 0 : (int)ceil(log2((double)(N - 1)));
    for (int it = 0; it < needed; ++it) {
        fox_minplus(&grid, A_local, B_local, C_local, bs);
        // Next power: A_local = C_local; B_local = A_local
        int* tmp = A_local; A_local = C_local; C_local = tmp;
        memcpy(B_local, A_local, (size_t)bs * bs * sizeof(int));
    }

    // Gather result blocks to grid root (rank 0 in grid communicator) into M_global
    if (grid.my_rank == 0) {
        // Place own block
        copy_block_in(M_global, N, r0, c0, bs, A_local);
        // Receive other blocks
        for (int pr = 0; pr < grid.q; ++pr) {
            for (int pc = 0; pc < grid.q; ++pc) {
                int coords[2] = {pr, pc};
                int rank_in_grid;
                MPI_Cart_rank(grid.grid_comm, coords, &rank_in_grid);
                if (rank_in_grid == grid.my_rank) continue;
                int* buf = (int*)malloc((size_t)bs * bs * sizeof(int));
                MPI_Recv(buf, bs * bs, MPI_INT, rank_in_grid, 777, grid.grid_comm, MPI_STATUS_IGNORE);
                copy_block_in(M_global, N, pr * bs, pc * bs, bs, buf);
                free(buf);
            }
        }
        // Print the top-left N x N, converting INF to 0
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int v = M_global[(size_t)i * N + j];
                if (v >= INF) v = 0;
                if (j) printf(" %d", v);
                else printf("%d", v);
            }
            printf("\n");
        }
    } else {
        // Send my block to grid root
        MPI_Send(A_local, bs * bs, MPI_INT, 0, 777, grid.grid_comm);
    }
    
    free(M_global);
    free(A_local);
    free(B_local);
    free(C_local);

    MPI_Finalize();
    return 0;
}
