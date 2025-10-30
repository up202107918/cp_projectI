run-fox:
	mpicc -O2 -o fox fox.c && 
	mpirun -np (1,4,9,16,25) --oversubscribe ./fox --padding=auto < matrix_examples/input(5,6,300,600,900,1200)