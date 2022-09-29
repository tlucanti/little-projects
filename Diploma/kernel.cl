
__kernel void fft(
	__constant const double *__restrict input,
	__global double *__restrict output
	)
{
	int i = get_global_id(0);
	output[i] = input[i];
}
