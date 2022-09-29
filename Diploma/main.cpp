
#include <cllib>
#include <iostream>
#include <benchmark.hpp>

typedef float FLOAT;
using tlucanti::Y;
using tlucanti::G;
using tlucanti::W;
using tlucanti::C;

double run_gpu_test(
		unsigned long long size,
		unsigned long long times,
		const cllib::CLcontext &context,
		const cllib::CLqueue &queue,
		cllib::CLkernel &kernel
	)
{
	std::vector<FLOAT> in(size);
	for (unsigned long long i=0; i < size; ++i)
		in.at(i) = i;
	std::vector<FLOAT> out(size);

	cllib::CLarray<FLOAT, cllib::read_only_array> input(in, context, queue);
	cllib::CLarray<FLOAT, cllib::write_only_array> output(in.size(), context);

	kernel.reset_args();
	kernel.set_next_arg(input);
	kernel.set_next_arg(output);

	__test_start();
	while (times--)
		kernel.run(queue, false);
	queue.flush();
	__test_end();
	return __time_delta();
}

double run_cpu_test(
		unsigned long long size,
		unsigned long long times
	)
{
	std::vector<FLOAT> in(size);
	std::vector<FLOAT> out(size);

	for (unsigned long long i=0; i < size; ++i)
		in.at(i) = i;

	double res = 0;
	while (times--)
	{
		__test_start();
		for (unsigned long long i=0; i < size; ++i)
			out[i] = in[i];
		__test_end();
		USED(in);
		USED(out);
		res += __time_delta();
	}
	return res;
}

int main()
{
	auto platform = cllib::get_platforms().at(0);
	std::cout << Y["selecting platform "]
		<< G["[0]"] << W[": "]
		<< C[platform.get_platform_name()] << std::endl;

	auto device = platform.get_devices().at(0);
	std::cout << Y["selecting device "]
		<< G["[0]"] << W[": "]
		<< C[device.get_device_name()] << std::endl;

	cllib::CLcontext context(device);
	cllib::CLqueue queue(context, device);

	cllib::CLprogram program(
		2,
		std::ifstream("kernel.cl"),
		"fft",
		context
	);
	program.compile(device, true);

	std::vector<FLOAT> in = {1, 2, 3, 4, 5, 6, 7, 8};
	std::vector<FLOAT> out(in.size());

	cllib::CLarray<FLOAT, cllib::read_only_array> input(in, context, queue);
	cllib::CLarray<FLOAT, cllib::write_only_array> output(input.size(), context);

	cllib::CLkernel kernel(program, input.size());
	kernel.set_next_arg(input);
	kernel.set_next_arg(output);

	kernel.run(queue);

	output.dump(out, queue);

	std::cout << "\ninput data:\n";
	for (const auto &i : in)
		std::cout << i << ' ';
	std::cout << "\noutput data\n";
	for (const auto &i : out)
		std::cout << i << ' ';
	std::cout << std::endl;

	tlucanti::Benchmark bm("cllib::CLarray");
	bm.small(run_gpu_test(E2, E6, context, queue, kernel), run_cpu_test(E2, E6));
	bm.medium(run_gpu_test(E4, E5, context, queue, kernel), run_cpu_test(E4, E5));
	bm.large(run_gpu_test(E5, E2, context, queue, kernel), run_cpu_test(E5, E2));
}
