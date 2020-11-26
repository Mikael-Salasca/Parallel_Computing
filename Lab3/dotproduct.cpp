/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu>

/* SkePU user functions */
// ac+bd
float add ( float a , float b) {
	return a + b ;
}

float mult ( float a , float b ) {
	return a * b ;
}


int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}

	const size_t size = std::stoul(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
//	spec.setCPUThreads(<integer value>);
	skepu::setGlobalBackendSpec(spec);


	/* Skeleton instances */
	auto dotprod_comb= skepu::MapReduce<2>(mult,add);

	auto tmp = skepu::Map<2>(mult);
	auto dotprod_sep = skepu::Reduce(add);


	/* SkePU containers */
	skepu::Vector<float> v1(size, 1.0f), v2(size, 2.0f);
	skepu::Vector<float> v3(size, 0.0f);

	/* Compute and measure time */
	float resComb, resSep;

	auto timeComb = skepu::benchmark::measureExecTimeIdempotent([&]
	{
		resComb = dotprod_comb(v1,v2);
	});

	auto timeSep = skepu::benchmark::measureExecTimeIdempotent([&]
	{
		tmp(v3,v1,v2);
		resSep = dotprod_sep(v3);

	});

	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << ( timeSep.count() / 10E6) << " seconds.\n";


	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";

	return 0;
}
