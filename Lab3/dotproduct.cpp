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

	// auto dotprod_map = skepu::Map<2>(mult);
	// auto dotprod_red = skepu::Reduce<2>(add);


	/* SkePU containers */
	skepu::Vector<float> v1(size, 1.0f), v2(size, 2.0f);
	// need one more ?

	/* Compute and measure time */
	float resComb, resSep;

	auto timeComb = skepu::benchmark::measureExecTime([&]
	{
		dotprod_comb(resComb,v1,v2);
	});

	auto timeSep = skepu::benchmark::measureExecTime([&]
	{
		// dotprod_map(v1,v2);
		// dotprod_red(v1,v2);

	});

	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << ( timeSep.count() / 10E6) << " seconds.\n";


	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";

	return 0;
}
