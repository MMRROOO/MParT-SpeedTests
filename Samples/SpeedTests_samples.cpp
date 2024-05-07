/**
TODO: ADD DESCRIPTION
 */


#include <random>
#include <fstream>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "MParT/ComposedMap.h"
#include <MParT/ConditionalMapBase.h>
#include <MParT/MapFactory.h>
#include <MParT/MultiIndices/MultiIndexSet.h>
#include <MParT/Utilities/ArrayConversions.h>

#include <random>
#include <chrono>

using namespace std::chrono;

using namespace mpart; 

template<typename MemorySpace>
struct Generator{

    const unsigned int seed = 2012;

    std::mt19937 mt;
    std::uniform_real_distribution<double> dist;

    Generator() : mt(2012), dist(-1,1){};

    Kokkos::View<double*,MemorySpace> Coeffs(unsigned int numCoeffs);
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> Points(unsigned int dim, unsigned int numPts);
};

template<>
Kokkos::View<double*,Kokkos::HostSpace> Generator<Kokkos::HostSpace>::Coeffs(unsigned int numCoeffs)
{
    Kokkos::View<double*,Kokkos::HostSpace> output("Coefficients", numCoeffs);
    for(unsigned int i=0; i<numCoeffs; ++i)
        output(i) = dist(mt);

    return output;
}

template<>
Kokkos::View<double**,Kokkos::LayoutLeft, Kokkos::HostSpace> Generator<Kokkos::HostSpace>::Points(unsigned int dim, unsigned int numPts)
{
    Kokkos::View<double**,Kokkos::LayoutLeft, Kokkos::HostSpace> output("Points", dim, numPts);
    for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<numPts; ++j)
            output(i,j) = dist(mt);
    }

    return output;
}

#if defined(KOKKOS_ENABLE_CUDA )
template<typename MemorySpace>
Kokkos::View<double*,MemorySpace> Generator<MemorySpace>::Coeffs(unsigned int numCoeffs)
{   
    Kokkos::Random_XorShift64_Pool<> rand_pool(5374857);

    Kokkos::View<double*, MemorySpace> coeffs("Coefficients", numCoeffs);
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(int const& i){
        
        auto rand_gen = rand_pool.get_state();
        for (int k = 0; k < numCoeffs; k++) 
            coeffs(k) = rand_gen.frand(-1,1);
        rand_pool.free_state(rand_gen);
    });

    Kokkos::fence();

    return coeffs;
}

template<typename MemorySpace>
Kokkos::View<double**,Kokkos::LayoutLeft, MemorySpace> Generator<MemorySpace>::Points(unsigned int dim, unsigned int numPts)
{   
    Kokkos::Random_XorShift64_Pool<> rand_pool(5374857);

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> pts("Points", dim, numPts);
    Kokkos::parallel_for(dim, KOKKOS_LAMBDA(int const& i){
        auto rand_gen = rand_pool.get_state();
        for (int k = 0; k < numPts; k++) 
            pts(i,k) = rand_gen.frand(-1,1);
        rand_pool.free_state(rand_gen);
    });

    Kokkos::fence();

    return pts;
}

#endif 

int main(int argc, char* argv[]){

    //assert(argc>=2);
    std::string backend = argv[1];
    
    const unsigned int dim = 5;
    const unsigned int order = 5;
    
    Kokkos::initialize(argc, argv);
    {
    MapOptions opts;
    //opts.quadType = QuadTypes::AdaptiveSimpson;
    //std::string quad_string = "simpson";

    // Or
    //opts.quadType = QuadTypes::ClenshawCurtis;
   // opts.quadPts = 5;
    //std::string quad_string = "cc" + std::to_string(opts.quadPts);
    opts.basisType = BasisTypes::HermiteFunctions; 
    unsigned int nn = 6;

    Generator<Kokkos::DefaultExecutionSpace::memory_space> gen;

    auto tEvalMat_m = Eigen::VectorXd(nn);
    auto tLogDetMat_m = Eigen::VectorXd(nn);
    auto tEvalMat_s = Eigen::VectorXd(nn);
    auto tLogDetMat_s = Eigen::VectorXd(nn);

    unsigned int nk = 50; // number of repeated trials

    auto tEval = Eigen::MatrixXd(nn,nk);
    auto tLogDet = Eigen::MatrixXd(nn,nk);
    auto tInv = Eigen::MatrixXd(nn,nk);
    auto tCoeffGrad = Eigen::MatrixXd(nn,nk);

    std::cout << "\nRunning Backend " << backend << std::endl;

    for (unsigned int n=0; n<nn;++n){
        unsigned int num_sigmoids = 5;
        unsigned int numCenters = 2 + num_sigmoids*(num_sigmoids+1)/2;
	Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> centers("Centers", numCenters);             
        double bound = 3.;
        centers(0) = -bound; centers(1) = bound;
        unsigned int center_idx = 2;
        
	Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_sigmoids,num_sigmoids}), 
			KOKKOS_LAMBDA (int j, int i)
				{if (i <= j){
					centers(2+(j)+num_sigmoids*i) = -bound+(2*bound)*(i+1)/(j+2);
					}
				}
			);
	
	//int numMaps = 3;	    
        //std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>>> maps(numMaps);
        //for(unsigned int i=0;i<numMaps;++i){
	    std::vector<StridedVector<const double, Kokkos::DefaultExecutionSpace::memory_space>> centers_vec;
	    for(int i = 0; i < dim; i++){
                centers_vec.push_back(centers);
            } 
	    std::shared_ptr<ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>> map = MapFactory::CreateSigmoidTriangular<Kokkos::DefaultExecutionSpace::memory_space>(dim, dim, order,centers_vec, opts);
        //}

        //std::shared_ptr<ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>> map = std::make_shared<ComposedMap<Kokkos::DefaultExecutionSpace::memory_space>>(maps);
        auto numCoeffs = map->numCoeffs;
        unsigned int numPts = pow(10,(5-n));

        // auto tEval = Eigen::VectorXd(nk);
        // auto tLogDet = Eigen::VectorXd(nk);

        std::cout << "\n    NPts = " << numPts << ",  Trial: " << 0 << "/" << nk-1 << std::flush;
	Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> sens("Sensitivities", map->outputDim, numPts);
	Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{map->outputDim,numPts}), 
			KOKKOS_LAMBDA (int i, int j)
				{sens(i,j) = 1.0 + 0.1*i + j;}
			);
	
        for(unsigned int k=0; k<nk;++k){

            // Print the current trial.  use \b to overwrite the previous number
            for(int iii=0; iii< 3 + std::floor(log10(nk))+ std::floor(std::log10(std::max<int>(k,1))); ++iii)
                std::cout << "\b";
            std::cout << k << "/" << nk-1 << std::flush;

            Kokkos::View<const double*> coeffs = gen.Coeffs(numCoeffs);            
            map->SetCoeffs(coeffs); 

            Kokkos::View<double**,Kokkos::LayoutLeft> pts = gen.Points(dim,numPts);
          
            auto start1 = high_resolution_clock::now();
            auto evals = map->Evaluate(pts);
            auto stop1 = high_resolution_clock::now();
            auto duration1 = duration_cast<microseconds>(stop1 - start1);
            tEval(n,k)=duration1.count();

            auto start2 = high_resolution_clock::now();
            auto logDet = map->LogDeterminant(pts);
            auto stop2 = high_resolution_clock::now();
            auto duration2 = duration_cast<microseconds>(stop2 - start2);
            tLogDet(n,k)=duration2.count();

	    auto start3 = high_resolution_clock::now();
            auto Inverse = map->Inverse(pts, evals);
            auto stop3 = high_resolution_clock::now();
            auto duration3 = duration_cast<microseconds>(stop3 - start3);
            tInv(n,k)=duration3.count();

           
	    auto start4 = high_resolution_clock::now();
            auto coeffGrad = map->CoeffGrad(pts, sens);
            auto stop4 = high_resolution_clock::now();
            auto duration4 = duration_cast<microseconds>(stop4 - start4);
            tCoeffGrad(n,k)=duration4.count();


        }
        // tEvalMat_m(n)=tEval.mean();
        // tLogDetMat_m(n)=tLogDet.mean();
        // tEvalMat_s(n)=std::sqrt((tEval - tEval.mean()).square().sum()/(tEval.size()-1));
        // tLogDetMat_s(n)=std::sqrt((tLogDet - tLogDet.mean()).square().sum()/(tLogDet.size()-1));
    }
    
    {
        std::stringstream filename;
        filename << "ST_CPP_eval_d5_to" << order << "_nt"  << backend << ".txt";

        std::ofstream file1(filename.str());  
        if(file1.is_open())  // si l'ouverture a réussi
        {   
        file1 << tEval << "\n";
        }
    }

    {
        std::stringstream filename;
        filename << "ST_CPP_logdet_d5_to" << order << "_nt"  << backend << ".txt";

        std::ofstream file2(filename.str());  
        if(file2.is_open())  // si l'ouverture a réussi
        {   
        file2 << tLogDet << "\n";
        }
    }
    
{
        std::stringstream filename;
        filename << "ST_CPP_inv_d5_to" << order << "_nt" << backend << ".txt";

        std::ofstream file3(filename.str());  
        if(file3.is_open())  // si l'ouverture a réussi
        {   
        file3 << tInv << "\n";
        }
    }
    {
        std::stringstream filename;
        filename << "ST_CPP_coeffgrad_d5_to" << order << "_nt" << backend << ".txt";

        std::ofstream file3(filename.str());  
        if(file3.is_open())  // si l'ouverture a réussi
        {   
        file3 << tCoeffGrad << "\n";
        }
    }
    }

    Kokkos::finalize();
	
    return 0;
}
