#pragma once
#include "para.h"
#include"contants.h"
#include"kernels.h"
#include"TimerClock.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<driver_types.h>
#include<fstream>
#include<iostream>
#include <bitset>
#include "util.h"
class MVTLMfit {
public:

	void checkCudaError(cudaError_t error)
	{
		if (error != cudaSuccess)
		{
			//std::cout << i << std::endl;
			printf("Error in CUDA function.\nError: %s\n", cudaGetErrorString(error));
			getchar();
			exit(EXIT_FAILURE);
		}
	}
	void runResolveKernel(Package* d_lUdpdata, SinglesStruct* d_lSingleData, SinglesStruct* d_lSingleDataBuffer, unsigned lUdpNum, EnegryCorrectData* d_eCorData, TimeCorrectData* d_tCorData, unsigned* d_len) {
		cudaError_t error;
		const int threadsNum = THREADS_RESOLVE;
		const int threadsElem = ELEMS_RESLOVE;
		const int elemsPerBlock = threadsNum * threadsElem;
		long long eventNum = lUdpNum * EVENT_NUM_PACKAGE;
		error = cudaMemset(d_len, 0, sizeof(unsigned));
		checkCudaError(error);

		dim3 dimGrid((eventNum - 1) / elemsPerBlock + 1, 1, 1);
		dim3 dimBlock(threadsNum, 1, 1);
		ResolveKernel << <dimGrid, dimBlock >> > (d_lUdpdata, d_lSingleData, d_lSingleDataBuffer, eventNum, d_eCorData, d_tCorData, d_len);
	}

	void runAddPaddingKernel(SinglesStruct* d_arrayPrimary, SinglesStruct* d_arrayBuffer, unsigned indexStart, unsigned indexEnd)
	{
		if (indexStart == indexEnd)
		{
			return;
		}

		unsigned paddingLength = indexEnd - indexStart;
		unsigned elemsPerThreadBlock = THREADS_PADDING * ELEMS_PADDING;
		dim3 dimGrid((paddingLength - 1) / elemsPerThreadBlock + 1, 1, 1);
		dim3 dimBlock(THREADS_PADDING, 1, 1);

		SinglesStruct maxVal;

		maxVal.timevalue = MAX_VAL;
		addPaddingKernel << <dimGrid, dimBlock >> > (d_arrayPrimary, d_arrayBuffer, indexStart, paddingLength, maxVal);   // 这里会有访问越界
	}

	void addPadding(SinglesStruct* dst, SinglesStruct* dstBuffer, unsigned arrayLength)
	{
		unsigned threadsMergeSort = THREADS_MERGESORT;
		unsigned elemsMergeSort = ELEMS_MERGESORT;
		unsigned elemsPerThreadBlock = threadsMergeSort * elemsMergeSort;
		unsigned arrayLenRoundedUp = std::max(nextPowerOf2(arrayLength), elemsPerThreadBlock);
		runAddPaddingKernel(dst, dstBuffer, arrayLength, arrayLenRoundedUp);
	}

	void runMergeSortKernel(SinglesStruct* d_data, unsigned arrayLength)
	{
		unsigned elemsPerThreadBlock, sharedMemSize;

		elemsPerThreadBlock = THREADS_MERGESORT * ELEMS_MERGESORT;              // 每个线程块操作的元素个数
		// "2 *" because buffer shared memory is used in kernel alongside primary shared memory
		sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*d_data);                 // 创建两倍元素数目的共享内存大小

		// 判断需要执行合并操作的数组长度，是线程块操作元素个数的整数倍，并不需要是2的幂次，因为后面补齐的数据都是有序的
		unsigned arrayLenRoundedUp = roundUp(arrayLength, elemsPerThreadBlock);
		dim3 dimGrid((arrayLenRoundedUp - 1) / elemsPerThreadBlock + 1, 1, 1);
		dim3 dimBlock(THREADS_MERGESORT, 1, 1);

		mergeSortKernel<THREADS_MERGESORT, ELEMS_MERGESORT> << <dimGrid, dimBlock, sharedMemSize >> > (d_data);
	}

	unsigned copyPaddedElements(
		SinglesStruct* d_dataFrom, SinglesStruct* d_dataTo, unsigned arrayLength,
		unsigned sortedBlockSize, unsigned lastPaddingMergePhase
	)
	{
		// 计算主要部分长度和剩余部分长度
		unsigned arrayLenMerge = previousPowerOf2(arrayLength);
		unsigned remainder = arrayLength - arrayLenMerge;

		// 剩余部分需要参与排序的两种情况，情况一：剩余部分长度＞=当前有序序列长度、情况二：主要部分已经全部有序
		// 由于剩余部分如果要参与排序就必须将剩余部分移动到参与排序的那个数组中
		if (remainder >= sortedBlockSize || arrayLenMerge == sortedBlockSize)
		{
			// 计算当前合并的阶段与上次剩余部分参与排序的阶段做差进行判断
			unsigned currentMergePhase = log2((double)(2 * sortedBlockSize));
			unsigned phaseDifference = currentMergePhase - lastPaddingMergePhase;

			if (phaseDifference % 2 == 0)
			{
				cudaError_t error;

				error = cudaMemcpy(d_dataTo, d_dataFrom, remainder * sizeof(*d_dataTo), cudaMemcpyDeviceToDevice);
				checkCudaError(error);

			}

			// 更新剩余部分参与合并排序的阶段
			lastPaddingMergePhase = currentMergePhase;
		}

		return lastPaddingMergePhase;
	}

	void runGenerateRanksKernel(
		SinglesStruct* d_data, unsigned* d_ranksEven, unsigned* d_ranksOdd, unsigned arrayLength, unsigned sortedBlockSize
	)
	{
		unsigned subBlockSize = SUB_BLOCK_SIZE;
		unsigned threadsKernel = THREADS_GEN_RANKS;

		unsigned arrayLenRoundedUp = calculateMergeArraySize(arrayLength, sortedBlockSize);     // 计算需要进行合并的数据规模
		// 根据合并规模计算出所需的子块数，所需的子块数即为端点个数，后面每个端点分配一个线程，分别算出每个端点对应划分出来的子块编号，
		// 每个端点在本身序列中的位置和相对应序列中的位置
		unsigned numAllSamples = (arrayLenRoundedUp - 1) / subBlockSize + 1;
		unsigned threadBlockSize = std::min(numAllSamples, threadsKernel);           // 计算出所需的线程数（每个线程块只启动128个线程）

		dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);        // 根据所需要处理的端点数的多少划分block
		dim3 dimBlock(threadBlockSize, 1, 1);

		generateRanksKernel<SUB_BLOCK_SIZE> << <dimGrid, dimBlock >> > (d_data, d_ranksEven, d_ranksOdd, sortedBlockSize);

	}

	void runMergeKernel(
		SinglesStruct* d_data, SinglesStruct* d_dataBuffer, unsigned* d_ranksEven,
		unsigned* d_ranksOdd, unsigned arrayLength, unsigned sortedBlockSize
	)
	{
		unsigned arrayLenMerge = calculateMergeArraySize(arrayLength, sortedBlockSize);         // 判断剩余部分是否需要加入合并，得到合并的规模
		unsigned mergedBlockSize = 2 * sortedBlockSize;
		// 合并完成后的序列数
		unsigned numMergedBlocks = (arrayLenMerge - 1) / mergedBlockSize + 1;
		unsigned subBlockSize = SUB_BLOCK_SIZE;
		// 合并一对序列需要的线程块数
		unsigned subBlocksPerMergedBlock = (mergedBlockSize - 1) / subBlockSize + 1;

		// 通过端点进行划分，n个端点划分出n+1个子块，n+1个子块使用n+1个线程块进行合并
		dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
		dim3 dimBlock(subBlockSize, 1, 1);

		mergeKernel<SUB_BLOCK_SIZE> << <dimGrid, dimBlock >> > (d_data, d_dataBuffer, d_ranksEven, d_ranksOdd, sortedBlockSize);

	}

	void parallelMergeSort(SinglesStruct* dst, SinglesStruct* dstBuffer, unsigned arrayLength) {
		unsigned* d_ranksEven = NULL;
		unsigned* d_ranksOdd = NULL;

		unsigned elemsPerThreadBlock = THREADS_MERGESORT * ELEMS_MERGESORT;
		unsigned arrayLenRoundedUp = std::max(nextPowerOf2(arrayLength), elemsPerThreadBlock);
		unsigned ranksLength = (arrayLenRoundedUp - 1) / SUB_BLOCK_SIZE + 1;         // 求出子块的数目

		cudaError_t error;
		error = cudaMalloc((void**)&d_ranksEven, 2 * ranksLength * sizeof(*d_ranksEven));
		checkCudaError(error);
		error = cudaMalloc((void**)&d_ranksOdd, 2 * ranksLength * sizeof(*d_ranksOdd));
		checkCudaError(error);

		unsigned sortedBlockSize;
		sortedBlockSize = THREADS_MERGESORT * ELEMS_MERGESORT;

		unsigned lastPaddingMergePhase = log2((double)(sortedBlockSize));                       // 计算合并阶段
		unsigned arrayLenPrevPower2 = previousPowerOf2(arrayLength);

		//TimerClock tc1;
		//tc1.update();
		addPadding(dst, dstBuffer, arrayLength);           // 两个数组都填满2的幂次（可能会导致越界）
		//cudaDeviceSynchronize();
		//std::cout << std::endl;
		//std::cout << "add Padding times: " << tc1.getSecond() << "s" << std::endl;

		//SinglesStruct* h_lSingles = new SinglesStruct[nextPowerOf2(arrayLength)];

		//cudaMemcpy(h_lSingles, (void*)dst, nextPowerOf2(arrayLength) * sizeof(*h_lSingles), cudaMemcpyDeviceToHost);
		//for (int i = arrayLength - 10; i < nextPowerOf2(arrayLength); i += 50000) {
		//	std::cout << h_lSingles[i].timevalue << " ";
		//}
		//std::cout << std::endl;

		//delete[] h_lSingles;

		//TimerClock tc2;
		//tc2.update();

		runMergeSortKernel(dst, arrayLength);         // data局部有序，databuffer无序 每1024个元素有序
		//cudaDeviceSynchronize();
		//std::cout << std::endl;
		//std::cout << "merge sort local times: " << tc2.getSecond() << "s" << std::endl;



		//TimerClock tc3;
		//tc3.update();
		while (sortedBlockSize < arrayLength)
		{
			SinglesStruct* temp = dst;
			dst = dstBuffer;
			dstBuffer = temp;


			lastPaddingMergePhase = copyPaddedElements(                            // 判断剩余部分是否需要参与排序，如果需要排序，是否应该进行数据转移，并记录最近一次剩余部分参与排序的阶段
				dst + arrayLenPrevPower2, dstBuffer + arrayLenPrevPower2,
				arrayLength, sortedBlockSize, lastPaddingMergePhase
			);

			runGenerateRanksKernel(dstBuffer, d_ranksEven, d_ranksOdd, arrayLength, sortedBlockSize);

			runMergeKernel(dstBuffer, dst, d_ranksEven, d_ranksOdd, arrayLength, sortedBlockSize);

			sortedBlockSize *= 2;

		}
		//cudaDeviceSynchronize();

		//std::cout << std::endl;
		//std::cout << "merge sort global times: " << tc3.getSecond() << "s" << std::endl;


		error = cudaFree(d_ranksEven);
		checkCudaError(error);

		error = cudaFree(d_ranksOdd);
		checkCudaError(error);

	}

	void runCoinTimeKernel(SinglesStruct* d_data, CoinStruct* mCoins, unsigned arrayLength, unsigned* realLength, unsigned& offset) {
		unsigned elemsPerThreadBlock;

		unsigned* d_realLength = NULL;
		unsigned* d_index = NULL;
		unsigned* d_realIndex = NULL;

		cudaError_t error;
		CoinStruct* d_mCoins;
		error = cudaMalloc((void**)&d_mCoins, ((arrayLength + 1) / 2) * sizeof(CoinStruct));
		checkCudaError(error);
		error = cudaMalloc((void**)&d_index, arrayLength * sizeof(*d_index));
		checkCudaError(error);
		error = cudaMalloc((void**)&d_realIndex, arrayLength * sizeof(*d_index));
		checkCudaError(error);
		error = cudaMalloc((void**)&d_realLength, sizeof(*d_realLength));
		checkCudaError(error);
		error = cudaMemset(d_realLength, 0, sizeof(unsigned));
		checkCudaError(error);
		unsigned* gIndex = NULL;
		unsigned* sIndex = NULL;

		elemsPerThreadBlock = THREADS_COINTIME * ELEMS_COINTIME;              // 每个线程块操作的元素个数
		unsigned blockNum = (arrayLength - 1) / elemsPerThreadBlock + 1;                  // 总的线程块数

		error = cudaMalloc((void**)&gIndex, blockNum * sizeof(*gIndex));
		checkCudaError(error);
		error = cudaMalloc((void**)&sIndex, blockNum * THREADS_COINTIME * sizeof(*sIndex));                      // 大小应与线程数一致
		checkCudaError(error);

		dim3 dimGrid(blockNum, 1, 1);
		dim3 dimBlock(THREADS_COINTIME, 1, 1);

		// step1：找到每个元素在一个时间窗口内最远能匹配到的元素
		coinTimeKernel<THREADS_COINTIME, ELEMS_COINTIME> << <dimGrid, dimBlock >> > (d_data, d_index, arrayLength, TIME_WINDOW);
		//std::cout << arrayLength << std::endl;

		// step2：找到实际情况中每个元素的匹配元素
		//coinTimeKernel2<THREADS_COINTIME, ELEMS_COINTIME> << <dimGrid, dimBlock >> > (d_data, d_index, d_realIndex, arrayLength, mCoinPetPara.m_nTimeWindow);
		coinTimeKernel2<THREADS_COINTIME, ELEMS_COINTIME> << <dimGrid, dimBlock >> > (d_data, d_mCoins, d_index, d_realIndex, gIndex, sIndex, arrayLength, TIME_WINDOW, d_realLength);

		//unsigned* realInedx = new unsigned[arrayLength];
		//error = cudaMemcpy(realInedx, (void*)d_realIndex, arrayLength * sizeof(*d_realIndex), cudaMemcpyDeviceToHost);
		//checkCudaError(error);

		//unsigned temp = 0;
		//for (unsigned i = 0; i < arrayLength; ++i) {
		//	if (realInedx[i] != 0) {
		//		//if (temp % 5000 == 0) {
		//		//	std::cout << temp << ":" << i << "-" << realInedx[i] << " ";
		//		//}
		//		++temp;
		//	}
		//}
		//std::cout << temp << std::endl;
		//error = cudaMemcpy(realLength, (void*)d_realLength, sizeof(*d_realLength), cudaMemcpyDeviceToHost);
		//checkCudaError(error);

		//std::cout << std::endl;
		//std::cout << *realLength << std::endl;


		// step3：根据匹配元素进行合并剔除
		coinTimeKernel3<THREADS_COINTIME, ELEMS_COINTIME> << <dimGrid, dimBlock >> > (d_data, d_mCoins, d_realIndex, gIndex, sIndex, arrayLength);

		error = cudaMemcpy(realLength, (void*)d_realLength, sizeof(*d_realLength), cudaMemcpyDeviceToHost);
		checkCudaError(error);

		//std::cout << std::endl;
		std::cout << "realLength:"<< * realLength << std::endl;
		error = cudaMemcpy(mCoins + offset, d_mCoins, *realLength * sizeof(CoinStruct), cudaMemcpyDeviceToHost);
		checkCudaError(error);
		//
		//for (int i = 0; i < *realLength; i+=50) {
		//	std::cout << std::fixed << mCoins[i + offset].nCoinStruct[0].timevalue << "       " << mCoins[i + offset].nCoinStruct[1].timevalue << std::endl;
		//}
		offset += *realLength;

		error = cudaFree(d_mCoins);
		checkCudaError(error);
		error = cudaFree(gIndex);
		checkCudaError(error);
		error = cudaFree(sIndex);
		checkCudaError(error);
		error = cudaFree(d_index);
		checkCudaError(error);
		error = cudaFree(d_realIndex);
		checkCudaError(error);
		error = cudaFree(d_realLength);
		checkCudaError(error);

	}

	void start(TimerClock& tc) {
		cudaError_t error;
		const unsigned headerLen1 = 510;
		const unsigned headerLen2 = 150;
		const unsigned SingleUdpLen = 1154;
		const unsigned numEvents = 48;
		const unsigned eventLength = 24;
		const unsigned binfileNum = 10;
		const unsigned corfileNum = 2;
		const unsigned totalNum = 12;
		const unsigned ChannelCounts = 72;
		const unsigned IPCounts = 88;
		const unsigned Rings = 2;
		std::ifstream lSingleFiles[totalNum];
		unsigned lFileSize[totalNum];
		std::string path1 = "../data";
		std::string path2 = "../Tool";
		std::string corName[corfileNum] = { "/TimeCalibration.bin", "/EnergyCalibration.bin" };
		std::string binName[binfileNum] = { "/20211130-104130142.bin", "/20211130-104131374.bin", "/20211130-104132606.bin", "/20211130-104133839.bin", "/20211130-104135071.bin",
								"/20211130-104136304.bin","/20211130-104137536.bin", "/20211130-104138769.bin", "/20211130-104140002.bin", "/20211130-104141234.bin" };
		// 读取样本文件
		for (int i = 0; i < 12; ++i) {
			if (i < 10) {
				std::string path = path1 + binName[i];
				lSingleFiles[i].open(path, std::ios::binary);
				if (!lSingleFiles[i].is_open()) {
					printf("Open file %s failed\n", path.c_str());
					return;
				}
				lSingleFiles[i].seekg(0, std::ios_base::end);
				lFileSize[i] = lSingleFiles[i].tellg();
				printf("filesize = %f\n", double(lFileSize[i]));

				lSingleFiles[i].seekg(headerLen1, std::ios_base::beg);
			}
			else {
				std::string path = path2 + corName[i - 10];
				lSingleFiles[i].open(path, std::ios::binary);
				if (!lSingleFiles[i].is_open()) {
					printf("Open file %s failed\n", path.c_str());
					return;
				}
				lSingleFiles[i].seekg(0, std::ios_base::end);
				lFileSize[i] = lSingleFiles[i].tellg();
				printf("filesize = %f\n", double(lFileSize[i]));
				lSingleFiles[i].seekg(headerLen2, std::ios_base::beg);
			}
		}

		const unsigned TableLine = ChannelCounts * IPCounts * Rings;
		EnegryCorrectData* eCorData = new EnegryCorrectData[TableLine];
		TimeCorrectData* tCorData = new TimeCorrectData[TableLine];
		for (unsigned i = 0; i < ChannelCounts * IPCounts * Rings; ++i) {
			lSingleFiles[10].read((char*)(tCorData[i].IP_CH), 2 * sizeof(uint16_t));
			lSingleFiles[10].read((char*)(&(tCorData[i].offset)), 1 * sizeof(float));
			lSingleFiles[11].read((char*)eCorData[i].IP_CH, 2 * sizeof(uint16_t));
			lSingleFiles[11].read((char*)eCorData[i].par, 7 * sizeof(float));
		}


		//for (unsigned i = 0; i < ChannelCounts * IPCounts * Rings; ++i) {
		//	//std::cout << tCorData[i].IP_CH[0] << "   " << tCorData[i].IP_CH[1] << std::endl;
		//	std::cout << eCorData[i].IP_CH[0] << "   " << eCorData[i].IP_CH[1] << "    " << eCorData[i].par[0] << "    " << eCorData[i].par[1] << "    " << eCorData[i].par[2]
		//		<< "    " << eCorData[i].par[3] << "    " << eCorData[i].par[4] << "    " << eCorData[i].par[5] << "    " << eCorData[i].par[6] << std::endl;
		//}

		unsigned UdpTotalNum = (lFileSize[0] - headerLen1) / SingleUdpLen;
		long long  eventNum = (long long)UdpTotalNum * numEvents;
		Package* lUdpdata = new Package[UdpTotalNum];
		float* energy = new float[eventNum];
		//float* d_energy;
		//error = cudaMalloc((void**)&d_energy, eventNum * sizeof(double));
		//checkCudaError(error);

		//可能有隐患 分配内存
		SinglesStruct* d_lSingleData;
		error = cudaMalloc((void**)&d_lSingleData, eventNum * sizeof(SinglesStruct));
		checkCudaError(error);

		SinglesStruct* d_lSingleDataBuffer;
		error = cudaMalloc((void**)&d_lSingleDataBuffer, eventNum * sizeof(SinglesStruct));
		checkCudaError(error);

		//转换成ReslovePackage类型
		Package* d_lUdpdata;
		error = cudaMalloc((void**)&d_lUdpdata, UdpTotalNum * sizeof(Package));
		checkCudaError(error);

		EnegryCorrectData* d_eCorData;
		error = cudaMalloc((void**)&d_eCorData, TableLine * sizeof(EnegryCorrectData));
		checkCudaError(error);

		TimeCorrectData* d_tCorData;
		error = cudaMalloc((void**)&d_tCorData, TableLine * sizeof(TimeCorrectData));
		checkCudaError(error);

		error = cudaMemcpy(d_eCorData, eCorData, TableLine * sizeof(EnegryCorrectData), cudaMemcpyHostToDevice);
		checkCudaError(error);

		error = cudaMemcpy(d_tCorData, tCorData, TableLine * sizeof(TimeCorrectData), cudaMemcpyHostToDevice);
		checkCudaError(error);


		unsigned* d_len;
		error = cudaMalloc((void**)&d_len, 1 * sizeof(unsigned));
		checkCudaError(error);

		unsigned* h_len = new unsigned;
		unsigned* realLength = new unsigned;

		CoinStruct* mCoins = new CoinStruct[eventNum * 5];

		//uint16_t ipmin = UINT16_MAX;
		//uint16_t ipmax = 0;
		unsigned offset = 0;
		for (unsigned i = 0; i < 10; ++i) {
			TimerClock tc2;
			tc2.update();
			for (int j = 0; j < UdpTotalNum; ++j) {
				lSingleFiles[i].read((char*)(lUdpdata[j].nEvents), numEvents * eventLength * sizeof(unsigned char));
				//lSingleFiles[i / 2].read((char*)(lUdpdata[j].nTail), sizeof(uint16_t));
				lSingleFiles[i].read((char*)(lUdpdata[j].nTail), 2 * sizeof(UINT8));
				//std::cout << std::bitset<8>(lUdpdata[j].nTail[0]) << "    " << std::bitset<8>(lUdpdata[j].nTail[1]) << std::endl;
				//if (lUdpdata[j].nTail[0] < ipmin) {
				//	ipmin = lUdpdata[j].nTail[0];
				//}
				//else if (lUdpdata[j].nTail[0] > ipmax) {
				//	ipmax = lUdpdata[j].nTail[0];
				//}
			}

			std::cout << "read time cost: " << tc2.getSecond() << "s" << std::endl;
			//拷贝数据到GPU 准备进行转换
			error = cudaMemcpy(d_lUdpdata, lUdpdata, UdpTotalNum * sizeof(Package), cudaMemcpyHostToDevice);

			TimerClock tc1;
			tc1.update();

			//GPU并行转换成SingleStruce类型数据
			runResolveKernel(d_lUdpdata, d_lSingleData, d_lSingleDataBuffer, UdpTotalNum, d_eCorData, d_tCorData, d_len);
			//cudaDeviceSynchronize();
			error = cudaMemcpy(h_len, d_len, sizeof(unsigned), cudaMemcpyDeviceToHost);
			checkCudaError(error);
			//std::cout << "h_len:" << *h_len << std::endl;

			parallelMergeSort(d_lSingleData, d_lSingleDataBuffer, *h_len);
			//std::cout << "time cost:" << tc1.getSecond() << "s" << std::endl;
			//
			
			// 判断是否需要转换指针
			unsigned elemsPerInitMergeSort = THREADS_MERGESORT * ELEMS_MERGESORT;
			unsigned arrayLenRoundedUp = std::max(nextPowerOf2(*h_len), elemsPerInitMergeSort);
			unsigned numMergePhases = log2((double)arrayLenRoundedUp) - log2((double)elemsPerInitMergeSort);


			if (numMergePhases % 2 == 1)
			{
				SinglesStruct* temp = d_lSingleData;
				d_lSingleData = d_lSingleDataBuffer;
				d_lSingleDataBuffer = temp;
			}

			SinglesStruct* data = new SinglesStruct[*h_len];
			error = cudaMemcpy(data, d_lSingleData, *h_len * sizeof(SinglesStruct), cudaMemcpyDeviceToHost);
			checkCudaError(error);

			//std::string lSavePath = "./thread" + std::to_string(0) + ".singles";
			//std::ofstream lSingleFile(lSavePath, std::ios::binary);
			//if (!lSingleFile.is_open()) {
			//	printf("Open file %s failed\n", lSavePath.c_str());
			//	return;
			//}
			//lSingleFile.write((char*)data, *h_len * sizeof(SinglesStruct));
			//for (int i = 0; i < *h_len; ++i) {
			//	std::cout << data[i].timevalue << "     " << data[i].globalCrystalIndex << std::endl;
			//}
			//时间筛选
			runCoinTimeKernel(d_lSingleData, mCoins, *h_len, realLength, offset);

		}


		error = cudaFree(d_lUdpdata);
		checkCudaError(error);

		delete[] lUdpdata;
		delete[] energy;

		std::cout << "total time cost :" << tc.getSecond() << " s" << std::endl;
		std::cout << "total rate :" << lFileSize[0] * 10.0 / tc.getSecond() / 1000.0 / 1000.0 << "MB/s" << std::endl;
		system("pause");
	}
};