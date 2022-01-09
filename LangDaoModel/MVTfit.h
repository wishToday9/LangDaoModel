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
		addPaddingKernel << <dimGrid, dimBlock >> > (d_arrayPrimary, d_arrayBuffer, indexStart, paddingLength, maxVal);   // ������з���Խ��
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

		elemsPerThreadBlock = THREADS_MERGESORT * ELEMS_MERGESORT;              // ÿ���߳̿������Ԫ�ظ���
		// "2 *" because buffer shared memory is used in kernel alongside primary shared memory
		sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*d_data);                 // ��������Ԫ����Ŀ�Ĺ����ڴ��С

		// �ж���Ҫִ�кϲ����������鳤�ȣ����߳̿����Ԫ�ظ�������������������Ҫ��2���ݴΣ���Ϊ���油������ݶ��������
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
		// ������Ҫ���ֳ��Ⱥ�ʣ�ಿ�ֳ���
		unsigned arrayLenMerge = previousPowerOf2(arrayLength);
		unsigned remainder = arrayLength - arrayLenMerge;

		// ʣ�ಿ����Ҫ���������������������һ��ʣ�ಿ�ֳ��ȣ�=��ǰ�������г��ȡ����������Ҫ�����Ѿ�ȫ������
		// ����ʣ�ಿ�����Ҫ��������ͱ��뽫ʣ�ಿ���ƶ�������������Ǹ�������
		if (remainder >= sortedBlockSize || arrayLenMerge == sortedBlockSize)
		{
			// ���㵱ǰ�ϲ��Ľ׶����ϴ�ʣ�ಿ�ֲ�������Ľ׶���������ж�
			unsigned currentMergePhase = log2((double)(2 * sortedBlockSize));
			unsigned phaseDifference = currentMergePhase - lastPaddingMergePhase;

			if (phaseDifference % 2 == 0)
			{
				cudaError_t error;

				error = cudaMemcpy(d_dataTo, d_dataFrom, remainder * sizeof(*d_dataTo), cudaMemcpyDeviceToDevice);
				checkCudaError(error);

			}

			// ����ʣ�ಿ�ֲ���ϲ�����Ľ׶�
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

		unsigned arrayLenRoundedUp = calculateMergeArraySize(arrayLength, sortedBlockSize);     // ������Ҫ���кϲ������ݹ�ģ
		// ���ݺϲ���ģ�����������ӿ�����������ӿ�����Ϊ�˵����������ÿ���˵����һ���̣߳��ֱ����ÿ���˵��Ӧ���ֳ������ӿ��ţ�
		// ÿ���˵��ڱ��������е�λ�ú����Ӧ�����е�λ��
		unsigned numAllSamples = (arrayLenRoundedUp - 1) / subBlockSize + 1;
		unsigned threadBlockSize = std::min(numAllSamples, threadsKernel);           // �����������߳�����ÿ���߳̿�ֻ����128���̣߳�

		dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);        // ��������Ҫ����Ķ˵����Ķ��ٻ���block
		dim3 dimBlock(threadBlockSize, 1, 1);

		generateRanksKernel<SUB_BLOCK_SIZE> << <dimGrid, dimBlock >> > (d_data, d_ranksEven, d_ranksOdd, sortedBlockSize);

	}

	void runMergeKernel(
		SinglesStruct* d_data, SinglesStruct* d_dataBuffer, unsigned* d_ranksEven,
		unsigned* d_ranksOdd, unsigned arrayLength, unsigned sortedBlockSize
	)
	{
		unsigned arrayLenMerge = calculateMergeArraySize(arrayLength, sortedBlockSize);         // �ж�ʣ�ಿ���Ƿ���Ҫ����ϲ����õ��ϲ��Ĺ�ģ
		unsigned mergedBlockSize = 2 * sortedBlockSize;
		// �ϲ���ɺ��������
		unsigned numMergedBlocks = (arrayLenMerge - 1) / mergedBlockSize + 1;
		unsigned subBlockSize = SUB_BLOCK_SIZE;
		// �ϲ�һ��������Ҫ���߳̿���
		unsigned subBlocksPerMergedBlock = (mergedBlockSize - 1) / subBlockSize + 1;

		// ͨ���˵���л��֣�n���˵㻮�ֳ�n+1���ӿ飬n+1���ӿ�ʹ��n+1���߳̿���кϲ�
		dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
		dim3 dimBlock(subBlockSize, 1, 1);

		mergeKernel<SUB_BLOCK_SIZE> << <dimGrid, dimBlock >> > (d_data, d_dataBuffer, d_ranksEven, d_ranksOdd, sortedBlockSize);

	}

	void parallelMergeSort(SinglesStruct* dst, SinglesStruct* dstBuffer, unsigned arrayLength) {
		unsigned* d_ranksEven = NULL;
		unsigned* d_ranksOdd = NULL;

		unsigned elemsPerThreadBlock = THREADS_MERGESORT * ELEMS_MERGESORT;
		unsigned arrayLenRoundedUp = std::max(nextPowerOf2(arrayLength), elemsPerThreadBlock);
		unsigned ranksLength = (arrayLenRoundedUp - 1) / SUB_BLOCK_SIZE + 1;         // ����ӿ����Ŀ

		cudaError_t error;
		error = cudaMalloc((void**)&d_ranksEven, 2 * ranksLength * sizeof(*d_ranksEven));
		checkCudaError(error);
		error = cudaMalloc((void**)&d_ranksOdd, 2 * ranksLength * sizeof(*d_ranksOdd));
		checkCudaError(error);

		unsigned sortedBlockSize;
		sortedBlockSize = THREADS_MERGESORT * ELEMS_MERGESORT;

		unsigned lastPaddingMergePhase = log2((double)(sortedBlockSize));                       // ����ϲ��׶�
		unsigned arrayLenPrevPower2 = previousPowerOf2(arrayLength);

		//TimerClock tc1;
		//tc1.update();
		addPadding(dst, dstBuffer, arrayLength);           // �������鶼����2���ݴΣ����ܻᵼ��Խ�磩
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

		runMergeSortKernel(dst, arrayLength);         // data�ֲ�����databuffer���� ÿ1024��Ԫ������
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


			lastPaddingMergePhase = copyPaddedElements(                            // �ж�ʣ�ಿ���Ƿ���Ҫ�������������Ҫ�����Ƿ�Ӧ�ý�������ת�ƣ�����¼���һ��ʣ�ಿ�ֲ�������Ľ׶�
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

		elemsPerThreadBlock = THREADS_COINTIME * ELEMS_COINTIME;              // ÿ���߳̿������Ԫ�ظ���
		unsigned blockNum = (arrayLength - 1) / elemsPerThreadBlock + 1;                  // �ܵ��߳̿���

		error = cudaMalloc((void**)&gIndex, blockNum * sizeof(*gIndex));
		checkCudaError(error);
		error = cudaMalloc((void**)&sIndex, blockNum * THREADS_COINTIME * sizeof(*sIndex));                      // ��СӦ���߳���һ��
		checkCudaError(error);

		dim3 dimGrid(blockNum, 1, 1);
		dim3 dimBlock(THREADS_COINTIME, 1, 1);

		// step1���ҵ�ÿ��Ԫ����һ��ʱ�䴰������Զ��ƥ�䵽��Ԫ��
		coinTimeKernel<THREADS_COINTIME, ELEMS_COINTIME> << <dimGrid, dimBlock >> > (d_data, d_index, arrayLength, TIME_WINDOW);
		//std::cout << arrayLength << std::endl;

		// step2���ҵ�ʵ�������ÿ��Ԫ�ص�ƥ��Ԫ��
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


		// step3������ƥ��Ԫ�ؽ��кϲ��޳�
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
		// ��ȡ�����ļ�
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

		//���������� �����ڴ�
		SinglesStruct* d_lSingleData;
		error = cudaMalloc((void**)&d_lSingleData, eventNum * sizeof(SinglesStruct));
		checkCudaError(error);

		SinglesStruct* d_lSingleDataBuffer;
		error = cudaMalloc((void**)&d_lSingleDataBuffer, eventNum * sizeof(SinglesStruct));
		checkCudaError(error);

		//ת����ReslovePackage����
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
			//�������ݵ�GPU ׼������ת��
			error = cudaMemcpy(d_lUdpdata, lUdpdata, UdpTotalNum * sizeof(Package), cudaMemcpyHostToDevice);

			TimerClock tc1;
			tc1.update();

			//GPU����ת����SingleStruce��������
			runResolveKernel(d_lUdpdata, d_lSingleData, d_lSingleDataBuffer, UdpTotalNum, d_eCorData, d_tCorData, d_len);
			//cudaDeviceSynchronize();
			error = cudaMemcpy(h_len, d_len, sizeof(unsigned), cudaMemcpyDeviceToHost);
			checkCudaError(error);
			//std::cout << "h_len:" << *h_len << std::endl;

			parallelMergeSort(d_lSingleData, d_lSingleDataBuffer, *h_len);
			//std::cout << "time cost:" << tc1.getSecond() << "s" << std::endl;
			//
			
			// �ж��Ƿ���Ҫת��ָ��
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
			//ʱ��ɸѡ
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