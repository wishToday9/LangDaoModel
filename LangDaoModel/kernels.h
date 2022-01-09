#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "contants.h"
#include "para.h"
#include "device_functions.h"
#include "kernels.h"

const int num_params = 4;  //��������
const int total_data = 8;
const int m = 8; //���ݸ���

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

template <unsigned numThreads, unsigned elemsThread>
inline __device__ void calcDataBlockLength(unsigned& offset, unsigned& dataBlockLength, unsigned arrayLength)
{
	unsigned elemsPerThreadBlock = numThreads * elemsThread;            // ����ÿ���߳̿�Ҫ�����������
	offset = blockIdx.x * elemsPerThreadBlock;
	dataBlockLength = offset + elemsPerThreadBlock <= arrayLength ? elemsPerThreadBlock : arrayLength - offset;       // �����һ���߳̿�����⴦��
}

template <unsigned blockSize>
inline __device__ unsigned intraWarpScan(volatile unsigned* scanTile, unsigned val) {
	unsigned index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));

	scanTile[index] = 0;              // ��ǰ��һ������
	index += min(blockSize, WARP_SIZE);
	scanTile[index] = val;

	if (blockSize >= 2)
	{
		scanTile[index] += scanTile[index - 1];
	}

	if (blockSize >= 4)
	{
		scanTile[index] += scanTile[index - 2];
	}
	if (blockSize >= 8)
	{
		scanTile[index] += scanTile[index - 4];
	}
	if (blockSize >= 16)
	{
		scanTile[index] += scanTile[index - 8];
	}
	if (blockSize >= 32)
	{
		scanTile[index] += scanTile[index - 16];
	}

	// ���Ԫ�ص�ֵ���кϲ�
	return scanTile[index] - val;

}


template <unsigned blockSize>
inline __device__ unsigned intraBlockScan(unsigned val) {
	__shared__ unsigned scanTile[blockSize * 2];
	unsigned warpIdx = threadIdx.x / WARP_SIZE;
	unsigned laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp


	unsigned warpResult = intraWarpScan<blockSize>(scanTile, val);
	__syncthreads();


	if (laneIdx == WARP_SIZE - 1)                 // �õ�32��ֵ���ܺͷ��ڶ�Ӧ��warpIdx��
	{
		scanTile[warpIdx] = warpResult + val;
	}
	__syncthreads();


	if (threadIdx.x < WARP_SIZE)                  // ��������һ��warp���в���
	{
		scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
	}
	__syncthreads();


	return warpResult + scanTile[warpIdx] + val;


}

inline __global__ void ResolveKernel(Package* d_lUdpdata, SinglesStruct* d_lSingleData, SinglesStruct* d_lSingleDataBuffer, unsigned eventNum,
					EnegryCorrectData* d_eCorData, TimeCorrectData* d_tCorData, unsigned* d_len) {
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREADS_RESOLVE, ELEMS_RESLOVE>(offset, dataBlockLength, eventNum);
	UINT16 nMaxBin[] = { 55,54,55 };
	unsigned local = 0, scan = 0;
	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_RESOLVE) {

		unsigned temp1 = (offset + tx) % EVENT_NUM_PACKAGE;
		unsigned temp2 = (offset + tx) / EVENT_NUM_PACKAGE;
		const Pulse& ev = d_lUdpdata[temp2].nEvents[temp1];
		unsigned nChannelId = ev.nData[3];

		uint16_t ip_ = *reinterpret_cast<uint16_t*>(d_lUdpdata[temp2].nTail);
		unsigned d_index = (ip_ - 514) * CHANNEL_COUNTS + nChannelId - 1;

		int nCurMaxBin = (ev.nData[3] <= 24 ? nMaxBin[0] : (ev.nData[3] <= 48 ? nMaxBin[1] : nMaxBin[2]));
		double fTmp = ev.nData[7] * 4294967296.0 + ev.nData[6] * 16777216.0 + ev.nData[5] * 65536.0 + ev.nData[4] * 256.0;
		double x[8];
		//�����¼���ʱ���
		x[0] = (fTmp + ev.nData[9]) * 5.0 - (ev.nData[8] - 128) * 5.0 / nCurMaxBin;
		x[1] = (fTmp + ev.nData[13]) * 5.0 - (ev.nData[12] - 128) * 5.0 / nCurMaxBin;
		x[2] = (fTmp + ev.nData[17]) * 5.0 - (ev.nData[16] - 128) * 5.0 / nCurMaxBin;
		x[3] = (fTmp + ev.nData[21]) * 5.0 - (ev.nData[20] - 128) * 5.0 / nCurMaxBin;
		x[4] = (fTmp + ev.nData[23]) * 5.0 - (192 - ev.nData[22]) * 5.0 / nCurMaxBin;
		x[5] = (fTmp + ev.nData[19]) * 5.0 - (192 - ev.nData[18]) * 5.0 / nCurMaxBin;
		x[6] = (fTmp + ev.nData[15]) * 5.0 - (192 - ev.nData[14]) * 5.0 / nCurMaxBin;
		x[7] = (fTmp + ev.nData[11]) * 5.0 - (192 - ev.nData[10]) * 5.0 / nCurMaxBin;

		x[1] += (ev.nData[13] - ev.nData[9] < -50 ? 1280 : (ev.nData[13] - ev.nData[9] > 50 ? -1280 : 0));
		x[2] += (ev.nData[17] - ev.nData[9] < -50 ? 1280 : (ev.nData[17] - ev.nData[9] > 50 ? -1280 : 0));
		x[3] += (ev.nData[21] - ev.nData[9] < -50 ? 1280 : (ev.nData[21] - ev.nData[9] > 50 ? -1280 : 0));
		x[4] += (ev.nData[23] < ev.nData[9] ? 1280 : 0);
		x[5] += (ev.nData[19] < ev.nData[9] ? 1280 : 0);
		x[6] += (ev.nData[15] < ev.nData[9] ? 1280 : 0);
		x[7] += (ev.nData[11] < ev.nData[9] ? 1280 : 0);
		d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].timevalue = x[0] + d_tCorData[d_index].offset;

		double t1 = x[0];
		x[0] = 0;
		x[1] = x[1] - t1;
		x[2] = x[2] - t1;
		x[3] = x[3] - t1;
		x[4] = x[4] - t1;
		x[5] = x[5] - t1;
		x[6] = x[6] - t1;
		x[7] = x[7] - t1;

		d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].globalCrystalIndex = ip_ - 514;

		unsigned energyThresholds[8] = { 20, 90, 160, 230,230 ,160, 90, 20 };

		float M = 0;
		float N = 0;

		if (d_eCorData->par[1] < 0.000001 && d_eCorData->par[2] < 0.000001 && d_eCorData->par[3] < 0.000001)
		{
			d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy = -1;
		}
		else {

			//�ʵ�ģ��
			for (int i = 0; i < 8; i++)
			{
				//���Ӷ�̬��ֵ�ж�
				if (x[i] < -100.0)
				{
					continue;
				}
				M = static_cast<float>(M + energyThresholds[i] * exp(d_eCorData->par[2] * (d_eCorData->par[1] - x[i]) - exp(d_eCorData->par[3] * (d_eCorData->par[1] - x[i]))));
				N = static_cast<float>(N + pow(exp(d_eCorData->par[2] * (d_eCorData->par[1] - x[i]) - exp(d_eCorData->par[3] * (d_eCorData->par[1] - x[i]))), 2));
			}

			if (N < 0.000001)
			{
				d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy = 0;


			}
			else {
				d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy = M / N;
			}
		}


		if (d_eCorData->par[5] > 0.0)
		{
			float fac = d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy * d_eCorData->par[0] / d_eCorData->par[5];
			if (fac < 1.0)
			{
				d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy = log(1.f - fac) / d_eCorData->par[6];
			}
			else
			{
				//energy_ = energy_ * energyCaliTable[index].K;
				d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy = -1;
			}
		}
		else
		{
			d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy = d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy * d_eCorData->par[0];
		}

		if (d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy >= 450.f && d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy <= 650.f) {
			local++;
		}
	}
	__shared__ unsigned global;
	scan = intraBlockScan<THREADS_RESOLVE>(local);
	__syncthreads();
	if (threadIdx.x == THREADS_RESOLVE - 1) {
		global = atomicAdd(d_len, scan);                                  // ԭ�Ӳ���������һ��������
	}
	__syncthreads();
	unsigned index = global + scan - local;
	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_RESOLVE) {
		unsigned temp1 = (offset + tx) % EVENT_NUM_PACKAGE;
		unsigned temp2 = (offset + tx) / EVENT_NUM_PACKAGE;
		if (d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy >= 450.f && d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1].energy <= 650.f) {
			d_lSingleData[index++] = d_lSingleDataBuffer[(temp2)*EVENT_NUM_PACKAGE + temp1];
		}
	}
}

template <unsigned stride>
inline __device__ int binarySearchExclusive(SinglesStruct* dataArray, SinglesStruct target, int indexStart, int indexEnd)
{
	while (indexStart <= indexEnd)
	{
		// Floor to multiplier of stride - needed for strides > 1
		int index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ UINT_MAX);

		if (target.timevalue < dataArray[index].timevalue)
		{
			indexEnd = index - stride;
		}
		else
		{
			indexStart = index + stride;
		}
	}

	return indexStart;
}


inline __device__ int binarySearchExclusive(SinglesStruct* dataArray, SinglesStruct target, int indexStart, int indexEnd)
{
	return binarySearchExclusive<1>(dataArray, target, indexStart, indexEnd);
}



template <unsigned stride>
inline __device__ int binarySearchInclusive(SinglesStruct* dataArray, SinglesStruct target, int indexStart, int indexEnd)
{
	while (indexStart <= indexEnd)
	{
		// С�ڸ�ֵ���ܱ�strike�����������
		int index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ UINT_MAX);   // �ҵ�ƫ���Ԫ��   

		if (target.timevalue <= dataArray[index].timevalue)
		{
			indexEnd = index - stride;
		}
		else
		{
			indexStart = index + stride;
		}
	}

	return indexStart;
}


inline __device__ int binarySearchInclusive(SinglesStruct* dataArray, SinglesStruct target, int indexStart, int indexEnd)
{
	return binarySearchInclusive<1>(dataArray, target, indexStart, indexEnd);
}





__global__ void addPaddingKernel(SinglesStruct* arrayPrimary, SinglesStruct* arrayBuffer, unsigned start, unsigned paddingLength, SinglesStruct tempVal)
{
	unsigned offset, dataBlockLength;
	calcDataBlockLength<THREADS_PADDING, ELEMS_PADDING>(offset, dataBlockLength, paddingLength);       // ����ÿ��block��ƫ�����ͳ��ȣ�������������ã�
	offset += start;

	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_PADDING)
	{
		unsigned index = offset + tx;
		arrayPrimary[index] = tempVal;
		arrayBuffer[index] = tempVal;

	}
}

template <unsigned threadsMerge, unsigned elemsThreadMerge>
__global__ void mergeSortKernel(SinglesStruct* dataTable)
{
	extern __shared__ SinglesStruct mergeSortTile[];

	unsigned elemsPerThreadBlock = threadsMerge * elemsThreadMerge;
	SinglesStruct* globalDataTable = dataTable + blockIdx.x * elemsPerThreadBlock;        // ����

	// ���ÿ���̶߳��������ϵ�Ԫ�ؽ�����������Ҫʹ�û��������� 
	SinglesStruct* mergeTile = mergeSortTile;
	SinglesStruct* bufferTile = mergeTile + elemsPerThreadBlock;

	// �����ݴ�ȫ���ڴ���빲������
	for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
	{
		mergeTile[tx] = globalDataTable[tx];
	}


	// �Թ��������е����ݽ�������
	for (unsigned stride = 1; stride < elemsPerThreadBlock; stride <<= 1)
	{
		__syncthreads();

		// �����������һ����ȡ��ȫ�����ݲ�д�ᣬ�Ͳ���Ҫд�뻺�������ˣ���ʱÿ���߳�ֻ�ܴ�����������
		for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock >> 1; tx += threadsMerge)       // �߳���������
		{
			unsigned offsetSample = tx & (stride - 1);
			unsigned offsetBlock = 2 * (tx - offsetSample);

			// ��ż�������������Ԫ��(�鱻�ϲ�)  
			SinglesStruct elemEven = mergeTile[offsetBlock + offsetSample];
			SinglesStruct elemOdd = mergeTile[offsetBlock + offsetSample + stride];            // �൱��һ���̴߳�������Ԫ��

			// ����ż�����е�Ԫ�����������е�λ�ã���֮��Ȼ  
			unsigned rankOdd = binarySearchInclusive(
				mergeTile, elemEven, offsetBlock + stride, offsetBlock + 2 * stride - 1
			);
			unsigned rankEven = binarySearchExclusive(
				mergeTile, elemOdd, offsetBlock, offsetBlock + stride - 1                 // ע�����߲��ҵ�λ�ò�ͬ
			);

			bufferTile[offsetSample + rankOdd - stride] = elemEven;
			bufferTile[offsetSample + rankEven] = elemOdd;
		}

		SinglesStruct* temp = mergeTile;        // �������ݽ���
		mergeTile = bufferTile;
		bufferTile = temp;
	}

	__syncthreads();
	// ���ϲ��õ����ݴӹ����ڴ����ȫ���ڴ�
	for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
	{
		globalDataTable[tx] = mergeTile[tx];
	}
}

template <unsigned subBlockSize>
__global__ void generateRanksKernel(SinglesStruct* data, unsigned* ranksEven, unsigned* ranksOdd, unsigned sortedBlockSize)
{
	unsigned subBlocksPerSortedBlock = sortedBlockSize / subBlockSize;                                      // ÿ���������黮�ֳɶ��ٸ��ӿ�
	unsigned subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;                                         // �����ϲ�������������ӿ�����

	// �ҵ�Ҫ����Ķ˵�ֵ
	SinglesStruct sampleValue = data[blockIdx.x * (blockDim.x * subBlockSize) + threadIdx.x * subBlockSize];     // ע��һ���̲߳���һ���˵�
	unsigned rankSampleCurrent = blockIdx.x * blockDim.x + threadIdx.x;                                     // �˵��
	unsigned rankSampleOpposite;

	// �ҵ��ö˵�ţ��ӿ飩���ڵ������Լ��������Ӧ���� 
	unsigned indexBlockCurrent = rankSampleCurrent / subBlocksPerSortedBlock;                               // �˵�Ŷ�Ӧ������
	unsigned indexBlockOpposite = indexBlockCurrent ^ 1;                                                    // ���ӦҪ���������

	// �ҵ�����˵�ֵ�����Ӧ���е��ĸ��ӿ�
	if (indexBlockCurrent % 2 == 0)
	{
		rankSampleOpposite = binarySearchInclusive<subBlockSize>(                              // �в����Ķ��ֲ���
			data, sampleValue, indexBlockOpposite * sortedBlockSize,                                      // ���Ҷ˵�����һ�������е�λ��
			indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
			);
		rankSampleOpposite = (rankSampleOpposite - sortedBlockSize) / subBlockSize;                       // ���λ��
	}
	else
	{
		rankSampleOpposite = binarySearchExclusive<subBlockSize>(
			data, sampleValue, indexBlockOpposite * sortedBlockSize,
			indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
			);
		rankSampleOpposite /= subBlockSize;
	}

	// ����ϲ����ڵ��������� 
	unsigned sortedIndex = rankSampleCurrent % subBlocksPerSortedBlock + rankSampleOpposite;         // ���ڿ���

	// ���������ڵ�ǰ�����������е�λ��
	unsigned rankDataCurrent = (rankSampleCurrent * subBlockSize % sortedBlockSize) + 1;             // �ڵ�ǰ�����е����λ��+1
	unsigned rankDataOpposite;

	// �������������ڵ��ӿ�������������С�˲��ҷ�Χ��
	unsigned indexSubBlockOpposite = sortedIndex % subBlocksPerMergedBlock - rankSampleCurrent % subBlocksPerSortedBlock - 1;
	unsigned indexStart = indexBlockOpposite * sortedBlockSize + indexSubBlockOpposite * subBlockSize + 1;     // ��ȡ�˵�ֵ
	unsigned indexEnd = indexStart + subBlockSize - 2;

	if ((int)(indexStart - indexBlockOpposite * sortedBlockSize) >= 0)
	{
		if (indexBlockOpposite % 2 == 0)   // ������������
		{
			rankDataOpposite = binarySearchExclusive(           // ���ֲ���λ��
				data, sampleValue, indexStart, indexEnd               // ���������ĺ���һ��λ�ã�������֤�����������д���������ͬ������ʱҲ�������㣩
			);
		}
		else                              // ������ż����
		{
			rankDataOpposite = binarySearchInclusive(
				data, sampleValue, indexStart, indexEnd              // ����������λ��
			);
		}

		rankDataOpposite -= indexBlockOpposite * sortedBlockSize;
	}
	else
	{
		rankDataOpposite = 0;
	}

	// ��Ӧ��ÿ���ӿ����λ��
	if ((rankSampleCurrent / subBlocksPerSortedBlock) % 2 == 0)    // ������ż����
	{
		ranksEven[sortedIndex] = rankDataCurrent;              // ��������1����֤������һ��Ԫ�ز���ϲ�
		ranksOdd[sortedIndex] = rankDataOpposite;
	}
	else
	{
		ranksEven[sortedIndex] = rankDataOpposite;
		ranksOdd[sortedIndex] = rankDataCurrent;               // ��������1
	}
}



//�ϲ���rank���������������ż���������ӿ顣
template <unsigned subBlockSize>
__global__ void mergeKernel(
	SinglesStruct* data, SinglesStruct* dataBuffer, unsigned* ranksEven, unsigned* ranksOdd, unsigned sortedBlockSize
)
{
	__shared__ SinglesStruct tileEven[subBlockSize];           // ÿ���ӿ��Ԫ�ظ�������512
	__shared__ SinglesStruct tileOdd[subBlockSize];            // �ӿ������ӿ��ż�ӿ��Ԫ�ظ�����������256

	unsigned indexRank = blockIdx.y * (sortedBlockSize / subBlockSize * 2) + blockIdx.x;               // �ҵ���ǰ�߳̿��Ӧ���ӿ�
	unsigned indexSortedBlock = blockIdx.y * 2 * sortedBlockSize;                  // ���߳̿�������ӿ����ڵ����еĲ�������ʼλ��

	// �������ڵ�ż���������飬���ǽ����ϲ�  
	unsigned indexStartEven, indexStartOdd, indexEndEven, indexEndOdd;
	unsigned offsetEven, offsetOdd;
	unsigned numElementsEven, numElementsOdd;

	// ��ȡż���������ӿ��START����  
	// ÿ���߳̿������ͬ����ʼ�ͽ���λ��
	if (blockIdx.x > 0)
	{
		indexStartEven = ranksEven[indexRank - 1];
		indexStartOdd = ranksOdd[indexRank - 1];
	}
	else
	{
		indexStartEven = 0;
		indexStartOdd = 0;
	}
	// ��ȡż���������ӿ��END����  
	if (blockIdx.x < gridDim.x - 1)
	{
		indexEndEven = ranksEven[indexRank];
		indexEndOdd = ranksOdd[indexRank];
	}
	else                           // ���ж�Ӧ�����һ���߳̿�
	{
		indexEndEven = sortedBlockSize;   // ���һ������
		indexEndOdd = sortedBlockSize;
	}

	numElementsEven = indexEndEven - indexStartEven;    // ������߳̿������Ԫ�ظ���
	numElementsOdd = indexEndOdd - indexStartOdd;

	// ��ż�������ӿ��ж�ȡ����
	if (threadIdx.x < numElementsEven)
	{
		offsetEven = indexSortedBlock + indexStartEven + threadIdx.x;
		tileEven[threadIdx.x] = data[offsetEven];
	}
	// �����������ӿ��ж�ȡ����
	if (threadIdx.x < numElementsOdd)
	{
		offsetOdd = indexSortedBlock + indexStartOdd + threadIdx.x;
		tileOdd[threadIdx.x] = data[offsetOdd + sortedBlockSize];
	}

	__syncthreads();

	if (threadIdx.x < numElementsEven)
	{
		unsigned rankOdd = binarySearchInclusive(tileOdd, tileEven[threadIdx.x], 0, numElementsOdd - 1);
		rankOdd += indexStartOdd;
		// �����ȫ�������е�λ��
		dataBuffer[offsetEven + rankOdd] = tileEven[threadIdx.x];
	}

	if (threadIdx.x < numElementsOdd)
	{
		unsigned rankEven = binarySearchExclusive(tileEven, tileOdd[threadIdx.x], 0, numElementsEven - 1);
		rankEven += indexStartEven;
		dataBuffer[offsetOdd + rankEven] = tileOdd[threadIdx.x];
	}
}


template <unsigned threadsCoinTime, unsigned elemsCoinTime>
__global__ void coinTimeKernel(SinglesStruct* data, unsigned* index, unsigned arrayLength, unsigned int timeWindow) {

	unsigned offset, dataBlockLength;
	calcDataBlockLength<threadsCoinTime, elemsCoinTime>(offset, dataBlockLength, arrayLength);     //	�ҵ�ÿ���߳̿����Ԫ�ص�ƫ�����ͳ���

	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += threadsCoinTime) {
		index[offset + tx] = 0;
		unsigned temp = offset + tx + 1;

		// �ҵ�һ����ʱ�䴰��������������Ψһһ�ԣ�����ж�ԵĻ�Ҫȫ���޳�
		while (temp < arrayLength && (data[temp].timevalue - data[offset + tx].timevalue) <= timeWindow) {         // �ҵ�ÿһ��Ԫ��������������Զλ��
			index[offset + tx] = temp;    // ��Ķ����±�
			++temp;
		}

	}

}

template <unsigned threadsCoinTime, unsigned elemsCoinTime>
__global__ void coinTimeKernel2(SinglesStruct* data, CoinStruct* mCoins, unsigned* index, unsigned* realIndex, unsigned* gIndex, unsigned* sIndex, unsigned arrayLength, unsigned int timeWindow, unsigned* realLength) {


	unsigned localIndex = 0;
	unsigned scanIndex = 0;

	unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
	//if (tid == 0) {
	//	*realLength = 0;
	//}

	unsigned offset = tid * elemsCoinTime;

	for (unsigned tx = 0; tx < elemsCoinTime && ((offset + tx) < arrayLength); ++tx) {
		// ���ݳ�ʼ��
		realIndex[offset + tx] = 0;

		// �ж��������޳����󲿷ֲ��������е�Ԫ���Լ�������ĩβ��Ԫ��
		if (index[offset + tx] != 0) {

			int tempLeft = offset + tx - 1;
			int tempRight;

			// �ҵ���ʼ����ʼλ��
			while (tempLeft > -1 && (data[tempLeft + 1].timevalue - data[tempLeft].timevalue) <= timeWindow) {
				--tempLeft;
			}
			tempRight = tempLeft + 1;

			// ������ʼλ�ã����µ���ָ��Խ������Ԫ�أ�ȷ��Ԫ�ز�����ұ�Ԫ��һ���޳���
			while (tempRight <= offset + tx) {
				if (index[tempRight] - tempRight != 1) {        // ֤��tempRight�±��Ӧ��Ԫ����Ҫ���޳������µõ����ڵ�tempLeft
					tempLeft = index[tempRight];
					tempRight = index[tempRight] + 1;
				}
				else {
					tempRight += 2;                             // ֤��tempRight�±��Ӧ��Ԫ�ز���Ҫ���޳���������������һ��
				}

			}
			// ����Ԫ��
			if (offset + tx > tempLeft && (offset + tx - tempLeft) % 2 == 1 && data[offset + tx].globalCrystalIndex != data[offset + tx + 1].globalCrystalIndex) {
				realIndex[offset + tx] = offset + tx + 1;
			}
		}
		localIndex += realIndex[offset + tx] == 0 ? 0 : 1;
	}

	// �õ�����߳̿��ڣ�ÿ���߳�֮ǰ���������̣߳��������߳���ΪTRUE��Ԫ�ظ���
	scanIndex = intraBlockScan<threadsCoinTime>(localIndex);
	__syncthreads();

	if (threadIdx.x == (threadsCoinTime - 1))										        // ÿ���߳̿��е����һ���̣߳����߳���scanLower��ŵ��Ǳ��߳̿���С��pivots��Ԫ������
	{
		atomicAdd(realLength, scanIndex);                                                   // ���ڼ����޳����������
		gIndex[blockIdx.x] = scanIndex;                                                     // gIndex�Ĵ�СΪ�߳̿����Ŀ����¼�˸����߳̿麬�з���������Ԫ�ظ���
	}

	sIndex[tid] = scanIndex - localIndex;                                                   // һ���̶߳�Ӧһ��ƫ����

}

template <unsigned threadsCoinTime, unsigned elemsCoinTime>
__global__ void coinTimeKernel3(SinglesStruct* data, CoinStruct* mCoins, unsigned* realIndex, unsigned* gIndex, unsigned* sIndex, unsigned arrayLength) {

	unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned offset = tid * elemsCoinTime;

	unsigned globalIndex = 0;

	for (unsigned i = 0; i < blockIdx.x; ++i) {
		globalIndex += gIndex[i];
	}

	unsigned indexTrue = globalIndex + sIndex[tid];

	for (unsigned tx = 0; tx < elemsCoinTime && ((offset + tx) < arrayLength); ++tx) {
		if (realIndex[offset + tx] != 0) {
			CoinStruct mergeData;
			mergeData.nCoinStruct[0] = data[offset + tx];
			mergeData.nCoinStruct[1] = data[offset + tx + 1];
			mCoins[indexTrue++] = mergeData;
		}
	}

}
