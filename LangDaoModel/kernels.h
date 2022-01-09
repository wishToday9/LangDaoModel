#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "contants.h"
#include "para.h"
#include "device_functions.h"
#include "kernels.h"

const int num_params = 4;  //参数个数
const int total_data = 8;
const int m = 8; //数据个数

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
	unsigned elemsPerThreadBlock = numThreads * elemsThread;            // 计算每个线程块要处理的数据量
	offset = blockIdx.x * elemsPerThreadBlock;
	dataBlockLength = offset + elemsPerThreadBlock <= arrayLength ? elemsPerThreadBlock : arrayLength - offset;       // 对最后一个线程块的特殊处理
}

template <unsigned blockSize>
inline __device__ unsigned intraWarpScan(volatile unsigned* scanTile, unsigned val) {
	unsigned index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));

	scanTile[index] = 0;              // 将前面一列置零
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

	// 多个元素的值进行合并
	return scanTile[index] - val;

}


template <unsigned blockSize>
inline __device__ unsigned intraBlockScan(unsigned val) {
	__shared__ unsigned scanTile[blockSize * 2];
	unsigned warpIdx = threadIdx.x / WARP_SIZE;
	unsigned laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp


	unsigned warpResult = intraWarpScan<blockSize>(scanTile, val);
	__syncthreads();


	if (laneIdx == WARP_SIZE - 1)                 // 得到32个值的总和放在对应的warpIdx中
	{
		scanTile[warpIdx] = warpResult + val;
	}
	__syncthreads();


	if (threadIdx.x < WARP_SIZE)                  // 仅用其中一个warp进行操作
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
		//计算事件的时间点
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

			//朗道模型
			for (int i = 0; i < 8; i++)
			{
				//增加动态阈值判断
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
		global = atomicAdd(d_len, scan);                                  // 原子操作，必须一个个进行
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
		// 小于该值的能被strike整除的最大数
		int index = ((indexStart + indexEnd) / 2) & ((stride - 1) ^ UINT_MAX);   // 找到偏左的元素   

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
	calcDataBlockLength<THREADS_PADDING, ELEMS_PADDING>(offset, dataBlockLength, paddingLength);       // 计算每个block的偏移量和长度（这个函数很有用）
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
	SinglesStruct* globalDataTable = dataTable + blockIdx.x * elemsPerThreadBlock;        // 划分

	// 如果每个线程对两个以上的元素进行排序，则需要使用缓冲区数组 
	SinglesStruct* mergeTile = mergeSortTile;
	SinglesStruct* bufferTile = mergeTile + elemsPerThreadBlock;

	// 将数据从全局内存存入共享缓存区
	for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
	{
		mergeTile[tx] = globalDataTable[tx];
	}


	// 对共享缓存区中的数据进行排序
	for (unsigned stride = 1; stride < elemsPerThreadBlock; stride <<= 1)
	{
		__syncthreads();

		// 如果可以做到一次性取出全部数据并写会，就不需要写入缓冲区中了，此时每个线程只能处理两个数据
		for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock >> 1; tx += threadsMerge)       // 线程是连续的
		{
			unsigned offsetSample = tx & (stride - 1);
			unsigned offsetBlock = 2 * (tx - offsetSample);

			// 从偶数和奇数块加载元素(块被合并)  
			SinglesStruct elemEven = mergeTile[offsetBlock + offsetSample];
			SinglesStruct elemOdd = mergeTile[offsetBlock + offsetSample + stride];            // 相当于一个线程处理两个元素

			// 计算偶数块中的元素在奇数块中的位置，反之亦然  
			unsigned rankOdd = binarySearchInclusive(
				mergeTile, elemEven, offsetBlock + stride, offsetBlock + 2 * stride - 1
			);
			unsigned rankEven = binarySearchExclusive(
				mergeTile, elemOdd, offsetBlock, offsetBlock + stride - 1                 // 注意两者查找的位置不同
			);

			bufferTile[offsetSample + rankOdd - stride] = elemEven;
			bufferTile[offsetSample + rankEven] = elemOdd;
		}

		SinglesStruct* temp = mergeTile;        // 数组内容交换
		mergeTile = bufferTile;
		bufferTile = temp;
	}

	__syncthreads();
	// 将合并好的数据从共享内存存入全局内存
	for (unsigned tx = threadIdx.x; tx < elemsPerThreadBlock; tx += threadsMerge)
	{
		globalDataTable[tx] = mergeTile[tx];
	}
}

template <unsigned subBlockSize>
__global__ void generateRanksKernel(SinglesStruct* data, unsigned* ranksEven, unsigned* ranksOdd, unsigned sortedBlockSize)
{
	unsigned subBlocksPerSortedBlock = sortedBlockSize / subBlockSize;                                      // 每个有序数组划分成多少个子块
	unsigned subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;                                         // 两个合并的排序数组的子块总数

	// 找到要处理的端点值
	SinglesStruct sampleValue = data[blockIdx.x * (blockDim.x * subBlockSize) + threadIdx.x * subBlockSize];     // 注意一个线程查找一个端点
	unsigned rankSampleCurrent = blockIdx.x * blockDim.x + threadIdx.x;                                     // 端点号
	unsigned rankSampleOpposite;

	// 找到该端点号（子块）所在的序列以及它的相对应序列 
	unsigned indexBlockCurrent = rankSampleCurrent / subBlocksPerSortedBlock;                               // 端点号对应的序列
	unsigned indexBlockOpposite = indexBlockCurrent ^ 1;                                                    // 相对应要处理的序列

	// 找到这个端点值在相对应序列的哪个子块
	if (indexBlockCurrent % 2 == 0)
	{
		rankSampleOpposite = binarySearchInclusive<subBlockSize>(                              // 有步幅的二分查找
			data, sampleValue, indexBlockOpposite * sortedBlockSize,                                      // 查找端点在另一个序列中的位置
			indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
			);
		rankSampleOpposite = (rankSampleOpposite - sortedBlockSize) / subBlockSize;                       // 相对位置
	}
	else
	{
		rankSampleOpposite = binarySearchExclusive<subBlockSize>(
			data, sampleValue, indexBlockOpposite * sortedBlockSize,
			indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
			);
		rankSampleOpposite /= subBlockSize;
	}

	// 计算合并块内的样本索引 
	unsigned sortedIndex = rankSampleCurrent % subBlocksPerSortedBlock + rankSampleOpposite;         // 所在块编号

	// 计算样本在当前和相对排序块中的位置
	unsigned rankDataCurrent = (rankSampleCurrent * subBlockSize % sortedBlockSize) + 1;             // 在当前序列中的相对位置+1
	unsigned rankDataOpposite;

	// 计算相对排序块内的子块索引（这里缩小了查找范围）
	unsigned indexSubBlockOpposite = sortedIndex % subBlocksPerMergedBlock - rankSampleCurrent % subBlocksPerSortedBlock - 1;
	unsigned indexStart = indexBlockOpposite * sortedBlockSize + indexSubBlockOpposite * subBlockSize + 1;     // 不取端点值
	unsigned indexEnd = indexStart + subBlockSize - 2;

	if ((int)(indexStart - indexBlockOpposite * sortedBlockSize) >= 0)
	{
		if (indexBlockOpposite % 2 == 0)   // 本身是奇序列
		{
			rankDataOpposite = binarySearchExclusive(           // 二分查找位置
				data, sampleValue, indexStart, indexEnd               // 满足条件的后面一个位置（这样保证了两个序列中存在两个相同的数字时也可以满足）
			);
		}
		else                              // 本身是偶序列
		{
			rankDataOpposite = binarySearchInclusive(
				data, sampleValue, indexStart, indexEnd              // 满足条件的位置
			);
		}

		rankDataOpposite -= indexBlockOpposite * sortedBlockSize;
	}
	else
	{
		rankDataOpposite = 0;
	}

	// 对应的每个子块结束位置
	if ((rankSampleCurrent / subBlocksPerSortedBlock) % 2 == 0)    // 本身是偶序列
	{
		ranksEven[sortedIndex] = rankDataCurrent;              // 这里多加了1，保证至少有一个元素参与合并
		ranksOdd[sortedIndex] = rankDataOpposite;
	}
	else
	{
		ranksEven[sortedIndex] = rankDataOpposite;
		ranksOdd[sortedIndex] = rankDataCurrent;               // 这里多加了1
	}
}



//合并由rank数组决定的连续的偶数和奇数子块。
template <unsigned subBlockSize>
__global__ void mergeKernel(
	SinglesStruct* data, SinglesStruct* dataBuffer, unsigned* ranksEven, unsigned* ranksOdd, unsigned sortedBlockSize
)
{
	__shared__ SinglesStruct tileEven[subBlockSize];           // 每个子块的元素个数不超512
	__shared__ SinglesStruct tileOdd[subBlockSize];            // 子块中奇子块和偶子块的元素个数均不超过256

	unsigned indexRank = blockIdx.y * (sortedBlockSize / subBlockSize * 2) + blockIdx.x;               // 找到当前线程块对应的子块
	unsigned indexSortedBlock = blockIdx.y * 2 * sortedBlockSize;                  // 该线程块操作的子块所在的序列的操作的起始位置

	// 索引相邻的偶数和奇数块，它们将被合并  
	unsigned indexStartEven, indexStartOdd, indexEndEven, indexEndOdd;
	unsigned offsetEven, offsetOdd;
	unsigned numElementsEven, numElementsOdd;

	// 读取偶数和奇数子块的START索引  
	// 每个线程块操作相同的起始和结束位置
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
	// 读取偶数和奇数子块的END索引  
	if (blockIdx.x < gridDim.x - 1)
	{
		indexEndEven = ranksEven[indexRank];
		indexEndOdd = ranksOdd[indexRank];
	}
	else                           // 序列对应的最后一个线程块
	{
		indexEndEven = sortedBlockSize;   // 最后一个序列
		indexEndOdd = sortedBlockSize;
	}

	numElementsEven = indexEndEven - indexStartEven;    // 求出该线程块操作的元素个数
	numElementsOdd = indexEndOdd - indexStartOdd;

	// 从偶数有序子块中读取数据
	if (threadIdx.x < numElementsEven)
	{
		offsetEven = indexSortedBlock + indexStartEven + threadIdx.x;
		tileEven[threadIdx.x] = data[offsetEven];
	}
	// 从奇数有序子块中读取数据
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
		// 求出在全局数组中的位置
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
	calcDataBlockLength<threadsCoinTime, elemsCoinTime>(offset, dataBlockLength, arrayLength);     //	找到每个线程块操作元素的偏移量和长度

	for (unsigned tx = threadIdx.x; tx < dataBlockLength; tx += threadsCoinTime) {
		index[offset + tx] = 0;
		unsigned temp = offset + tx + 1;

		// 找到一个是时间窗口内满足条件的唯一一对，如果有多对的话要全部剔除
		while (temp < arrayLength && (data[temp].timevalue - data[offset + tx].timevalue) <= timeWindow) {         // 找到每一个元素满足条件的最远位置
			index[offset + tx] = temp;    // 存的都是下标
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
		// 数据初始化
		realIndex[offset + tx] = 0;

		// 判断条件，剔除掉大部分不在序列中的元素以及在序列末尾的元素
		if (index[offset + tx] != 0) {

			int tempLeft = offset + tx - 1;
			int tempRight;

			// 找到初始的起始位置
			while (tempLeft > -1 && (data[tempLeft + 1].timevalue - data[tempLeft].timevalue) <= timeWindow) {
				--tempLeft;
			}
			tempRight = tempLeft + 1;

			// 更新起始位置，更新到右指针越过操作元素（确保元素不会和右边元素一起剔除）
			while (tempRight <= offset + tx) {
				if (index[tempRight] - tempRight != 1) {        // 证明tempRight下标对应的元素需要被剔除，更新得到现在的tempLeft
					tempLeft = index[tempRight];
					tempRight = index[tempRight] + 1;
				}
				else {
					tempRight += 2;                             // 证明tempRight下标对应的元素不需要被剔除，则两两绑定找下一对
				}

			}
			// 更新元素
			if (offset + tx > tempLeft && (offset + tx - tempLeft) % 2 == 1 && data[offset + tx].globalCrystalIndex != data[offset + tx + 1].globalCrystalIndex) {
				realIndex[offset + tx] = offset + tx + 1;
			}
		}
		localIndex += realIndex[offset + tx] == 0 ? 0 : 1;
	}

	// 得到这个线程块内，每个线程之前（包括该线程）的所有线程中为TRUE的元素个数
	scanIndex = intraBlockScan<threadsCoinTime>(localIndex);
	__syncthreads();

	if (threadIdx.x == (threadsCoinTime - 1))										        // 每个线程块中的最后一个线程，此线程中scanLower存放的是本线程块中小于pivots的元素总数
	{
		atomicAdd(realLength, scanIndex);                                                   // 用于计算剔除后的数据量
		gIndex[blockIdx.x] = scanIndex;                                                     // gIndex的大小为线程块的数目，记录了各个线程块含有符合条件的元素个数
	}

	sIndex[tid] = scanIndex - localIndex;                                                   // 一个线程对应一个偏移量

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
