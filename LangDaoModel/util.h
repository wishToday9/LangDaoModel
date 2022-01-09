#pragma once
unsigned previousPowerOf2(unsigned value) {
	if ((value != 0) && ((value & (value - 1)) == 0))
	{
		return value;
	}

	value--;
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	value -= value >> 1;

	return value;
}

unsigned nextPowerOf2(unsigned value)
{
	if ((value != 0) && ((value & (value - 1)) == 0))
	{
		return value;
	}

	value--;
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	value++;

	return value;
}
int roundUp(int numToRound, int multiple)
{
	if (multiple == 0)
	{
		return numToRound;
	}

	int remainder = numToRound % multiple;

	if (remainder == 0)
	{
		return numToRound;
	}

	return numToRound + multiple - remainder;
}

// 计算合并数组规模的函数
unsigned calculateMergeArraySize(unsigned arrayLength, unsigned sortedBlockSize)
{
	unsigned arrayLenMerge = previousPowerOf2(arrayLength);
	unsigned mergedBlockSize = 2 * sortedBlockSize;

	// 数组长度本身是否是2的幂次，如果是则直接返回
	if (arrayLenMerge != arrayLength)
	{
		// 计算剩余部分长度
		unsigned remainder = arrayLength - arrayLenMerge;

		// 如果剩余部分长度大于有序长度，则剩余部分需要参与合并，如果剩余部分长度小于有序长度，则不需合并，等待主要部分合并完成以后再合并剩余部分
		// 判断剩余部分长度是否大于有序长度
		if (remainder >= sortedBlockSize)
		{
			arrayLenMerge += roundUp(remainder, 2 * sortedBlockSize);    // 剩余部分可以执行合并
		}
		// 判断主要部分是否合并完成
		else if (arrayLenMerge == sortedBlockSize)
		{
			arrayLenMerge += sortedBlockSize;
		}
	}

	// 返回最终的合并规模
	return arrayLenMerge;
}