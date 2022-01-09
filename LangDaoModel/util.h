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

// ����ϲ������ģ�ĺ���
unsigned calculateMergeArraySize(unsigned arrayLength, unsigned sortedBlockSize)
{
	unsigned arrayLenMerge = previousPowerOf2(arrayLength);
	unsigned mergedBlockSize = 2 * sortedBlockSize;

	// ���鳤�ȱ����Ƿ���2���ݴΣ��������ֱ�ӷ���
	if (arrayLenMerge != arrayLength)
	{
		// ����ʣ�ಿ�ֳ���
		unsigned remainder = arrayLength - arrayLenMerge;

		// ���ʣ�ಿ�ֳ��ȴ������򳤶ȣ���ʣ�ಿ����Ҫ����ϲ������ʣ�ಿ�ֳ���С�����򳤶ȣ�����ϲ����ȴ���Ҫ���ֺϲ�����Ժ��ٺϲ�ʣ�ಿ��
		// �ж�ʣ�ಿ�ֳ����Ƿ�������򳤶�
		if (remainder >= sortedBlockSize)
		{
			arrayLenMerge += roundUp(remainder, 2 * sortedBlockSize);    // ʣ�ಿ�ֿ���ִ�кϲ�
		}
		// �ж���Ҫ�����Ƿ�ϲ����
		else if (arrayLenMerge == sortedBlockSize)
		{
			arrayLenMerge += sortedBlockSize;
		}
	}

	// �������յĺϲ���ģ
	return arrayLenMerge;
}