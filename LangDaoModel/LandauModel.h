#pragma once
class LandauModel {
private:
	void runResolveKernel(Package* d_lUdpdata, SinglesStruct* d_lSingleData, SinglesStruct* d_lSingleDataBuffer, unsigned lUdpNum, EnegryCorrectData* d_eCorData, TimeCorrectData* d_tCorData, unsigned* d_len) {
		cudaError_t error;
		const int threadsNum = THREADS_RESOLVE;
		const int threadsElem = ELEMS_RESLOVE;
		const int elemsPerBlock = threadsNum * threadsElem;
		long long eventNum = lUdpNum * EVENT_NUM_PACKAGE;

		dim3 dimGrid((eventNum - 1) / elemsPerBlock + 1, 1, 1);
		dim3 dimBlock(threadsNum, 1, 1);
		ResolveKernel << <dimGrid, dimBlock >> > (d_lUdpdata, d_lSingleData, d_lSingleDataBuffer, eventNum, d_eCorData, d_tCorData, d_len);
	}
public:
	void start() {
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
};