//
//  Authors: wangyonglin03
//
//  Created by WYL on 2024/08/28.
//  

#include <MNN/AutoTime.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <atomic>

#include "tool.hpp"
#include "ThreadPool.hpp"
using namespace std;

#include <errno.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

#define BUFFER_SIZE 1024
#define ENABLE_CPU_AFFINITY
#undef ENABLE_CPU_AFFINITY

#ifdef ENABLE_CPU_AFFINITY
static uint32_t getNumberOfCPU() {
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        printf("[HM_Log]: Can't open /proc/cpuinfo.\n");
        return 1;
    }
    uint32_t number = 0;
    char buffer[BUFFER_SIZE];
    while (!feof(fp)) {
        char* str = fgets(buffer, BUFFER_SIZE, fp);
        if (!str) {
            break;
        }
        if (memcmp(buffer, "processor", 9) == 0) {
            number++;
        }
    }
    fclose(fp);
    if (number < 1) {
        number = 1;
    }
    return number;
}

static int getCPUMaxFreqKHz(int cpuID) {
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuID);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuID);
        fp = fopen(path, "rb");
        if (!fp) {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuID);
            fp = fopen(path, "rb");
            if (!fp) {
                printf("[HM_Log]: Can't get cpufreq by number.\n");
                return -1;
            }
            int maxfrequency = -1;
            fscanf(fp, "%d", &maxfrequency);
            fclose(fp);
            return maxfrequency;
        }
    }
    int maxfrequency = 0;
    while (!feof(fp)) {
        int frequency = 0;
        int history   = fscanf(fp, "%d %*d", &frequency);
        if (history != 1) {
            break;
        }
        if (frequency > maxfrequency) {
            maxfrequency = frequency;
        }
    }
    fclose(fp);
    return maxfrequency;
}

static int sortCPUIDByMaxFrequency(std::vector<int>& cpuIDs, int* littleClusterOffset) {
    const int cpuNumbers = cpuIDs.size();
    *littleClusterOffset = 0;
    if (cpuNumbers == 0) {
        printf("[HM_Log]: Cpu numbers zero.\n");
        return 0;
    }
    std::vector<int> cpusFrequency;
    cpusFrequency.resize(cpuNumbers);
    for (int i = 0; i < cpuNumbers; ++i) {
        int frequency    = getCPUMaxFreqKHz(i);
        cpuIDs[i]        = i;
        cpusFrequency[i] = frequency;
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        for (int j = i + 1; j < cpuNumbers; ++j) {
            if (cpusFrequency[i] < cpusFrequency[j]) {
                // id
                int temp  = cpuIDs[i];
                cpuIDs[i] = cpuIDs[j];
                cpuIDs[j] = temp;
                // frequency
                temp             = cpusFrequency[i];
                cpusFrequency[i] = cpusFrequency[j];
                cpusFrequency[j] = temp;
            }
        }
    }
    int midMaxFrequency = (cpusFrequency.front() + cpusFrequency.back()) / 2;
    if (midMaxFrequency == cpusFrequency.back()) {
        printf("[HM_Log]: MidMaxFrequency == cpusFrequency.back()\n");
        return 0;
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        if (cpusFrequency[i] < midMaxFrequency) {
            *littleClusterOffset = i;
            break;
        }
    }
    return 0;
}


//#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))

void set_cpu_affinity()
{
    int cpu_core_num = sysconf(_SC_NPROCESSORS_CONF);
    //LOG_MCNN_CL_INF("cpu core num = %d\n", cpu_core_num);
    int cpu_id = 0;
    cpu_set_t mask;
    CPU_ZERO(&mask);

    auto numberOfCPUs = getNumberOfCPU();
    printf("[HM_Log]: Get_cpu_number = %d\n", (int)numberOfCPUs);
    static std::vector<int> sortedCPUIDs;
    static int littleClusterOffset = 0;
    if (sortedCPUIDs.empty()) {
        sortedCPUIDs.resize(numberOfCPUs);
        for (int i = 0; i < numberOfCPUs; ++i) {
            sortedCPUIDs[i] = i;
        }
        sortCPUIDByMaxFrequency(sortedCPUIDs, &littleClusterOffset);
    }

    printf("[HM_Log]: Cpu max core:\n");
    for (cpu_id = 0; cpu_id < littleClusterOffset; cpu_id++)
    {
        printf("[HM_Log]: %d\n", sortedCPUIDs[cpu_id]);
        CPU_SET(sortedCPUIDs[cpu_id], &mask);
    }
    // CPU_SET(1, &mask);
    printf("\n");


    int sys_call_res = syscall(__NR_sched_setaffinity, gettid(), sizeof(mask), &mask);
    //LOG_MCNN_CL_INF("sys call res = %d\n", sys_call_res);
    if (sys_call_res)
    {
        printf("[HM_Log]: set_cpu_affinity errno = %d\n", (int)errno);
    }
    printf("[HM_Log]: Set_cpu_affinity success.\n");
}
#endif

// Usage: ./lsTM_LOOP model.mnn threads_num precision_num forward_type shapexshape
int main(int argc, const char* argv[]) {
    if (argc < 6) {
        MNN_PRINT("Usage: ./lsTM_LOOP model.mnn threads_num precision_num forward_type shapexshape\n");
        return 0;
    }
    // 获取输入参数：模型路径、线程数、运行时精度、推理后端类型、输入形状
    std::shared_ptr<Interpreter> net;
    net.reset(Interpreter::createFromFile(argv[1]));
    // net.reset(Interpreter::createFromBuffer(bankcard_detect_fp16_mnn, bankcard_detect_fp16_mnn_len));
    if (net == nullptr) {
        MNN_ERROR("Invalid Model\n");
        return 0;
    }
    std::string pwd = "./";
    int threads_num = std::atoi(argv[2]);
    int precision_num = std::atoi(argv[3]);
    // 设置运行时配置信息
    ScheduleConfig config;
    config.numThread = threads_num;
    int forwardTypeInt = std::atoi(argv[4]);
    config.type = static_cast<MNNForwardType>(forwardTypeInt);

    MNN::BackendConfig backendConfig;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision_num);
    config.backendConfig = &backendConfig;
    MNN_PRINT("ForwardType:%d, Thread:%d, Precision:%d\n", config.type, config.numThread, config.backendConfig->precision);

    std::vector<int> inputDims;
    if (argc > 5) {
        std::string inputShape(argv[5]);
        const char* delim = "x";
        std::ptrdiff_t p1 = 0, p2;
        while (1) {
            p2 = inputShape.find(delim, p1);
            if (p2 != std::string::npos) {
                inputDims.push_back(atoi(inputShape.substr(p1, p2 - p1).c_str()));
                p1 = p2 + 1;
            } else {
                inputDims.push_back(atoi(inputShape.substr(p1).c_str()));
                break;
            }
        }
    }
    // 从[500, 40]序列文件中获取float数据，保存在inputData中
    std::ostringstream fileName;
    fileName << pwd << "input_0" << ".txt";
    std::ifstream input_file(fileName.str().c_str());
    std::vector<float> inputData;
    inputData.reserve(512);
    if (input_file.is_open()) {
        float value;
        while (input_file >> value) {
            inputData.emplace_back(value);
        }
        input_file.close();
    } else {
        MNN_PRINT("Failed to open input file\n");
        return 0;
    }
    for (int k=0; k<12; k++) {
        inputData.emplace_back(0.0);
    }
    
    unordered_map<int, Session*> SessionMap;
    int s_index=0;
    for (int i = 0; i < 4; i++) {
        auto session = net->createSession(config);
        auto inputTensor = net->getSessionInput(session, nullptr);
        if (!inputDims.empty()) {
            net->resizeTensor(inputTensor, inputDims);
            net->resizeSession(session);
            MNN_PRINT("===========>%d Resize Done...\n", i);
        }
        SessionMap[i] = session;

        auto dimType = inputTensor->getDimensionType();
        MNN::Tensor givenTensor(inputTensor, dimType);
        if (givenTensor.getType().code == halide_type_float) {
            auto inputdata = givenTensor.host<float>();
            auto size      = givenTensor.elementSize();
            // MNN_PRINT("%d givenTensor inputdata size:%d\n", i, size);
            for (int i = 0; i < 128; ++i) {
                inputdata[i] = inputData[s_index+i];
            }
            s_index += 128;
            // MNN_PRINT("%d s_index:%d\n", i, s_index);
        }
        inputTensor->copyFromHostTensor(&givenTensor);
    }
#ifdef ENABLE_CPU_AFFINITY
    set_cpu_affinity();
#endif

    threadpool executor(4);
    unordered_map<int, vector<float>> process_answers;
    // Linear* linear = new Linear(128, 64);
    std::vector<std::future<bool>> results;
    double total_time = 0.0;
    std::chrono::steady_clock::time_point time_s, time_e;

    while (1) {
    time_s = std::chrono::steady_clock::now();
    for (int j=0; j<4; j++) {
        auto session = SessionMap[j];
        results.emplace_back(
            executor.commit([session, net, &process_answers, j](){
                net->runSession(session);
                auto outputTensor = net->getSessionOutput(session, NULL);
                auto out_shape = outputTensor->shape();
                // printf("Output_tensor shape: ");
                // for (int i=0; i<out_shape.size(); i++)
                // {
                //     std::cout<<out_shape[i]<<" ";
                // }
                // printf("\n");

                MNN::Tensor expectTensor(outputTensor, outputTensor->getDimensionType());
                outputTensor->copyToHostTensor(&expectTensor);

                int dimension = expectTensor.buffer().dimensions;
                int width     = 1;
                if (dimension > 1) {
                    width = expectTensor.length(dimension - 1);
                }
                const int outside = expectTensor.elementSize() / width;
                auto data = expectTensor.host<float>();    
                Linear linear(out_shape[2]*out_shape[1], (out_shape[2]*out_shape[1])/2);   
                vector<float> process_res;         
                for (int z = 0; z < outside; ++z) {              
                    for (int x = 0; x < width; ++x) {            
                        process_res.emplace_back(data[x + z * width]);
                    }                                                                      
                }
                vector<float> final_ans = linear.forward(process_res);
                // 每一个SessionMap[j]处理得到的结果通过j索引做定位，存进map 
                process_answers.emplace(j, final_ans);
                return true;
            })
        );
    }
    for (auto&& result : results)  // 等待所有线程执行完毕~
			std::cout << result.get() << ' ';
    results.clear();
    results.shrink_to_fit();

    time_e = std::chrono::steady_clock::now();
    printf("******* Run model time: %.3f ms\n",
        std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s).count() / 1000.0);

    auto outputFile = pwd + "output.txt";
    if (1) {
        std::ofstream outFile(outputFile);
        if (outFile.is_open()) {
            for (int id=0; id<4; id++) {
                auto final_ans = process_answers[id];
                // 按照最初从0-4的顺序将处理结果拼接起来，保证顺序~
                for (auto value : final_ans) {
                    outFile << value << "\n";
                }
            }  
            outFile.close();
            MNN_PRINT("Output file saved to %s\n", outputFile.c_str());
        } else {
            MNN_PRINT("Failed to open output file\n");
        }
    }   
    }

    return 0;
}