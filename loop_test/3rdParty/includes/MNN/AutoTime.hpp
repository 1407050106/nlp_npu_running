//
//  AutoTime.hpp
//  MNN
//
//  Created by MNN on 2018/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef AutoTime_hpp
#define AutoTime_hpp

#include <stdint.h>
#include <stdio.h>
#include <MNN/MNNDefine.h>

namespace MNN {

class MNN_PUBLIC Timer {
public:
    Timer();
    ~Timer();
    Timer(const Timer&)  = delete;
    Timer(const Timer&&) = delete;
    Timer& operator=(const Timer&)  = delete;
    Timer& operator=(const Timer&&) = delete;
    
    // reset timer
    void reset();
    // get duration (us) from init or latest reset.
    uint64_t durationInUs();
    
protected:
#ifdef __EMSCRIPTEN__
    double mLastResetTime;
#else
    uint64_t mLastResetTime;
#endif
};

/** time tracing util. prints duration between init and deinit. */
class MNN_PUBLIC AutoTime : Timer {
public:
    AutoTime(int line, const char* func);
    ~AutoTime();
    AutoTime(const AutoTime&)  = delete;
    AutoTime(const AutoTime&&) = delete;
    AutoTime& operator=(const AutoTime&) = delete;
    AutoTime& operator=(const AutoTime&&) = delete;

private:
    int mLine;
    char* mName;
};
} // namespace MNN

#ifdef MNN_OPEN_TIME_TRACE
#define AUTOTIME MNN::AutoTime ___t(__LINE__, __func__)
#else
#define AUTOTIME
#endif

#endif /* AutoTime_hpp */
