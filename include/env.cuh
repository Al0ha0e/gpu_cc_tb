#ifndef ENV_H
#define ENV_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#define NVRTC_GET_TYPE_NAME 1

#include <nvrtc.h>
#include <cuda.h>

namespace common
{
#define NVRTC_SAFE_CALL(x)                                    \
    do                                                        \
    {                                                         \
        nvrtcResult result = x;                               \
        if (result != NVRTC_SUCCESS)                          \
        {                                                     \
            std::cerr << "\nerror: " #x " failed with error " \
                      << nvrtcGetErrorString(result) << '\n'; \
            exit(1);                                          \
        }                                                     \
    } while (0)

#define CUDA_DRIVER_SAFE_CALL(x)                                                             \
    do                                                                                       \
    {                                                                                        \
        CUresult result = x;                                                                 \
        if (result != CUDA_SUCCESS)                                                          \
        {                                                                                    \
            const char *msg;                                                                 \
            cuGetErrorName(result, &msg);                                                    \
            std::cerr << __FILE__ << " " << __LINE__ << "\nerror: " #x " failed with error " \
                      << msg << '\n';                                                        \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

#define CUDA_SAFE_CALL(x)                                                                                     \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t result = x;                                                                               \
        if (result != cudaSuccess)                                                                            \
        {                                                                                                     \
            const char *name = cudaGetErrorName(result);                                                      \
            const char *msg = cudaGetErrorString(result);                                                     \
            std::cerr << __FILE__ << " " << __LINE__ << "\nerror: " #x " failed with error " << name << '\n'; \
            std::cerr << "msg: " << msg << '\n';                                                              \
            exit(1);                                                                                          \
        }                                                                                                     \
    } while (0)

    class Context
    {
    public:
        Context()
        {
            CUDA_DRIVER_SAFE_CALL(cuInit(0));
            CUDA_DRIVER_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
            CUDA_DRIVER_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
        }

        CUdevice cuDevice;
        CUcontext context;
    };

    class RTCProgram
    {
    public:
        RTCProgram() {}
        RTCProgram(std::string src, std::string inc_pth, std::string prog_name, bool load)
            : include_path(inc_pth), prog_name(prog_name)
        {
            if (load)
            {
                std::ifstream ifs(src);
                std::stringstream buffer;
                buffer << ifs.rdbuf();
                source = std::string(buffer.str());
                ifs.close();
            }
            else
            {
                source = src;
            }

            NVRTC_SAFE_CALL(
                nvrtcCreateProgram(&program,          // prog
                                   source.c_str(),    // buffer
                                   prog_name.c_str(), // name
                                   0,                 // numHeaders
                                   NULL,              // headers
                                   NULL));            // includeNames
        }

        ~RTCProgram()
        {
            NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));
            cuModuleUnload(module);
        }

        void AddNameExpression(std::string expr)
        {
            names.push_back(expr);
            NVRTC_SAFE_CALL(nvrtcAddNameExpression(program, expr.c_str()));
        }

        bool Compile(std::vector<const char *> opts)
        {
            const char *default_option = "-DNVRTC_COMPILE";
            opts.push_back(default_option);

            nvrtcResult compileResult = nvrtcCompileProgram(program,      // prog
                                                            opts.size(),  // numOptions
                                                            opts.data()); // options

            // Obtain compilation log from the program.
            size_t logSize;
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(program, &logSize));
            if (logSize > 1)
            {
                char *log = new char[logSize + 1];
                NVRTC_SAFE_CALL(nvrtcGetProgramLog(program, log));
                std::cerr << "LOG: " << log << '\n';
                log[logSize] = '\0';
                delete[] log;
            }

            if (compileResult != NVRTC_SUCCESS)
                return false;

            for (auto &name : names)
            {
                const char *compiled_name;
                NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                    program,
                    name.c_str(),  // name expression
                    &compiled_name // lowered name
                    ));

                compiled_names[name] = compiled_name;
            }

            // Obtain PTX from the program.
            size_t ptxSize;
            NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &ptxSize));
            char *ptx_content = new char[ptxSize];
            NVRTC_SAFE_CALL(nvrtcGetPTX(program, ptx_content));
            CUDA_DRIVER_SAFE_CALL(cuModuleLoadDataEx(&module, ptx_content, 0, 0, 0));

            std::string pth = std::string("./out/") + prog_name + ".ptx";
            std::ofstream fo(pth, std::ios::out | std::ios::binary);
            fo.write(ptx_content, ptxSize);
            fo.close();

            return true;
        }

        void Call(std::string func, dim3 num_blocks, dim3 num_threads, void *args[], CUstream stream)
        {
            // TODO shared mem, stream
            CUfunction kernel;
            const char *compiled_name = compiled_names[func];
            CUDA_DRIVER_SAFE_CALL(cuModuleGetFunction(&kernel, module, compiled_name));

            CUDA_DRIVER_SAFE_CALL(
                cuLaunchKernel(kernel,
                               num_blocks.x, num_blocks.y, num_blocks.z,    // grid dim
                               num_threads.x, num_threads.y, num_threads.z, // block dim
                               0, stream,                                   // shared mem and stream
                               args, 0));                                   // arguments

            // CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());
        }

        void CallCooperative(std::string func, dim3 num_blocks, dim3 num_threads, void *args[], CUstream stream)
        {
            // TODO shared mem, stream
            CUfunction kernel;
            const char *compiled_name = compiled_names[func];
            CUDA_DRIVER_SAFE_CALL(cuModuleGetFunction(&kernel, module, compiled_name));

            CUDA_DRIVER_SAFE_CALL(
                cuLaunchCooperativeKernel(
                    kernel, num_blocks.x, num_blocks.y, num_blocks.z,
                    num_threads.x, num_threads.y, num_threads.z,
                    0, stream, args));
            // CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());
        }

        std::string source;
        std::string include_path;
        std::string prog_name;
        nvrtcProgram program;
        std::vector<std::string> names;
        std::map<std::string, const char *> compiled_names;
        CUmodule module;
    };
}

#endif