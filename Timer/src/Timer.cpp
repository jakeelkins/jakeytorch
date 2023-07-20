#include <Timer.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

// we can do it like this since we declared the pimpl Implementation thing in the header file.
class Timer::Implementation {
public:
  Implementation () {
    
    checkCudaErrors (
      cudaEventCreate (
        &start_
      )
    );

    checkCudaErrors (
      cudaEventCreate (
        &stop_
      )
    );

    // this is done so that all future calls to elapsed will succeed
    // else, when you get elased time, itll error out. 
    start();
    stop();

  }
  
  // delete: we wiould have memory leak if we dont. would kill computer/device.

  ~Implementation() {
    
    checkCudaErrors (
      cudaEventDestroy (
        start_
      )
    );
    
    checkCudaErrors (
      cudaEventDestroy (
        stop_
      )
    );
  
  }
  
  // see c++ "rule of five": will help you not have memory leaks.
  // if you need to write a destructor, you probably need to write a copy constructor, copy assignment operator, move
  // constructor, and move assignment operator...
  
  // copy assignment just shows how to copy via an assignment operator. move is all about efficiency, if youre never gonna use that
  // obj again and youre copying it via assignment, just reuse its resources without reallocating. (efficiency)

  // Delete copy constructor and assignment operator
  Implementation (Implementation const &) = delete;
  Implementation & operator= (Implementation const &) = delete;
  // Not technically required due to the previous lines
  Implementation (Implementation &&) = delete; // move constructor
  Implementation & operator= (Implementation &&) = delete;  // move assignment
  // simplest way to assign is to say "you cant do that", so delete.

  void start() {
    checkCudaErrors (
      cudaEventRecord (
        start_
      )
    );  
  }

  void stop() {
    
    checkCudaErrors (
      cudaEventRecord (
        stop_
      )
    );

  }

  float elapsedTime_ms () {
    
    checkCudaErrors (
      cudaEventSynchronize (
        stop_
      )
    );

    float milliseconds = 0.0f;

    checkCudaErrors (
      cudaEventElapsedTime (
        &milliseconds,
        start_,
        stop_
      )
    );

    return milliseconds;

  }
// he deleted the move/copy events bc you dont want to deal with how to transfer state of cuda events, etc
// always use the same event, dont reallocate at each cudaEvent: deal with it using a pointer.
private:
  
  cudaEvent_t start_;
  cudaEvent_t stop_;

};

// init: makes a safe pointer. make_shared: when it goes out of scope, delete memory (shortcut).

Timer::Timer () : 
  implementation_ (
    std::make_shared<Implementation>()
  ) {}
  
// defining all the methods Timer uses.

void Timer::start() {

  implementation_->start();

}

void Timer::stop() {

  implementation_->stop();

}

float Timer::elapsedTime_ms() {

  return implementation_->elapsedTime_ms();

}
