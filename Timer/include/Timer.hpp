#ifndef __timer_hpp
#define __timer_hpp

#include <memory>

class Timer {
public:
  Timer();  // constructor. heres how you make the timer.
  void start();  // heres how you start..
  void stop();  // here's how you stop..
  float elapsedTime_ms();  // here's how you get the time it got.
  
  // we could expand this to be a metrics file, where we get effective bandwidth, etc..
  
private:
  // this idion is called "PImpl", pointer to implementation, he calls it "private implementation".
  // headerfiles expose how you interact with something
  // we dont have to use, but learn about it. might be useful.
  class Implementation;
  std::shared_ptr<Implementation> implementation_;
};

#endif


