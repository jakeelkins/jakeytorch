# Jakeytorch - My Custom C++/CUDA Parallelized Neural Network Trainer

## -- Building and Running --

After uploading this folder to the DMC, at the root, run:

	module load blas && module load cuda && make run

I also left a "make clean" command like you had in your Makefile Dr. Wise.

In the program output, I currently output the running loss of the last 1000 epochs of training.
Cross entropy typically starts at about 2.3 (-ln(10)), so anything decreasing from that is learning something.
The test is currently set up for the one I reported in my presentation:

100,000 epochs, batch 32, NN (400x300), lr 1e-4.

The program also prints the final batch y_hat and y, just for checking to see what it actually learned (I like
to go through and match them and see the learned probabilities).

## File Structure

- folder/filename (description)

root
 - data (where all data is stored - currently wine and cifar10, in CSV, preprocessed, split by train/test/x/y)
	- cifar*.csv
	- wine*.csv
 - jakeytorch (my main module)
	- include
		- jake_kernels.h  (defines interface to my custom kernels in ../src/jake_kernels.cu)
	- src
		- jake_kernels.cu  (all my custom kernels used in ../test/src/test.cpp)
	- test
		- src
			*_dev.cpp  (these are all dev programs for testing)
			test.cpp  ([!] the primary program, runs the test [!])
 - Timer  (same timer you provided us, I won't detail this one)
 - .runTests.sh  (shell script to run on the DMC)
 - Makefile  (makefile for the program)


As you can see, I left the file structure from the old saxpy tests the same throughout the course.
I was previously scared to death to change the Makefile, but I got the hang of it. I still included
BLAS in the Makefile, though it's not used. THis would likely speed up building, I just never got around
to it. 

You can run any of my reference tests in jakeytorch/test/src/ by renaming test.cpp to something else, and
renaming the test file you want to run as test.cpp (that's how I was developing them). I figured you
would want to see those, so I left them.

Anyways, thanks Dr. Wise! That should be enough to get you to build and run everything. Thank you for
an awesome class and a great semester. Rest assured, I will use these skills for the rest of my career.

Jake
