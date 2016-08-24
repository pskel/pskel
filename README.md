# PSkel

### Introduction

PSkel is an application programming interface (API) that leverages the extensibility of C++ to provide common stencil functionality. Using parallel skeletons, PSkel releases the programmer from the responsibility of writing boiler-plate code for stencil programming (e.g., explicit synchronization and data exchanges between GPU memory and main memory). Furthermore, the framework translates the abstractions described using its API into lowlevel C++ code compatible with Intel TBB, OpenMP and NVIDIA CUDA. PSkel's API is a C++ template library that implements a stencil parallel skeleton and provides useful constructs for developing parallel stencil applications. The API provides templates for manipulating input and output data; specifying stencil masks; encapsulating memory management, computations, and runtime details.

### Requirements

#### Dependencies

* NVIDIA CUDA: PSkel has not been tested with any NVIDIA CUDA version under 5.5.
* OpenMP: OpenMP is used throughout PSkel for efficiency.
* Intel TBB: PSkel can use either OpenMP or TBB for multi-threading the skeletons computations.
* GAlib: GAlib is used to solve optimization problems required for some autotuning mechanisms.
* Google Test: Google testing framework is used for unit-testing PSkel.

### Credits

#### Developers

* Alyson D. Pereira
* Rodrigo C. O. Rocha
* Luiz Ramos
* Luís F. W. Góes
