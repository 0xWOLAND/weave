# Notes on the Halide 

Halide is an image processing library designed to address the trilemma of 
- Parallelism
- Locality
- Redundant compute mitigation

Halide is equipped with a set of scheduling primitives (`compute_at`, `reorder`, and `tile`) that let the programmer specify "locality-incresing program transformations". But this requires determining the appropriate loop bounds and intermediate buffer sizes based on hardware requirements. The program DAG is equipped with this function bound information, and it helps inform program restructuring optimizations. 

Large Halide programs are partitioned into subprograms, which are are indepently processed for "producer-consumer locality and input reuse locality transformations". The system only considers the set of schedules that tiel the loop nest corresponding to the group's output function. 

- _Compute per-direction input reuse_: for each function, we try to figure out along which axis to iterate to maximize input data reuse
- _Grouping_: restructuring computation ordering to improve producer-consumer locality
    - Initially, functions are grouped individually and then merged if there is a performance benefit