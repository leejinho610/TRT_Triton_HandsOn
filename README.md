# TRT_Triton Hands_On (2022/04/21 AI Developer Meetup)

## Set up environments
```docker run --gpus '"device=0"' -it --rm -p 8887:8887 -v $(pwd):/hands_on nvcr.io/nvidia/pytorch:22.03-py3```

```cd /hands_on```

```jupyter notebook --ip 0.0.0.0 --port 8887```


## Additional Resources
- [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/)
- [DLI Deploying a Model for Inference at Production Scale](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-03+V1/)
- [TRT Quick Start](https://github.com/NVIDIA/TensorRT/tree/main/quickstart)
- [TRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TF-TRT](https://github.com/tensorflow/tensorrt)
- [Torch-TRT](https://github.com/NVIDIA/Torch-TensorRT)
- [Triton Server](https://github.com/triton-inference-server/server)
- [Triton Client](https://github.com/triton-inference-server/client)
- [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer)

- [GTC On-Demand](https://www.nvidia.com/en-us/on-demand/) - To watch more deep dive sessions
