# Workload and Topology Aware KV Cache Migration on vLLM

This repository is a research fork of
[vLLM](https://github.com/vllm-project/vllm) for **CS 6222: Systems for
Machine Learning** at Georgia Tech.

The project explores workload- and topology-aware placement of KV cache data.
It uses predicted request reuse times together with memory capacity, transfer
latency, and bandwidth information to decide when and where KV cache data
should be migrated.

## Design

See the
[Notion design document](https://www.notion.so/Workload-and-Topology-Aware-KV-Cache-Migration-on-vLLM-3e189d71d61480bb8233fdd702f1dce1)
for the current architecture and research plan.

## Team

- Ashwin Mudaliar
- Datta Kansal
- Jason Mo
