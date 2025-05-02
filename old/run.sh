#!/bin/bash
./prefetched > prefetch_latency_test.log 2>&1
./contention > prefetch_latency_test.log 2>&1
./hostm > prefetch_latency_test.log 2>&1
