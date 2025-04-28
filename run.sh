#!/bin/bash
make
./prefetch_latency_test > prefetch_latency_test.log 2>&1
./nvlink_contention_test > prefetch_latency_test.log 2>&1
./host_migration_parallelism_test > prefetch_latency_test.log 2>&1
