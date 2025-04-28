#!/bin/bash
make
./prefetch_latency_test
./nvlink_contention_test
./host_migration_parallelism_test
