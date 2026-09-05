| model                          |       size |     params | backend    | ngl |  n_cpu_moe | n_ubatch | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | -------: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | SYCL       | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 |          pp4096 |        584.98 ± 0.89 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | SYCL       | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 |         pp16384 |       551.11 ± 23.32 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | SYCL       | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 |           tg512 |         46.56 ± 0.10 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | SYCL       | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 |  pp4096 @ d4096 |        577.46 ± 5.23 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | SYCL       | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 | pp16384 @ d4096 |        559.07 ± 1.73 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | SYCL       | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 |   tg512 @ d4096 |         44.72 ± 0.05 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | SYCL       | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 | pp4096 @ d16384 |        560.71 ± 7.06 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | SYCL       | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 | pp16384 @ d16384 |        530.08 ± 8.88 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | SYCL       | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 |  tg512 @ d16384 |         40.27 ± 0.02 |

build: e107984bc (10788)
