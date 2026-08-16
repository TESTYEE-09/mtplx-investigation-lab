from __future__ import annotations

from mtplx.kpi.reference_vllm import (
    parse_cuda_api_summary,
    parse_cuda_kernel_summary,
    summarize_vllm_reference,
)


def test_parse_cuda_kernel_summary_counts_instances():
    text = """
 Time (%)  Total Time (ns)  Instances   Avg (ns)  Name
 --------  ---------------  ---------  --------  ----
     70.4       4893770288         27  181250751.4  180718460.0  179067039  197596858  3373234.0  void flash::flash_fwd_kernel
      5.2        361409181        117    3088967.4    1558606.0     533509   15066533  2982564.1  ncclDevKernel_AllReduce
"""

    summary = parse_cuda_kernel_summary(text)

    assert summary["kernel_types"] == 2
    assert summary["total_kernel_instances"] == 144
    assert summary["top_kernels"][0]["instances"] == 27


def test_summarize_vllm_reference_launches_per_token():
    kernel_text = "70.4 4893770288 27 181250751.4 180718460.0 179067039 197596858 3373234.0 flash\n"
    api_text = "90.0 9000 9 1000.0 1000.0 900 1100 1.0 cudaGraphLaunch\n"
    bench_json = '{"summary": {"mean_decode_tok_s": 100.0}, "rows": [{"completion_tokens": 9, "decode_tok_s": 100.0}]}'

    summary = summarize_vllm_reference(
        cuda_kernel_summary_text=kernel_text,
        cuda_api_summary_text=api_text,
        bench_json_text=bench_json,
    )

    assert summary["kernel_launches_per_generated_token"] == 3.0
    assert summary["launch_like_cuda_api_calls_per_generated_token"] == 1.0
    assert summary["bench"]["mean_decode_tok_s"] == 100.0
    assert summary["promotion_target"]["mtplx_kernel_instances_per_token_target"] == 24.0
    assert summary["promotion_target"]["mtplx_command_buffers_per_token_target"] == 8.0


def test_parse_cuda_api_summary_counts_launch_like_calls():
    text = """
 Time (%)  Total Time (ns)  Num Calls   Avg (ns)  Name
 --------  ---------------  ---------  --------  ----
     80.0             9000          9     1000.0      1000.0        900       1100       1.0  cudaGraphLaunch
     20.0             2000          4      500.0       500.0        400        600       1.0  cudaStreamSynchronize
"""

    summary = parse_cuda_api_summary(text)

    assert summary["api_types"] == 2
    assert summary["total_api_calls"] == 13
    assert summary["launch_like_api_calls"] == 9
