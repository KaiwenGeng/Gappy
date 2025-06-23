
what's the issue here?
06/23/2025 13:42:32.276 [ 1872850 ] ERROR: Unable to finish job 0 due to exception ray::ray_process_job() (pid=8451, ip=10.204.240.62)
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mlresearch/aws/ray_parallelization.py", line 47, in ray_process_job
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/execute/execute_Hydra.py", line 21, in run_Hydra
    best_model, metrics, y_true, y_pred, weight = run_rolling_window(target_cols, sample_weight_cols, predictor_cols, pred_len, training_years, val_year, lookback, args, setting)
                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/exp/run_rolling_window.py", line 91, in run_rolling_window
    best_model, exp_train_loss, exp_vali_loss,exp_train_pearson, exp_vali_pearson, exp_train_spearman, exp_vali_spearman = exp.train(setting)
                                                                                                                           ^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/exp/Forecast.py", line 295, in train
    outputs = self.model(batch_x)
              ^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 72, in forward
    dec_out = self.forecast(x_enc, x_mark_enc)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 63, in forecast
    enc_out  = self.encoder(enc_out)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 43, in forward
    x = layer(x)
        ^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 27, in forward
    x = self.hydra(x)
        ^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/hydra/hydra/modules/hydra.py", line 134, in forward
    return hydra_split_conv1d_scan_combined(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/autograd/function.py", line 598, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/cuda/amp/autocast_mode.py", line 115, in decorate_fwd
    return fwd(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/hydra/hydra/modules/ops.py", line 125, in forward
    scan = _mamba_chunk_scan_combined_fwd(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mamba_ssm/ops/triton/ssd_combined.py", line 313, in _mamba_chunk_scan_combined_fwd
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mamba_ssm/ops/triton/ssd_chunk_state.py", line 746, in _chunk_state_fwd
    _chunk_state_fwd_kernel[grid](
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/jit.py", line 167, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 143, in run
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 143, in <dictcomp>
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 122, in _bench
    return do_bench(kernel_call, warmup=self.warmup, rep=self.rep, quantiles=(0.5, 0.2, 0.8))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/testing.py", line 102, in do_bench
    fn()
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 110, in kernel_call
    self.fn.run(
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/jit.py", line 416, in run
    self.cache[device][key] = compile(
                              ^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/compiler/compiler.py", line 193, in compile
    next_module = compile_ir(module, metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/compiler/backends/cuda.py", line 199, in <lambda>
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/compiler/backends/cuda.py", line 173, in make_llir
    ret = translate_triton_gpu_to_llvmir(src, capability, tma_infos, runtime.TARGET.NVVM)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: map::at!
 Traceback (most recent call last):
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mlresearch/aws/ray_parallelization.py", line 33, in process_single_result
    result = ray.get(receipt_)
             ^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/_private/client_mode_hook.py", line 102, in wrapper
    return getattr(ray, func.__name__)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/util/client/api.py", line 42, in get
    return self.worker.get(vals, timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/util/client/worker.py", line 433, in get
    res = self._get(to_get, op_timeout)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/util/client/worker.py", line 461, in _get
    raise err
ray.exceptions.RayTaskError(IndexError): ray::ray_process_job() (pid=8451, ip=10.204.240.62)
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mlresearch/aws/ray_parallelization.py", line 47, in ray_process_job
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/execute/execute_Hydra.py", line 21, in run_Hydra
    best_model, metrics, y_true, y_pred, weight = run_rolling_window(target_cols, sample_weight_cols, predictor_cols, pred_len, training_years, val_year, lookback, args, setting)
                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/exp/run_rolling_window.py", line 91, in run_rolling_window
    best_model, exp_train_loss, exp_vali_loss,exp_train_pearson, exp_vali_pearson, exp_train_spearman, exp_vali_spearman = exp.train(setting)
                                                                                                                           ^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/exp/Forecast.py", line 295, in train
    outputs = self.model(batch_x)
              ^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 72, in forward
    dec_out = self.forecast(x_enc, x_mark_enc)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 63, in forecast
    enc_out  = self.encoder(enc_out)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 43, in forward
    x = layer(x)
        ^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 27, in forward
    x = self.hydra(x)
        ^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/hydra/hydra/modules/hydra.py", line 134, in forward
    return hydra_split_conv1d_scan_combined(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/autograd/function.py", line 598, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/cuda/amp/autocast_mode.py", line 115, in decorate_fwd
    return fwd(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/hydra/hydra/modules/ops.py", line 125, in forward
    scan = _mamba_chunk_scan_combined_fwd(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mamba_ssm/ops/triton/ssd_combined.py", line 313, in _mamba_chunk_scan_combined_fwd
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mamba_ssm/ops/triton/ssd_chunk_state.py", line 746, in _chunk_state_fwd
    _chunk_state_fwd_kernel[grid](
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/jit.py", line 167, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 143, in run
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 143, in <dictcomp>
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 122, in _bench
    return do_bench(kernel_call, warmup=self.warmup, rep=self.rep, quantiles=(0.5, 0.2, 0.8))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/testing.py", line 102, in do_bench
    fn()
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 110, in kernel_call
    self.fn.run(
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/jit.py", line 416, in run
    self.cache[device][key] = compile(
                              ^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/compiler/compiler.py", line 193, in compile
    next_module = compile_ir(module, metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/compiler/backends/cuda.py", line 199, in <lambda>
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/compiler/backends/cuda.py", line 173, in make_llir
    ret = translate_triton_gpu_to_llvmir(src, capability, tma_infos, runtime.TARGET.NVVM)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: map::at
Traceback (most recent call last):
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mlresearch/aws/ray_parallelization.py", line 33, in process_single_result
    result = ray.get(receipt_)
             ^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/_private/client_mode_hook.py", line 102, in wrapper
    return getattr(ray, func.__name__)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/util/client/api.py", line 42, in get
    return self.worker.get(vals, timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/util/client/worker.py", line 433, in get
    res = self._get(to_get, op_timeout)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/ray/util/client/worker.py", line 461, in _get
    raise err
ray.exceptions.RayTaskError(IndexError): ray::ray_process_job() (pid=8451, ip=10.204.240.62)
  File "/mnt/netapp_hfalgo/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mlresearch/aws/ray_parallelization.py", line 47, in ray_process_job
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/execute/execute_Hydra.py", line 21, in run_Hydra
    best_model, metrics, y_true, y_pred, weight = run_rolling_window(target_cols, sample_weight_cols, predictor_cols, pred_len, training_years, val_year, lookback, args, setting)
                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/exp/run_rolling_window.py", line 91, in run_rolling_window
    best_model, exp_train_loss, exp_vali_loss,exp_train_pearson, exp_vali_pearson, exp_train_spearman, exp_vali_spearman = exp.train(setting)
                                                                                                                           ^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/exp/Forecast.py", line 295, in train
    outputs = self.model(batch_x)
              ^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 72, in forward
    dec_out = self.forecast(x_enc, x_mark_enc)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 63, in forecast
    enc_out  = self.encoder(enc_out)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 43, in forward
    x = layer(x)
        ^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/Hydra.py", line 27, in forward
    x = self.hydra(x)
        ^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/hydra/hydra/modules/hydra.py", line 134, in forward
    return hydra_split_conv1d_scan_combined(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/autograd/function.py", line 598, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/torch/cuda/amp/autocast_mode.py", line 115, in decorate_fwd
    return fwd(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ua_1872850/seq2seq_alpha/seq2seq_alpha/models/hydra/hydra/modules/ops.py", line 125, in forward
    scan = _mamba_chunk_scan_combined_fwd(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mamba_ssm/ops/triton/ssd_combined.py", line 313, in _mamba_chunk_scan_combined_fwd
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/mamba_ssm/ops/triton/ssd_chunk_state.py", line 746, in _chunk_state_fwd
    _chunk_state_fwd_kernel[grid](
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/jit.py", line 167, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 143, in run
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 143, in <dictcomp>
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 122, in _bench
    return do_bench(kernel_call, warmup=self.warmup, rep=self.rep, quantiles=(0.5, 0.2, 0.8))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/testing.py", line 102, in do_bench
    fn()
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/autotuner.py", line 110, in kernel_call
    self.fn.run(
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/runtime/jit.py", line 416, in run
    self.cache[device][key] = compile(
                              ^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/compiler/compiler.py", line 193, in compile
    next_module = compile_ir(module, metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/compiler/backends/cuda.py", line 199, in <lambda>
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/apps/hfalgo_ext/7d42eb22386/python/Python-3.11.11/lib/python3.11/site-packages/triton/compiler/backends/cuda.py", line 173, in make_llir
    ret = translate_triton_gpu_to_llvmir(src, capability, tma_infos, runtime.TARGET.NVVM)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: map::at
06/23/2025 13:42:33.479 [ 1872850 ] INFO: S3FileRetriever: finished 0 of 0 items submitted (0%).

