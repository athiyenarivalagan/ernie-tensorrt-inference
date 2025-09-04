# --- Export to ONNX if requested ---
# if args.export_onnx:
#     print("Exporting optimized model to ONNX...")

#     model_opt.eval()
#     sample_batch = next(iter(val_loader))
#     inputs = {
#         k: (v.to(device).half() if v.dtype == torch.float32 else v.to(device))
#         for k, v in sample_batch.items()
#         if k in ["input_ids", "attention_mask", "token_type_ids"]
#     }

#     torch.onnx.export(
#         model_opt,
#         (inputs,),  # safer dict-style
#         "ernie_opt.onnx",
#         input_names=list(inputs.keys()),
#         output_names=["logits"],
#         dynamic_axes={
#             "input_ids": {0: "batch_size", 1: "seq_len"},
#             "attention_mask": {0: "batch_size", 1: "seq_len"},
#             **({"token_type_ids": {0: "batch_size", 1: "seq_len"}} if "token_type_ids" in inputs else {}),
#             "logits": {0: "batch_size"}
#         },
#         do_constant_folding=True,
#         opset_version=14
#     )
#     print("ONNX export completed successfully!")
