class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 128, 2560]", arg1_1: "bf16[248320, 2560]"):
        # File: /root/real_root/qwen35_fx_capture/capture_qwen35_fx.py:352 in torch_dynamo_resume_in_forward_at_343, code: hidden = self.lm_head(hidden[:, -1:, :])
        slice_1: "bf16[1, 1, 2560]" = torch.ops.aten.slice.Tensor(arg0_1, 1, -1, 9223372036854775807);  arg0_1 = None
        permute: "bf16[2560, 248320]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view: "bf16[1, 2560]" = torch.ops.aten.view.default(slice_1, [1, 2560]);  slice_1 = None
        mm: "bf16[1, 248320]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[1, 1, 248320]" = torch.ops.aten.view.default(mm, [1, 1, 248320]);  mm = None
        return (view_1,)
        