class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_last_hidden_state: "bf16[1, 128, 2560]", L_self_modules_lm_head_parameters_weight_: "bf16[248320, 2560]"):
        l_stack0_last_hidden_state = L_stack0_last_hidden_state
        l_self_modules_lm_head_parameters_weight_ = L_self_modules_lm_head_parameters_weight_
        
        # File: /root/real_root/qwen35_fx_capture/capture_qwen35_fx.py:352 in torch_dynamo_resume_in_forward_at_343, code: hidden = self.lm_head(hidden[:, -1:, :])
        getitem: "bf16[1, 1, 2560]" = l_stack0_last_hidden_state[(slice(None, None, None), slice(-1, None, None), slice(None, None, None))];  l_stack0_last_hidden_state = None
        hidden: "bf16[1, 1, 248320]" = torch._C._nn.linear(getitem, l_self_modules_lm_head_parameters_weight_, None);  getitem = l_self_modules_lm_head_parameters_weight_ = None
        return (hidden,)
        