import torch
import pufferlib.native as nativelib

output = torch.zeros(10, 30).cuda()
nativelib.launch_linear_forward(torch.rand(10, 20).cuda(),
                                torch.rand(30, 20).cuda(),
                                torch.rand(10, 30).cuda(),
                                output
                                )
 
print(f"Output : {output}")
