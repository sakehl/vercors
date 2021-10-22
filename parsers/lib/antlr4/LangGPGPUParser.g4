parser grammar LangGPGPUParser;

gpgpuBarrier
    : valEmbedContract? GPGPU_BARRIER '(' expression ')'
    ;

gpgpuCudaKernelInvocation
    : clangIdentifier GPGPU_CUDA_OPEN_EXEC_CONFIG expression ',' expression GPGPU_CUDA_CLOSE_EXEC_CONFIG '(' argumentExpressionList ')' valEmbedWithThen?
    | clangIdentifier GPGPU_CUDA_OPEN_EXEC_CONFIG expression ',' expression ',' expression GPGPU_CUDA_CLOSE_EXEC_CONFIG '(' argumentExpressionList ')' valEmbedWithThen?
    ;

gpgpuAtomicBlock
    : GPGPU_ATOMIC compoundStatement valEmbedWithThenBlock?
    ;

gpgpuKernelSpecifier: GPGPU_KERNEL;