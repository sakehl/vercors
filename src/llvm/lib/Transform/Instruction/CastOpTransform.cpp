#include "Transform/Instruction/CastOpTransform.h"

#include "Transform/BlockTransform.h"
#include "Transform/Transform.h"
#include "Util/Exceptions.h"

const std::string SOURCE_LOC = "Transform::Instruction::CastOp";
void llvm2col::transformCastOp(llvm::Instruction &llvmInstruction,
                               col::Block &colBlock,
                               pallas::FunctionCursor &funcCursor) {
    switch (llvm::Instruction::CastOps(llvmInstruction.getOpcode())) {
    case llvm::Instruction::SExt:
        transformSExt(llvm::cast<llvm::SExtInst>(llvmInstruction), colBlock,
                      funcCursor);
        break;
    case llvm::Instruction::ZExt:
        transformZExt(llvm::cast<llvm::ZExtInst>(llvmInstruction), colBlock,
                      funcCursor);
        break;
    case llvm::Instruction::FPExt:
        transformFPExt(llvm::cast<llvm::FPExtInst>(llvmInstruction), colBlock,
                       funcCursor);
        break;
    case llvm::Instruction::Trunc:
        transformTrunc(llvm::cast<llvm::TruncInst>(llvmInstruction), colBlock,
                       funcCursor);
        break;
    default:
        reportUnsupportedOperatorError(SOURCE_LOC, llvmInstruction);
    }
}

void llvm2col::transformSExt(llvm::SExtInst &sextInstruction,
                             col::Block &colBlock,
                             pallas::FunctionCursor &funcCursor) {
    col::Assign &assignment =
        funcCursor.createAssignmentAndDeclaration(sextInstruction, colBlock);
    col::Expr *sextExpr = assignment.mutable_value();
    col::LlvmSignExtend *sext = sextExpr->mutable_llvm_sign_extend();
    sext->set_allocated_origin(
        llvm2col::generateSingleStatementOrigin(sextInstruction));
    llvm2col::transformAndSetType(*sextInstruction.getSrcTy(),
                                  *sext->mutable_input_type());
    llvm2col::transformAndSetType(*sextInstruction.getDestTy(),
                                  *sext->mutable_output_type());
    llvm2col::transformAndSetExpr(funcCursor, sextInstruction,
                                  *sextInstruction.getOperand(0),
                                  *sext->mutable_value());
}

void llvm2col::transformZExt(llvm::ZExtInst &zextInstruction,
                             col::Block &colBlock,
                             pallas::FunctionCursor &funcCursor) {
    col::Assign &assignment =
        funcCursor.createAssignmentAndDeclaration(zextInstruction, colBlock);
    col::Expr *zextExpr = assignment.mutable_value();
    col::LlvmZeroExtend *zext = zextExpr->mutable_llvm_zero_extend();
    zext->set_allocated_origin(
        llvm2col::generateSingleStatementOrigin(zextInstruction));
    llvm2col::transformAndSetType(*zextInstruction.getSrcTy(),
                                  *zext->mutable_input_type());
    llvm2col::transformAndSetType(*zextInstruction.getDestTy(),
                                  *zext->mutable_output_type());
    llvm2col::transformAndSetExpr(funcCursor, zextInstruction,
                                  *zextInstruction.getOperand(0),
                                  *zext->mutable_value());
}

void llvm2col::transformTrunc(llvm::TruncInst &truncInstruction,
                              col::Block &colBlock,
                              pallas::FunctionCursor &funcCursor) {
    col::Assign &assignment =
        funcCursor.createAssignmentAndDeclaration(truncInstruction, colBlock);
    col::Expr *truncExpr = assignment.mutable_value();
    col::LlvmTruncate *trunc = truncExpr->mutable_llvm_truncate();
    trunc->set_allocated_origin(
        llvm2col::generateSingleStatementOrigin(truncInstruction));
    llvm2col::transformAndSetType(*truncInstruction.getSrcTy(),
                                  *trunc->mutable_input_type());
    llvm2col::transformAndSetType(*truncInstruction.getDestTy(),
                                  *trunc->mutable_output_type());
    llvm2col::transformAndSetExpr(funcCursor, truncInstruction,
                                  *truncInstruction.getOperand(0),
                                  *trunc->mutable_value());
}

void llvm2col::transformFPExt(llvm::FPExtInst &fpextInstruction,
                              col::Block &colBlock,
                              pallas::FunctionCursor &funcCursor) {
    col::Assign &assignment =
        funcCursor.createAssignmentAndDeclaration(fpextInstruction, colBlock);
    col::Expr *fpextExpr = assignment.mutable_value();
    col::LlvmFloatExtend *fpext = fpextExpr->mutable_llvm_float_extend();
    fpext->set_allocated_origin(
        llvm2col::generateSingleStatementOrigin(fpextInstruction));
    llvm2col::transformAndSetType(*fpextInstruction.getSrcTy(),
                                  *fpext->mutable_input_type());
    llvm2col::transformAndSetType(*fpextInstruction.getDestTy(),
                                  *fpext->mutable_output_type());
    llvm2col::transformAndSetExpr(funcCursor, fpextInstruction,
                                  *fpextInstruction.getOperand(0),
                                  *fpext->mutable_value());
}
