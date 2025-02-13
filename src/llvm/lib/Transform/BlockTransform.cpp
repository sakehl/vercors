#include "Transform/BlockTransform.h"

#include "Origin/OriginProvider.h"
#include "Transform/Instruction/BinaryOpTransform.h"
#include "Transform/Instruction/CastOpTransform.h"
#include "Transform/Instruction/FuncletPadOpTransform.h"
#include "Transform/Instruction/MemoryOpTransform.h"
#include "Transform/Instruction/OtherOpTransform.h"
#include "Transform/Instruction/TermOpTransform.h"
#include "Transform/Instruction/UnaryOpTransform.h"
#include "Transform/LoopContractTransform.h"
#include "Util/Exceptions.h"

const std::string SOURCE_LOC = "Transform::BlockTransform";

void llvm2col::transformLLVMBlock(llvm::BasicBlock &llvmBlock,
                                  pallas::FunctionCursor &functionCursor) {
    if (functionCursor.isVisited(llvmBlock)) {
        return;
    }
    col::LlvmBasicBlock &labeled = functionCursor.visitLLVMBlock(llvmBlock);
    if (functionCursor.getLoopInfo().isLoopHeader(&llvmBlock)) {
        llvm::Loop *llvmLoop =
            functionCursor.getLoopInfo().getLoopFor(&llvmBlock);
        col::LlvmLoop *loop = labeled.mutable_loop();
        loop->set_allocated_origin(generateLoopOrigin(*llvmLoop));
        transformLoopContract(*llvmLoop, *loop->mutable_contract(),
                              functionCursor);

        loop->mutable_header()->set_id(labeled.label().id());
        col::LlvmBasicBlock &labeled_latch =
            functionCursor.getOrSetLLVMBlock2ColBlockEntry(
                *llvmLoop->getLoopLatch());
        loop->mutable_latch()->set_id(labeled_latch.label().id());
        for (auto &bb : llvmLoop->blocks()) {
            col::LlvmBasicBlock &labeled_bb =
                functionCursor.getOrSetLLVMBlock2ColBlockEntry(*bb);
            loop->add_block_labels()->set_id(labeled_bb.label().id());
        }
    }
    for (auto &I : llvmBlock) {
        transformInstruction(functionCursor, I, labeled);
    }

    // When the last instuction is a branch, the block already gets completed
    // in the call to transformInstruction.
    if (!functionCursor.isComplete(labeled)) {
        functionCursor.complete(labeled);
    }
}

void llvm2col::transformInstruction(pallas::FunctionCursor &funcCursor,
                                    llvm::Instruction &llvmInstruction,
                                    col::LlvmBasicBlock &colBodyBlock) {
    u_int32_t opCode = llvmInstruction.getOpcode();
    if (llvm::Instruction::TermOpsBegin <= opCode &&
        opCode < llvm::Instruction::TermOpsEnd) {
        llvm2col::transformTermOp(llvmInstruction, colBodyBlock, funcCursor);
    } else if (llvm::Instruction::BinaryOpsBegin <= opCode &&
               opCode < llvm::Instruction::BinaryOpsEnd) {
        llvm2col::transformBinaryOp(llvmInstruction, colBodyBlock, funcCursor);
    } else if (llvm::Instruction::UnaryOpsBegin <= opCode &&
               opCode < llvm::Instruction::UnaryOpsEnd) {
        llvm2col::transformUnaryOp(llvmInstruction, colBodyBlock, funcCursor);
    } else if (llvm::Instruction::MemoryOpsBegin <= opCode &&
               opCode < llvm::Instruction::MemoryOpsEnd) {
        llvm2col::transformMemoryOp(llvmInstruction, colBodyBlock, funcCursor);
    } else if (llvm::Instruction::CastOpsBegin <= opCode &&
               opCode < llvm::Instruction::CastOpsEnd) {
        llvm2col::transformCastOp(llvmInstruction, colBodyBlock, funcCursor);
    } else if (llvm::Instruction::FuncletPadOpsBegin <= opCode &&
               opCode < llvm::Instruction::FuncletPadOpsEnd) {
        llvm2col::transformFuncletPadOp(llvmInstruction, colBodyBlock,
                                        funcCursor);
    } else if (llvm::Instruction::OtherOpsBegin <= opCode &&
               opCode < llvm::Instruction::OtherOpsEnd) {
        llvm2col::transformOtherOp(llvmInstruction, colBodyBlock, funcCursor);
    } else {
        reportUnsupportedOperatorError(SOURCE_LOC, llvmInstruction);
    }
}

void llvm2col::reportUnsupportedOperatorError(
    const std::string &source, llvm::Instruction &llvmInstruction) {
    std::stringstream errorStream;
    errorStream << "Unsupported operator \"" << llvmInstruction.getOpcodeName()
                << '"';
    pallas::ErrorReporter::addError(source, errorStream.str(), llvmInstruction);
}
