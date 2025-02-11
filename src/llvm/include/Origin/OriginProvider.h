#ifndef PALLAS_ORIGINPROVIDER_H
#define PALLAS_ORIGINPROVIDER_H

#include "vct/col/ast/Origin.pb.h"
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>

/**
 * Generators for VerCors origin objects for various LLVM Value types.
 *
 * For more info on VerCors origins see:
 * https://github.com/utwente-fmt/vercors/discussions/884
 */
namespace llvm2col {
namespace col = vct::col::ast;

col::Origin *generateLabelledOrigin(const std::string label);

col::Origin *generateProgramOrigin(llvm::Module &llvmModule);

col::Origin *generateFuncDefOrigin(llvm::Function &llvmFunction);

col::Origin *generateFunctionContractOrigin(llvm::Function &llvmFunction,
                                            const std::string &contract);

col::Origin *generateGlobalValOrigin(llvm::Module &llvmModule,
                                     const std::string &globVal);

col::Origin *generateArgumentOrigin(llvm::Argument &llvmArgument);

col::Origin *generateBlockOrigin(llvm::BasicBlock &llvmBlock);

col::Origin *generateLabelOrigin(llvm::BasicBlock &llvmBlock);

col::Origin *generateLoopOrigin(llvm::Loop &llvmLoop);

/**
 * Generates an origin for a pallas function-contract.
 * Assumes that the provided metadata-node is a well-formed encoding of a
 * source-location (adhering to the location-format of pallas).
 */
col::Origin *generatePallasFunctionContractOrigin(const llvm::Function &f,
                                                  const llvm::MDNode &mdSrcLoc);

col::Origin *generateSingleStatementOrigin(llvm::Instruction &llvmInstruction);

col::Origin *generateAssignTargetOrigin(llvm::Instruction &llvmInstruction);

col::Origin *generateBinExprOrigin(llvm::Instruction &llvmInstruction);

col::Origin *generateFunctionCallOrigin(llvm::CallInst &callInstruction);

/**
 * Generates an origin for generated call to a wrapper function of the clause
 * of a pallas function-contract.
 * Assumes that the provided metadata-node is a well-formed encoding of a
 * source-location (adhering to the location-format of pallas).
 */
col::Origin *generatePallasWrapperCallOrigin(const llvm::Function &wrapperFunc,
                                             const llvm::MDNode &clauseSrcLoc);

/**
 * Generates an origin for a clause of a pallas function contract that is
 * attached to the given function. Assumes that the provided metadata-node is a
 * well-formed encoding of a source-location (adhering to the location-format of
 * pallas).
 */
col::Origin *
generatePallasFContractClauseOrigin(const llvm::Function &parentFunc,
                                    const llvm::MDNode &clauseSrcLoc,
                                    unsigned int clauseNum);

col::Origin *generateOperandOrigin(llvm::Instruction &llvmInstruction,
                                   llvm::Value &llvmOperand);

col::Origin *
generateGlobalVariableOrigin(llvm::Module &llvmModule,
                             llvm::GlobalVariable &llvmGlobalVariable);

col::Origin *generateGlobalVariableInitializerOrigin(
    llvm::Module &llvmModule, llvm::GlobalVariable &llvmGlobalVariable,
    llvm::Value &llvmInitializer);

col::Origin *generateVoidOperandOrigin(llvm::Instruction &llvmInstruction);

col::Origin *generateTypeOrigin(llvm::Type &llvmType);

col::Origin *generateMemoryOrderingOrigin(llvm::AtomicOrdering &llvmOrdering);

std::string extractShortPosition(const col::Origin &origin);

col::Origin *deepenOperandOrigin(const col::Origin &origin,
                                 llvm::Value &llvmOperand);

} // namespace llvm2col
#endif // PALLAS_ORIGINPROVIDER_H
