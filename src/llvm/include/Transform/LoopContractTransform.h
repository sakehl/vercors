#ifndef PALLAS_LOOPCONTRACTTRANSFORM_H
#define PALLAS_LOOPCONTRACTTRANSFORM_H

#include "vct/col/ast/col.pb.h"
#include <llvm/Analysis/LoopInfo.h>

/**
 * Implements the transformation of loop-contracts.
 */
namespace llvm2col {
namespace col = vct::col::ast;

void transformLoopContract(llvm::Loop &llvmLoop,
                           col::LoopContract &colContract);

void initializeEmptyLoopContract(col::LoopContract &colContract);

llvm::MDNode *getPallasLoopContract(llvm::MDNode &loopID);

} // namespace llvm2col

#endif // PALLAS_LOOPCONTRACTTRANSFORM_H