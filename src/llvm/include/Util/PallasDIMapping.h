#ifndef PALLAS_DIMAPPING_H
#define PALLAS_DIMAPPING_H

#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

/**
 * Utility-functions for mapping debug-info to llvm instructions.
 */
namespace pallas::utils {

llvm::Value *mapDIVarToValue(llvm::Function &f, llvm::DIVariable &diVar);


} // namespace pallas::utils

#endif // PALLAS_DIMAPPING_H
