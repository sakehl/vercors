#include "Util/PallasDIMapping.h"
#include "Util/Exceptions.h"

#include <string.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Support/Casting.h>

const std::string SOURCE_LOC = "Util::PallasDIMapping";

namespace pallas::utils {

namespace {

llvm::SmallVector<llvm::DbgVariableIntrinsic *>
getIntrinsicsForDIVar(llvm::Function &f, llvm::DILocalVariable &diVar) {
    llvm::SmallVector<llvm::DbgVariableIntrinsic *> intrinsics;
    for (auto i = inst_begin(&f), end = inst_end(&f); i != end; ++i) {
        auto *asIntr = llvm::dyn_cast<llvm::DbgVariableIntrinsic>(&*i);
        // Check if the intrinsic actually uses the local variable.
        if (asIntr != nullptr && asIntr->getVariable() == &diVar)
            intrinsics.push_back(asIntr);
    }
    return intrinsics;
}

} // namespace

llvm::Value *mapDIVarToValue(llvm::Function &f, llvm::DIVariable &diVar) {
    auto *locDiVar = llvm::dyn_cast<llvm::DILocalVariable>(&diVar);
    // TODO: Also support global variables
    if (locDiVar == nullptr) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Unable to map DIVariable (Global variables unsupported)");
        return nullptr;
    }

    // Check scope of DIVariable
    if ((!llvm::isa<llvm::DILocalScope>(locDiVar->getScope())) ||
        (llvm::cast<llvm::DILocalScope>(locDiVar->getScope())
             ->getSubprogram() != f.getSubprogram())) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Unable to map DIVariable (Incorrect scope)");
        return nullptr;
    }

    // Get the debug-intrinsic that uses the local variable.
    llvm::SmallVector<llvm::DbgVariableIntrinsic *> intrinsics =
        getIntrinsicsForDIVar(f, *locDiVar);
    // Ensure that there is a unique association with an intrinsic
    if (intrinsics.size() != 1) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Unable to map DIVariable (Ambiguous intrinsics)");
        return nullptr;
    }
    llvm::DbgVariableIntrinsic *intr = intrinsics.front();

    // Resolution of DIExpression is not yet supported.
    // TODO: Implement this!
    if (intr->getExpression() != nullptr &&
        intr->getExpression()->getNumElements() != 0) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Unable to map DIVariable (DIExpressions not yet supported)");
        return nullptr;
    }

    if (auto *declIntr = llvm::dyn_cast<llvm::DbgDeclareInst>(intr)) {
        // TODO: Map debug-declare to corresponding llvm-value
        // TODO: Implement
        return declIntr->getAddress();
    } else if (auto *valIntr = llvm::dyn_cast<llvm::DbgValueInst>(intr)) {
        return valIntr->getValue();
    }

    pallas::ErrorReporter::addError(
        SOURCE_LOC, "Unable to map DIVariable (Unsupported dbg-entry)");
    return nullptr;
}

} // namespace pallas::utils
