#include "Util/Constants.h"
#include "Util/PallasMD.h"
#include <llvm/Support/Casting.h>

#include <llvm/IR/Function.h>

namespace pallas::utils {

std::optional<std::string> isPallasSpecLib(const llvm::Function &f) {

    auto *mdMarker = f.getMetadata(constants::PALLAS_SPEC_LIB_MARKER);
    if (mdMarker == nullptr || mdMarker->getNumOperands() != 1)
        return {};

    auto *mdTypeStr =
        llvm::dyn_cast<llvm::MDString>(mdMarker->getOperand(0).get());
    if (mdTypeStr == nullptr)
        return {};

    return mdTypeStr->getString().str();
}

bool hasPallasContract(const llvm::Function &f) {
    return f.hasMetadata(pallas::constants::PALLAS_FUNC_CONTRACT);
}

bool hasVcllvmContract(const llvm::Function &f) {
    return f.hasMetadata(pallas::constants::METADATA_CONTRACT_KEYWORD);
}

bool isPallasExprWrapper(const llvm::Function &f) {
    return f.hasMetadata(pallas::constants::PALLAS_WRAPPER_FUNC);
}

bool isWellformedPallasLocation(
    const llvm::MDNode *mdNode) {

    if (mdNode == nullptr)
        return false;

    if (mdNode->getNumOperands() != 5)
        return false;

    // Check that first operand is a string-identifier
    if (auto *mdStr = dyn_cast<llvm::MDString>(mdNode->getOperand(0).get())) {
        if (mdStr->getString().str() != pallas::constants::PALLAS_SRC_LOC_ID)
            return false;
    } else {
        return false;
    }

    // Check that the last four operands are integer constants
    if (!isConstantInt(mdNode->getOperand(1).get()) ||
        !isConstantInt(mdNode->getOperand(2).get()) ||
        !isConstantInt(mdNode->getOperand(3).get()) ||
        !isConstantInt(mdNode->getOperand(4).get())) {
        return false;
    }

    return true;
}

bool isConstantInt(llvm::Metadata *md) {
    if (auto *mdConst = dyn_cast<llvm::ConstantAsMetadata>(md)) {
        if (isa<llvm::ConstantInt>(mdConst->getValue())) {
            return true;
        }
    }
    return false;
}

} // namespace pallas::utils
