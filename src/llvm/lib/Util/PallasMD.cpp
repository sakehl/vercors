#include "Util/PallasMD.h"
#include "Util/Constants.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>

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

} // namespace pallas::utils
