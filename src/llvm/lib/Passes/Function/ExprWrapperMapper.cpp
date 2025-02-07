#include "Passes/Function/ExprWrapperMapper.h"
#include "Passes/Function/FunctionDeclarer.h"

#include "Origin/OriginProvider.h"
#include "Util/Constants.h"
#include "Util/Exceptions.h"
#include "Util/PallasMD.h"

#include <llvm/IR/Metadata.h>

namespace pallas {
const std::string SOURCE_LOC = "Passes::Function::ExprWrapperMapper";

using namespace llvm;
namespace col = vct::col::ast;

/*
 * EWMResult
 */

EWMResult::EWMResult(llvm::Function *parentFunc,
                     std::optional<PallasWrapperContext> ctx)
    : parentFunc(parentFunc), context(ctx) {}

llvm::Function *EWMResult::getParentFunc() { return parentFunc; }

std::optional<PallasWrapperContext> EWMResult::getContext() { return context; }

/*
 * ExpressionWrapperMapper
 */

AnalysisKey ExprWrapperMapper::Key;

ExprWrapperMapper::Result ExprWrapperMapper::run(Function &F,
                                                 FunctionAnalysisManager &FAM) {

    auto *llvmModule = F.getParent();

    // Check all functions in the current module
    for (Function &parentF : llvmModule->functions()) {
        // Check if the function has a pallas-contract
        if (!utils::hasPallasContract(parentF)) {
            continue;
        }
        auto *contract = parentF.getMetadata(constants::PALLAS_FUNC_CONTRACT);

        // Look at all of the clauses and check if they reference the
        // wrapper-function
        auto numOps = contract->getNumOperands();
        unsigned int clauseIdx = 2;
        for (clauseIdx = 2; clauseIdx < numOps; ++clauseIdx) {
            //  Try to get the third operand as a function
            auto *clause =
                dyn_cast<MDNode>(contract->getOperand(clauseIdx).get());
            if (clause == nullptr || clause->getNumOperands() < 3)
                continue;
            auto *clauseWrapperMD =
                dyn_cast<ValueAsMetadata>(clause->getOperand(2).get());
            if (clauseWrapperMD == nullptr)
                continue;
            auto *clauseWrapper =
                dyn_cast_if_present<Function>(clauseWrapperMD->getValue());
            if (clauseWrapper == nullptr)
                continue;
            // Check if the wrapper-function in the clause is the function that
            // we are looking for.
            if (clauseWrapper == &F) {
                // Determine the context in which the wrapper is used.
                std::optional<PallasWrapperContext> ctx = std::nullopt;
                if (auto *fClauseTMD =
                        dyn_cast<MDString>(clause->getOperand(0).get())) {
                    auto clauseTStr = fClauseTMD->getString().str();
                    if (clauseTStr == pallas::constants::PALLAS_REQUIRES) {
                        ctx = PallasWrapperContext::FuncContractPre;
                    } else if (clauseTStr ==
                               pallas::constants::PALLAS_ENSURES) {
                        ctx = PallasWrapperContext::FuncContractPost;
                    }
                }
                return EWMResult(&parentF, ctx);
            }
        }
    }
    return EWMResult(nullptr, std::nullopt);
}

} // namespace pallas
