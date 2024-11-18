#include "Passes/Function/PallasFunctionContractDeclarerPass.h"

#include "Origin/OriginProvider.h"
#include "Passes/Function/FunctionContractDeclarer.h"
#include "Passes/Function/FunctionDeclarer.h"
#include "Util/Constants.h"
#include "Util/Exceptions.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/Support/Casting.h>

namespace pallas {
const std::string SOURCE_LOC =
    "Passes::Function::PallasFunctionContractDeclarerPass";

using namespace llvm;

/*
 * Pallas Function Contract Declarer Pass
 */
PreservedAnalyses
PallasFunctionContractDeclarerPass::run(Function &f,
                                        FunctionAnalysisManager &fam) {
    // Get empty COL LLVM-contract
    FDCResult cResult = fam.getResult<FunctionContractDeclarer>(f);
    auto colLLVMContract =
        cResult.getAssociatedColFuncContract().mutable_pallas_function_contract();

    // Get COL function
    FDResult fResult = fam.getResult<FunctionDeclarer>(f);
    col::LlvmFunctionDefinition &colFunction =
        fResult.getAssociatedColFuncDef();
    // Initialize COL LLVM-contract
    colLLVMContract->set_allocated_blame(new col::Blame());
    // TODO: Set origin

    col::ApplicableContract *colContract = 
        colLLVMContract->mutable_content();
    colContract->set_allocated_blame(new col::Blame());
    // TODO: Set origin

    // Check if a Pallas-contract is attached to the function:
    if (!f.hasMetadata(pallas::constants::PALLAS_FUNC_CONTRACT)) {
        // No contract is present --> add trivial contract
        initializeTrivialContract(*colContract);
        return PreservedAnalyses::all();
    }

    // TODO: Remove once the rest is implemented
    initializeTrivialContract(*colContract);
    return PreservedAnalyses::all();

    /*
    // Check wellformedness of the contract-metadata
    auto *contractNode = f.getMetadata(pallas::constants::PALLAS_FUNC_CONTRACT);
    if (contractNode->getNumOperands() < 3) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Ill-formed contract. Expected at least 3 operands", f);
        return PreservedAnalyses::all();
    }

    // Get src location:
    // TODO: Add to origin

    // Check pure
    auto pureSuccess = setPurity(colFunction, *contractNode, f);
    if (!pureSuccess)
        return PreservedAnalyses::all();

    // Handle contract clauses
    unsigned int clauseIdx = 2;
    while (clauseIdx < contractNode->getNumOperands()) {
        // Check that the metadata-node is wellformed
        auto *clause = contractNode->getOperand(clauseIdx).get();
        if (clause == nullptr || !isa<MDNode>(clause)) {
            pallas::ErrorReporter::addError(
                SOURCE_LOC,
                "Ill-formed contract clause. Expected MDNode as operand.", f);
            return PreservedAnalyses::all();
        }

        auto addClauseSuccess =
            addClauseToContract(*colContract, *cast<MDNode>(clause), fam, f);
        if (!addClauseSuccess)
            return PreservedAnalyses::all();

        ++clauseIdx;
    }

    return PreservedAnalyses::all();
    */
}

bool PallasFunctionContractDeclarerPass::setPurity(
    col::LlvmFunctionDefinition &colFunc, MDNode &contract, Function &ctxFunc) {
    auto *pureMD = dyn_cast<ConstantAsMetadata>(contract.getOperand(1).get());
    auto *pureVal = dyn_cast_if_present<ConstantInt>(pureMD->getValue());
    if (pureVal == nullptr || (!pureVal->getBitWidth() == 1)) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Ill-formed function contract. Second operand should be boolean.",
            ctxFunc);
        return false;
    }
    bool isPure = pureVal->isOne();
    colFunc.set_pure(isPure);
    return true;
}

bool PallasFunctionContractDeclarerPass::addClauseToContract(
    col::ApplicableContract &contract, MDNode &clause,
    FunctionAnalysisManager &fam, Function &ctxFunc) {
    if (clause.getNumOperands() < 3) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Ill-formed contract clause. Expected at least 3 operands.",
            ctxFunc);
        return false;
    }

    // Check clause type (i.e. requires or ensures)
    auto *clauseTypeMD = dyn_cast<MDString>(clause.getOperand(0).get());
    if (clauseTypeMD == nullptr) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Ill-formed contract clause. First operand should be a string.",
            ctxFunc);
        return false;
    }
    auto clauseTypeStr = clauseTypeMD->getString().str();

    // Check source location
    // TODO: Add source location to the origin

    // Get pointer to the LLVM wrapper-function
    auto *wrapperF = getWrapperFuncFromClause(clause, ctxFunc);
    if (wrapperF == nullptr)
        return false;

    // Get COL representation of wrapper function
    auto wrapperFResult = fam.getResult<FunctionDeclarer>(*wrapperF);
    col::LlvmFunctionDefinition &colWrapperF =
        wrapperFResult.getAssociatedColFuncDef();

    // Check wellformed of the variables
    /*
    SmallVector<DIVariable, 8> diVars;
    unsigned int vIdx = 3;
    while (cIdx < clause->getNumOperands()) {
        // Check that operand is a DIVariable
        auto *diVar = dyn_cast<DIVariable>(clause->getOperand(vIdx).get());
        if (diVar == nullptr) {
            pallas::ErrorReporter::addError(
                SOURCE_LOC,
                "Ill-formed contract clause. Expected DIVariable as "
                "operand.",
                f);
            return PreservedAnalyses::all();
        }
        diVars.push_back(diVar);
        cIdx++;
    }
    */

    // Resolve the DIVariables to llvm-variables
    // TODO: implement

    // Resolve the llvm-variables to COL variables
    // TODO: implement

    // Construct an AccountedPredicate from the contract clause
    // TODO: implement

    if (clauseTypeStr == pallas::constants::PALLAS_REQUIRES) {
        // TODO: Add to requires clauses
    } else if (clauseTypeStr == pallas::constants::PALLAS_ENSURES) {
        // TODO: Add to ensures clauses
    } else {
        // Raise error
    }

    // TODO: Change to true
    return false;
}

Function *PallasFunctionContractDeclarerPass::getWrapperFuncFromClause(
    MDNode &clause, Function &ctxFunc) {
    auto *wrapperFuncMD = dyn_cast<ValueAsMetadata>(clause.getOperand(2).get());
    if (wrapperFuncMD == nullptr || !wrapperFuncMD->getType()->isFunctionTy() ||
        wrapperFuncMD->getValue() == nullptr ||
        !isa<Function>(wrapperFuncMD->getValue())) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Ill-formed contract clause. Second operand should be a "
            "pointer to a function.",
            ctxFunc);
        return nullptr;
    }

    // Check that the function is marked as a pallas wrapper function.
    auto *wrapperF = cast<Function>(wrapperFuncMD->getValue());
    if (!wrapperF->hasMetadata(pallas::constants::PALLAS_WRAPPER_FUNC)) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Ill-formed contract clause. Second operand does not point to "
            "wrapper-function.",
            ctxFunc);
        return nullptr;
    }

    return wrapperF;
}

void PallasFunctionContractDeclarerPass::initializeTrivialContract(
    col::ApplicableContract &contract) {

    // Build predicate for the requires-clause
    auto requiresExpr = new col::Expr();
    requiresExpr->mutable_boolean_value()->set_value(true);

    auto *requiresPred =
        contract.mutable_requires_()->mutable_unit_accounted_predicate();
    requiresPred->set_allocated_pred(requiresExpr);
}

} // namespace pallas
