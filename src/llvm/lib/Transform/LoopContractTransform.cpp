#include "Transform/LoopContractTransform.h"
#include "Origin/OriginProvider.h"
#include "Passes/Function/FunctionDeclarer.h"
#include "Util/Constants.h"
#include "Util/Exceptions.h"
#include "Util/PallasDIMapping.h"
#include "Util/PallasMD.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <string>

const std::string SOURCE_LOC = "Transform::LoopContractTransform";

void llvm2col::transformLoopContract(llvm::Loop &llvmLoop,
                                     col::LoopContract &colContract,
                                     pallas::FunctionCursor &functionCursor) {
    llvm::MDNode *contractMD = pallas::utils::getPallasLoopContract(llvmLoop);
    if (contractMD == nullptr) {
        initializeEmptyLoopContract(colContract);
        return;
    }

    // Get the source-location from the contract
    llvm::MDNode *contractSrcLoc =
        llvm::dyn_cast<llvm::MDNode>(contractMD->getOperand(1).get());
    if (contractSrcLoc == nullptr ||
        !pallas::utils::isWellformedPallasLocation(contractSrcLoc)) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Malformed loop contract. Expected src-location as second operand.",
            *llvmLoop.getHeader()->getParent());
        return;
    }

    col::LoopInvariant *colInvariant = colContract.mutable_loop_invariant();
    colInvariant->set_allocated_origin(
        generatePallasLoopContractOrigin(llvmLoop, *contractSrcLoc));
    colInvariant->mutable_blame();

    // Check that the loop-contract contains invariants.
    if (contractMD->getNumOperands() < 3) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Malformed loop contract. No invariants were provided.",
            *llvmLoop.getHeader()->getParent());
        return;
    }

    // Extract invariants and add them to the contract
    unsigned int opIdx = 2;
    while (opIdx < contractMD->getNumOperands()) {
        // Cast operand into MDNode
        llvm::MDNode *invMD = llvm::dyn_cast_if_present<llvm::MDNode>(
            contractMD->getOperand(opIdx).get());
        if (invMD == nullptr) {
            pallas::ErrorReporter::addError(
                SOURCE_LOC,
                "Malformed loop-contract. Expected MDNode as operand.",
                *llvmLoop.getHeader()->getParent());
            return;
        }
        // Add invariant to contract
        if (!addInvariantToContract(*invMD, llvmLoop, *colInvariant,
                                    functionCursor)) {
            return;
        }
        ++opIdx;
    }

    // TODO: Ensure that the loop-contract has been fully initialized

    return;
}

bool llvm2col::addInvariantToContract(llvm::MDNode &invMD, llvm::Loop &llvmLoop,
                                      col::LoopInvariant &colContract,
                                      pallas::FunctionCursor &functionCursor) {
    pallas::FunctionAnalysisManager &fam =
        functionCursor.getFunctionAnalysisManager();
    llvm::Function *llvmParentFunc = llvmLoop.getHeader()->getParent();

    // Check wellformedness of MD-Node
    if (invMD.getNumOperands() < 2) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Malformed loop-invariant. Expected at least two operands.",
            *llvmParentFunc);
        return false;
    }

    // Extract src-location
    llvm::MDNode *srcLoc =
        llvm::dyn_cast_if_present<llvm::MDNode>(invMD.getOperand(0).get());
    if (srcLoc == nullptr ||
        !pallas::utils::isWellformedPallasLocation(srcLoc)) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Malformed loop-invariant. Expected src-location as first operand.",
            *llvmParentFunc);
        return false;
    }

    // Get wrapper-function
    auto *llvmWFunc = pallas::utils::getWrapperFromLoopInv(invMD);
    if (llvmWFunc == nullptr) {
        pallas::ErrorReporter::addError(SOURCE_LOC,
                                        "Malformed loop-invariant. Expected "
                                        "wrapper-function as second operand.",
                                        *llvmParentFunc);
        return false;
    }
    col::LlvmFunctionDefinition &colWFunc =
        fam.getResult<pallas::FunctionDeclarer>(*llvmWFunc)
            .getAssociatedColFuncDef();

    // Get DIVariables from MD
    llvm::SmallVector<llvm::DIVariable *, 8> diVars;
    unsigned int idx = 2;
    while (idx < invMD.getNumOperands()) {
        // Check that operand is a DIVariable
        auto *diVar = llvm::dyn_cast_if_present<llvm::DIVariable>(
            invMD.getOperand(idx).get());
        if (diVar == nullptr) {
            pallas::ErrorReporter::addError(
                SOURCE_LOC,
                "Malformed loop invariant. Expected DIVariable as operand.",
                *llvmParentFunc);
            return false;
        }
        diVars.push_back(diVar);
        idx++;
    }

    pallas::FDResult &colFResult =
        fam.getResult<pallas::FunctionDeclarer>(*llvmParentFunc);
    col::LlvmFunctionDefinition &colParentFunc =
        colFResult.getAssociatedColFuncDef();

    // Match DIVariables to COL variables
    llvm::SmallVector<col::Variable *, 8> colVars;
    for (auto *diVar : diVars) {
        llvm::Value *llvmVal =
            pallas::utils::mapDIVarToVar(*llvmParentFunc, *diVar);
        if (llvmVal == nullptr) {
            pallas::ErrorReporter::addError(
                SOURCE_LOC, "Unable to map DIVariable to value.",
                *llvmParentFunc);
            return false;
        }

        // Get the col-variable that corresponds to the llvm-value
        // If value is an argumet, get it from fucntion's arg-map.
        // Otherwise, check the var-map of the function cursor.
        // TODO: Make this work with global variables!
        if (auto *arg = llvm::dyn_cast<llvm::Argument>(llvmVal)) {
            colVars.push_back(&colFResult.getFuncArgMapEntry(*arg));
        } else {
            colVars.push_back(
                &functionCursor.getVariableMapEntry(*llvmVal, false));
            // TODO: Make this fail more gracefully
        }
    }

    // Build call to wrapper-function
    auto *wrapperCall = new col::LlvmFunctionInvocation();
    wrapperCall->set_allocated_origin(
        llvm2col::generatePallasWrapperCallOrigin(*llvmWFunc, *srcLoc));
    wrapperCall->set_allocated_blame(new col::Blame());
    wrapperCall->mutable_ref()->set_id(colWFunc.id());
    // Add arguments to wrapper-call
    for (auto *v : colVars) {
        auto *argExpr = wrapperCall->add_args()->mutable_local();
        // TODO: Make origin more fine-grained
        argExpr->set_allocated_origin(
            llvm2col::generatePallasWrapperCallOrigin(*llvmWFunc, *srcLoc));
        argExpr->mutable_ref()->set_id(v->id());
    }

    // Append wrapper-call to loop-contract
    if (colContract.has_invariant()) {
        auto *oldInv = colContract.release_invariant();
        auto *newInv = colContract.mutable_invariant()->mutable_and_();
        newInv->set_allocated_left(oldInv);
        newInv->mutable_right()->set_allocated_llvm_function_invocation(
            wrapperCall);
    } else {
        colContract.mutable_invariant()->set_allocated_llvm_function_invocation(
            wrapperCall);
    }

    // TODO: Change to true
    return false;
}

void llvm2col::initializeEmptyLoopContract(col::LoopContract &colContract) {
    col::LoopInvariant *invariant = colContract.mutable_loop_invariant();
    col::BooleanValue *tt =
        invariant->mutable_invariant()->mutable_boolean_value();
    tt->set_value(true);
    tt->set_allocated_origin(generateLabelledOrigin("constant true"));
    invariant->set_allocated_origin(generateLabelledOrigin("constant true"));
    invariant->mutable_blame();
}