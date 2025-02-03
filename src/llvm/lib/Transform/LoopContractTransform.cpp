#include "Transform/LoopContractTransform.h"
#include "Origin/OriginProvider.h"
#include "Util/Constants.h"
#include "Util/Exceptions.h"
#include "Util/PallasMD.h"

#include <llvm/IR/Metadata.h>
#include <string>

const std::string SOURCE_LOC = "Transform::LoopContractTransform";

void llvm2col::transformLoopContract(llvm::Loop &llvmLoop,
                                     col::LoopContract &colContract) {
    // Extract the LoopID
    llvm::MDNode *loopID = llvmLoop.getLoopID();
    if (loopID == nullptr) {
        initializeEmptyLoopContract(colContract);
        return;
    }

    // Check if the loop has pallas loop-contract
    llvm::MDNode *contractMD = getPallasLoopContract(*loopID);
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

    // Check that the loop-contract contains invariants.
    if (contractMD->getNumOperands() < 3) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Malformed loop contract. No invariants were provided.",
            *llvmLoop.getHeader()->getParent());
        return;
    }

    // Extract invariants and add them to the contract
    unsigned int opIdx = 2;
    while (opIdx < contractMD->getNumOperands()) {
        /*
        auto addClauseSuccess = addClauseToContract(
            *colContract, contractNode->getOperand(opIdx).get(), fam, f,
            opIdx - 1, *mdSrcLoc);
        if (!addClauseSuccess)
            return PreservedAnalyses::all();
        ++opIdx;
        */
    }

    // Match DIVariables to COL variables

    // Construct wrapper-calls & inv-expression
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

llvm::MDNode *llvm2col::getPallasLoopContract(llvm::MDNode &loopID) {
    for (const llvm::MDOperand &op : loopID.operands()) {
        if (auto *opNode = dyn_cast<llvm::MDNode>(op.get())) {
            // Check that the first operand is a MDString identifier for a
            // loop invariant
            if (opNode->getNumOperands() <= 2 ||
                !opNode->getOperand(0).equalsStr(
                    pallas::constants::PALLAS_LOOP_CONTR_ID)) {
                continue;
            }
            return opNode;
        }
    }
    return nullptr;
}
