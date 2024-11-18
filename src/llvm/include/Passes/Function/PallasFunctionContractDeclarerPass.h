#ifndef PALLAS_PALLASFUNCTIONCONTRACTDECLARERPASS_H
#define PALLAS_PALLASFUNCTIONCONTRACTDECLARERPASS_H

#include "vct/col/ast/col.pb.h"

#include <memory>

#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/PassManager.h>

/**
 * Pass that transforms Pallas function contracts that are defined as
 * metadata that is attached to an LLVM-function into
 * LlvmfunctionContract objects.
 *
 * It assumes that the FunctionContractDeclarer-pass has already been run to
 * create the contract-objects in the buffer.
 *
 * The results can be accessed through FDCResult-objects using a
 * FunctionAnalysisManager.
 *
 */
namespace pallas {
using namespace llvm;
namespace col = vct::col::ast;

class PallasFunctionContractDeclarerPass
    : public AnalysisInfoMixin<PallasFunctionContractDeclarerPass> {
  public:
    /**
     * Retrieves the LlvmfunctionDefinition object in the buffer from the
     * FDCResult object and sets the origin and data-fields of the contract.
     * The value-filed of the contract is left empty, because the
     * ApplicableContract is constructed directly.
     */
    PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);

  private:
    /**
     * Initialized the given ApplicableContract so that it represents a
     * trivial contract (i.e. it only contains a requires-true-clause).
     */
    void initializeTrivialContract(col::ApplicableContract &contract);

    /**
     * Tries to add a clause, that is represented by the given metadata-node, to
     * the given COL-contract.
     * Returns false if an error occurred (e.g. ill-formed metadata-node) and
     * true otherwise. In case of an error, an error is added to the
     * ErrorReporter. ctxFunc is used to build the error messages.
     */
    bool addClauseToContract(col::ApplicableContract &contract, MDNode &clause,
                             FunctionAnalysisManager &fam, Function &ctxFunc);

    /**
     * Sets the pure-flag of the given function based on the given metadata-node
     * that encodes a palas contract.
     * This expects that the second operand of the metadata-node is a boolean
     * constant.
     * Returns false if an error occurred (e.g. ill-formed metadata-node) and
     * true otherwise.
     * The ctxFunc is used to build error messages.
     */
    bool setPurity(col::LlvmFunctionDefinition &colFunc, MDNode &contract,
                   Function &ctxFunc);

    /**
     * Tries to extract the wrapper-function from the given metadata-node that
     * represents a clause of a Pallas contract (i.e. the operand at index 2
     * is expected to point to a function).
     * Also checks, if the function is marked as a wrapper-function.
     * Returns a nullptr id the function could not be extracted.
     * ctxFunc is used to build error messages.
     */
    Function *getWrapperFuncFromClause(MDNode &clause, Function &ctxFunc);
};
} // namespace pallas
#endif // PALLAS_PALLASFUNCTIONCONTRACTDECLARERPASS_H
