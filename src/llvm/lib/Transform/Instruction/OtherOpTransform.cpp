#include "Transform/Instruction/OtherOpTransform.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/FMF.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

#include "Passes/Function/ExprWrapperMapper.h"
#include "Transform/BlockTransform.h"
#include "Transform/Transform.h"
#include "Util/Constants.h"
#include "Util/Exceptions.h"
#include "Util/PallasMD.h"

const std::string SOURCE_LOC = "Transform::Instruction::OtherOp";

void llvm2col::transformOtherOp(llvm::Instruction &llvmInstruction,
                                col::Block &colBlock,
                                pallas::FunctionCursor &funcCursor) {
    switch (llvm::Instruction::OtherOps(llvmInstruction.getOpcode())) {
    case llvm::Instruction::PHI:
        transformPhi(llvm::cast<llvm::PHINode>(llvmInstruction), colBlock,
                     funcCursor);
        break;
    case llvm::Instruction::ICmp:
        transformICmp(llvm::cast<llvm::ICmpInst>(llvmInstruction), colBlock,
                      funcCursor);
        break;
    case llvm::Instruction::FCmp:
        transformFCmp(llvm::cast<llvm::FCmpInst>(llvmInstruction), colBlock,
                      funcCursor);
        break;
    case llvm::Instruction::Call:
        transformCallExpr(llvm::cast<llvm::CallInst>(llvmInstruction), colBlock,
                          funcCursor);
        break;
    default:
        reportUnsupportedOperatorError(SOURCE_LOC, llvmInstruction);
    }
}

void llvm2col::transformPhi(llvm::PHINode &phiInstruction, col::Block &colBlock,
                            pallas::FunctionCursor &funcCursor) {
    col::Variable &varDecl = funcCursor.declareVariable(phiInstruction);
    for (auto &B : phiInstruction.blocks()) {
        // add assignment of the variable to the block of the conditional
        // branch
        col::Block &targetBlock =
            funcCursor.getOrSetLLVMBlock2LabeledColBlockEntry(*B).block;
        // In some cases, the phi-assignments needs to be re-targeted to an
        // empty block.
        col::Block *newTargetBlock =
            funcCursor.getTargetForPhiAssignment(targetBlock, colBlock);

        col::Assign &assignment = funcCursor.createPhiAssignment(
            phiInstruction, *newTargetBlock, varDecl);
        // assign correct value by looking at the value-block pair of phi
        // instruction.
        col::Expr *value = assignment.mutable_value();
        llvm2col::transformAndSetExpr(
            funcCursor, phiInstruction,
            *phiInstruction.getIncomingValueForBlock(B), *value);
    }
}

void llvm2col::transformICmp(llvm::ICmpInst &icmpInstruction,
                             col::Block &colBlock,
                             pallas::FunctionCursor &funcCursor) {
    col::Assign &assignment =
        funcCursor.createAssignmentAndDeclaration(icmpInstruction, colBlock);
    switch (llvm::ICmpInst::Predicate(icmpInstruction.getPredicate())) {
    case llvm::CmpInst::ICMP_EQ: {
        col::Eq &eq = *assignment.mutable_value()->mutable_eq();
        transformCmpExpr(icmpInstruction, eq, funcCursor);
        break;
    }
    case llvm::CmpInst::ICMP_NE: {
        col::Neq &neq = *assignment.mutable_value()->mutable_neq();
        transformCmpExpr(icmpInstruction, neq, funcCursor);
        break;
    }
    case llvm::CmpInst::ICMP_SGT:
    case llvm::CmpInst::ICMP_UGT: {
        col::Greater &gt = *assignment.mutable_value()->mutable_greater();
        transformCmpExpr(icmpInstruction, gt, funcCursor);
        break;
    }
    case llvm::CmpInst::ICMP_SGE:
    case llvm::CmpInst::ICMP_UGE: {
        col::GreaterEq &geq = *assignment.mutable_value()->mutable_greater_eq();
        transformCmpExpr(icmpInstruction, geq, funcCursor);
        break;
    }
    case llvm::CmpInst::ICMP_SLT:
    case llvm::CmpInst::ICMP_ULT: {
        col::Less &lt = *assignment.mutable_value()->mutable_less();
        transformCmpExpr(icmpInstruction, lt, funcCursor);
        break;
    }
    case llvm::CmpInst::ICMP_SLE:
    case llvm::CmpInst::ICMP_ULE: {
        col::LessEq &leq = *assignment.mutable_value()->mutable_less_eq();
        transformCmpExpr(icmpInstruction, leq, funcCursor);
        break;
    }
    default:
        pallas::ErrorReporter::addError(SOURCE_LOC, "Unknown ICMP predicate",
                                        icmpInstruction);
    }
}

void llvm2col::transformFCmp(llvm::FCmpInst &fcmpInstruction,
                             col::Block &colBlock,
                             pallas::FunctionCursor &funcCursor) {
    // TODO: Deal with fastmath flags
    // TODO: Deal with NaNs, LLVM generally pretends signalling NaNs don't
    //       exist so we should probably also only worry about QNaNs but we
    //       don't support NaNs at all right now in VerCors anyway so all this
    //       doesn't matter yet.
    col::Assign &assignment =
        funcCursor.createAssignmentAndDeclaration(fcmpInstruction, colBlock);
    switch (llvm::FCmpInst::Predicate(fcmpInstruction.getPredicate())) {
    // From the documentation:
    //    FCMP_FALSE = 0, ///< 0 0 0 0    Always false (always folded)
    //    FCMP_OEQ = 1,   ///< 0 0 0 1    True if ordered and equal
    //    FCMP_OGT = 2,   ///< 0 0 1 0    True if ordered and greater than
    //    FCMP_OGE = 3,   ///< 0 0 1 1    True if ordered and greater than or
    //    equal FCMP_OLT = 4,   ///< 0 1 0 0    True if ordered and less than
    //    FCMP_OLE = 5,   ///< 0 1 0 1    True if ordered and less than or equal
    //    FCMP_ONE = 6,   ///< 0 1 1 0    True if ordered and operands are
    //    unequal FCMP_ORD = 7,   ///< 0 1 1 1    True if ordered (no nans)
    //    FCMP_UNO = 8,   ///< 1 0 0 0    True if unordered: isnan(X) | isnan(Y)
    //    FCMP_UEQ = 9,   ///< 1 0 0 1    True if unordered or equal
    //    FCMP_UGT = 10,  ///< 1 0 1 0    True if unordered or greater than
    //    FCMP_UGE = 11,  ///< 1 0 1 1    True if unordered, greater than, or
    //    equal FCMP_ULT = 12,  ///< 1 1 0 0    True if unordered or less than
    //    FCMP_ULE = 13,  ///< 1 1 0 1    True if unordered, less than, or equal
    //    FCMP_UNE = 14,  ///< 1 1 1 0    True if unordered or not equal
    //    FCMP_TRUE = 15, ///< 1 1 1 1    Always true (always folded)
    case llvm::CmpInst::FCMP_FALSE: {
        col::BooleanValue &boolean =
            *assignment.mutable_value()->mutable_boolean_value();
        boolean.set_value(false);
        boolean.set_allocated_origin(generateBinExprOrigin(fcmpInstruction));
    }
    case llvm::CmpInst::FCMP_OEQ:
    case llvm::CmpInst::FCMP_UEQ: {
        col::Eq &eq = *assignment.mutable_value()->mutable_eq();
        transformCmpExpr(fcmpInstruction, eq, funcCursor);
        break;
    }
    case llvm::CmpInst::FCMP_OGT:
    case llvm::CmpInst::FCMP_UGT: {
        col::Greater &gt = *assignment.mutable_value()->mutable_greater();
        transformCmpExpr(fcmpInstruction, gt, funcCursor);
        break;
    }
    case llvm::CmpInst::FCMP_OGE:
    case llvm::CmpInst::FCMP_UGE: {
        col::GreaterEq &geq = *assignment.mutable_value()->mutable_greater_eq();
        transformCmpExpr(fcmpInstruction, geq, funcCursor);
        break;
    }
    case llvm::CmpInst::FCMP_OLT:
    case llvm::CmpInst::FCMP_ULT: {
        col::Less &lt = *assignment.mutable_value()->mutable_less();
        transformCmpExpr(fcmpInstruction, lt, funcCursor);
        break;
    }
    case llvm::CmpInst::FCMP_OLE:
    case llvm::CmpInst::FCMP_ULE: {
        col::LessEq &leq = *assignment.mutable_value()->mutable_less_eq();
        transformCmpExpr(fcmpInstruction, leq, funcCursor);
        break;
    }
    case llvm::CmpInst::FCMP_ONE:
    case llvm::CmpInst::FCMP_UNE: {
        col::Neq &neq = *assignment.mutable_value()->mutable_neq();
        transformCmpExpr(fcmpInstruction, neq, funcCursor);
        break;
    }
    case llvm::CmpInst::FCMP_TRUE: {
        col::BooleanValue &boolean =
            *assignment.mutable_value()->mutable_boolean_value();
        boolean.set_value(true);
        boolean.set_allocated_origin(generateBinExprOrigin(fcmpInstruction));
    }
    case llvm::CmpInst::FCMP_ORD:
    case llvm::CmpInst::FCMP_UNO: {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Checking for NaNs is unsupported", fcmpInstruction);
    }
    default:
        pallas::ErrorReporter::addError(SOURCE_LOC, "Unknown FCMP predicate",
                                        fcmpInstruction);
    }
}

void llvm2col::transformCmpExpr(llvm::CmpInst &cmpInstruction,
                                auto &colCompareExpr,
                                pallas::FunctionCursor &funcCursor) {
    transformBinExpr(cmpInstruction, colCompareExpr, funcCursor);
}

bool llvm2col::checkCallSupport(llvm::CallInst &callInstruction) {
    if (callInstruction.isIndirectCall()) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Indirect calls are not supported", callInstruction);
        return false;
    }
    // tail recursion
    if (callInstruction.isMustTailCall()) {
        pallas::ErrorReporter::addError(SOURCE_LOC,
                                        "Tail call optimization not supported",
                                        callInstruction);
        return false;
    }
    // fast math
    if (callInstruction.getFastMathFlags().any()) {
        pallas::ErrorReporter::addError(SOURCE_LOC, "Fast math not supported",
                                        callInstruction);
        return false;
    }
    // return attributes
    for (auto &A : callInstruction.getAttributes().getRetAttrs()) {
        // TODO: Deal with these most of them do not affect the semantics we
        // care about so we could ignore them
        std::stringstream errorStream;
        errorStream << "Return attribute \"" << A.getAsString()
                    << "\" not supported";
        pallas::ErrorReporter::addWarning(SOURCE_LOC, errorStream.str(),
                                          callInstruction);
        return true;
    }
    // address space is platform dependent (unlikely to change semantics)
    // function attributes are just extra compiler information (no semanatic
    // changes)

    // operand bundles
    if (callInstruction.hasOperandBundles()) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Operand bundles not supported", callInstruction);
        return false;
    }

    return true;
}

void llvm2col::transformCallExpr(llvm::CallInst &callInstruction,
                                 col::Block &colBlock,
                                 pallas::FunctionCursor &funcCursor) {
    if (!checkCallSupport(callInstruction) ||
        callInstruction.getCalledFunction() == nullptr)
        return;

    if (callInstruction.getCalledFunction()->isIntrinsic()) {
        // TODO: Deal with intrinsic functions
        return;
    }

    // If it is a call to a function from the pallas specification library,
    // we transform it into the appropriate col-node.
    if (pallas::utils::isPallasSpecLib(*callInstruction.getCalledFunction())) {
        transformPallasSpecLibCall(callInstruction, colBlock, funcCursor);
        return;
    }

    // allocate expression to host the function call in advance
    col::Expr *functionCallExpr;
    // if void function add an eval expression
    if (callInstruction.getType()->isVoidTy()) {
        col::Eval *eval = colBlock.add_statements()->mutable_eval();
        eval->set_allocated_origin(
            llvm2col::generateSingleStatementOrigin(callInstruction));
        functionCallExpr = eval->mutable_expr();
    } else { // else create an assignment
        col::Assign &assignment = funcCursor.createAssignmentAndDeclaration(
            callInstruction, colBlock);
        functionCallExpr = assignment.mutable_value();
    }
    // create actual invocation
    col::LlvmFunctionInvocation *invocation =
        functionCallExpr->mutable_llvm_function_invocation();
    invocation->set_allocated_blame(new col::Blame());
    // set origin
    invocation->set_allocated_origin(
        llvm2col::generateFunctionCallOrigin(callInstruction));
    // set function reference
    invocation->mutable_ref()->set_id(
        funcCursor.getFDResult(*callInstruction.getCalledFunction())
            .getFunctionId());
    // process function arguments
    for (auto &A : callInstruction.args()) {
        llvm2col::transformAndSetExpr(funcCursor, callInstruction, *A,
                                      *invocation->add_args());
    }
}

void llvm2col::transformPallasSpecLibCall(llvm::CallInst &callInstruction,
                                          col::Block &colBlock,
                                          pallas::FunctionCursor &funcCursor) {
    auto specLibType =
        pallas::utils::isPallasSpecLib(*callInstruction.getCalledFunction())
            .value();

    if (specLibType == pallas::constants::PALLAS_SPEC_RESULT) {
        transformPallasSpecResult(callInstruction, colBlock, funcCursor);
    } else if (specLibType == pallas::constants::PALLAS_SPEC_FRAC_OF) {
        transformPallasFracOf(callInstruction, colBlock, funcCursor);
    } else if (specLibType == pallas::constants::PALLAS_SPEC_PERM) {
        transformPallasPerm(callInstruction, colBlock, funcCursor);
    } else {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Unsupported Pallas specification function",
            callInstruction);
    }
}

void llvm2col::transformPallasSpecResult(llvm::CallInst &callInstruction,
                                         col::Block &colBlock,
                                         pallas::FunctionCursor &funcCursor) {
    auto *llvmSpecFunc = callInstruction.getCalledFunction();
    bool isRegularReturn = !llvmSpecFunc->getReturnType()->isVoidTy();

    // Get the function to whose contract this call instuction belongs to.
    auto *wrapperFunc = callInstruction.getFunction();
    auto *llvmParentFunc =
        funcCursor.getFunctionAnalysisManager()
            .getResult<pallas::ExprWrapperMapper>(*wrapperFunc)
            .getParentFunc();
    if (llvmParentFunc == nullptr) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Encountered call to spec-lib that cannot be associated "
            "with a function",
            callInstruction);
        return;
    }
    auto &colParentFunc = funcCursor.getFDResult(*llvmParentFunc);

    if (isRegularReturn) {
        // Case 1: Result is returned as regular return-value
        // %2 = call i32 @pallas.result.0()

        // Check that the function signature is wellformed
        if (!llvmSpecFunc->arg_empty()) {
            pallas::ErrorReporter::addError(
                SOURCE_LOC,
                "Malformed pallas spec-lib result-function. Expected no "
                "arguments.",
                callInstruction);
            return;
        }

        // Build the assignment-expression
        col::Assign &assignment = funcCursor.createAssignmentAndDeclaration(
            callInstruction, colBlock);
        auto *assignExpr = assignment.mutable_value();
        auto *resultNode = assignExpr->mutable_llvm_result();
        resultNode->set_allocated_origin(
            llvm2col::generateFunctionCallOrigin(callInstruction));
        // Set ref to the function to which this contract is attached to
        resultNode->mutable_func()->set_id(colParentFunc.getFunctionId());

    } else {
        // Case 2: Result is returned as a sret parameter
        if (llvmSpecFunc->arg_size() != 1 ||
            !llvmSpecFunc->getArg(0)->hasStructRetAttr()) {
            pallas::ErrorReporter::addError(
                SOURCE_LOC,
                "Malformed pallas spec-lib result-function. Expected one "
                "sret-argument.",
                callInstruction);
            return;
        }

        // Replace the call to the result-function with a store-instruction that
        // stores the value of \result.
        col::LlvmStore *store = colBlock.add_statements()->mutable_llvm_store();
        store->set_allocated_origin(
            llvm2col::generateFunctionCallOrigin(callInstruction));
        store->set_allocated_blame(new col::Blame());
        // Value
        col::LlvmResult *value = store->mutable_value()->mutable_llvm_result();
        value->set_allocated_origin(
            llvm2col::generateFunctionCallOrigin(callInstruction));
        value->mutable_func()->set_id(colParentFunc.getFunctionId());
        // Pointer
        llvm2col::transformAndSetExpr(funcCursor, callInstruction,
                                      *callInstruction.getArgOperand(0),
                                      *store->mutable_pointer());
        // Memory ordering (Set to sequentially consistent)
        col::LlvmMemorySequentiallyConsistent *memOrder =
            store->mutable_ordering()
                ->mutable_llvm_memory_sequentially_consistent();
        memOrder->set_allocated_origin(
            llvm2col::generateFunctionCallOrigin(callInstruction));
    }
}

void llvm2col::transformPallasFracOf(llvm::CallInst &callInstruction,
                                     col::Block &colBlock,
                                     pallas::FunctionCursor &funcCursor) {
    auto *llvmSpecFunc = callInstruction.getCalledFunction();

    //  Check that the signature matches
    if (llvmSpecFunc->arg_size() != 3 ||
        !llvmSpecFunc->getArg(0)->hasStructRetAttr() ||
        !llvmSpecFunc->getArg(1)->getType()->isIntegerTy() ||
        !llvmSpecFunc->getArg(2)->getType()->isIntegerTy()) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Malformed pallas spec-lib fracOf-function.",
            callInstruction);
        return;
    }

    // Check that the value of the sret-argument is an alloca
    auto *sretAlloc =
        dyn_cast_if_present<llvm::AllocaInst>(callInstruction.getArgOperand(0));
    if (sretAlloc == nullptr) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC,
            "Malformed call to fracOf. First argument should be alloca",
            callInstruction);
        return;
    }

    col::LlvmFracOf *fracOf = colBlock.add_statements()->mutable_llvm_frac_of();
    fracOf->set_allocated_origin(
        llvm2col::generateFunctionCallOrigin(callInstruction));
    fracOf->set_allocated_blame(new col::Blame());

    fracOf->mutable_sret()->set_id(
        funcCursor.getVariableMapEntry(*sretAlloc, false).id());
    llvm2col::transformAndSetExpr(funcCursor, callInstruction,
                                  *callInstruction.getArgOperand(1),
                                  *fracOf->mutable_num());
    llvm2col::transformAndSetExpr(funcCursor, callInstruction,
                                  *callInstruction.getArgOperand(2),
                                  *fracOf->mutable_denom());
}

void llvm2col::transformPallasPerm(llvm::CallInst &callInstruction,
                                   col::Block &colBlock,
                                   pallas::FunctionCursor &funcCursor) {
    // Check that the function signature is wellformed
    auto *llvmSpecFunc = callInstruction.getCalledFunction();
    if (llvmSpecFunc->arg_size() != 2 ||
        !llvmSpecFunc->getArg(0)->getType()->isPointerTy() ||
        !llvmSpecFunc->getArg(1)->getType()->isPointerTy() ||
        !llvmSpecFunc->getArg(1)->hasByValAttr()) {
        pallas::ErrorReporter::addError(
            SOURCE_LOC, "Malformed pallas spec-lib definition (Perm).",
            callInstruction);
        return;
    }

    col::Assign &assignment =
        funcCursor.createAssignmentAndDeclaration(callInstruction, colBlock);
    auto *perm = assignment.mutable_value()->mutable_llvm_perm();
    perm->set_allocated_origin(
        llvm2col::generateFunctionCallOrigin(callInstruction));
    perm->set_allocated_blame(new col::Blame());
    perm->mutable_loc()->set_id(
        funcCursor.getVariableMapEntry(*callInstruction.getArgOperand(0), false)
            .id());
    perm->mutable_perm()->set_id(
        funcCursor.getVariableMapEntry(*callInstruction.getArgOperand(1), false)
            .id());
}