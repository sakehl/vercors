#ifndef PALLAS_MD_H
#define PALLAS_MD_H

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <optional>
#include <string>

/**
 * Utils for working with the metadata-node of pallas specifications.
 */
namespace pallas::utils {

/**
 * Checks if the given function is labeled as a function from the pallas
 * specification-library.
 * If so, it returns an optinal that contains the string-identifier of the kind
 * of spec-livb function.
 * If it is not a function from the specification library, an empty optional is
 * returned.
 * @param f The function to check
 */
std::optional<std::string> isPallasSpecLib(const llvm::Function &f);

/**
 * Checks if the given function has a metadata-node that is labeled as a
 * Pallas function contract.
 */
bool hasPallasContract(const llvm::Function &f);

/**
 * Checks if the given function has a metadata-node that is labeled as a
 * VCLLVM contract.
 */
bool hasVcllvmContract(const llvm::Function &f);

/**
 * Checks if the given llvm function is marked as an expression wrapper of a
 * pallas specification.
 */
bool isPallasExprWrapper(const llvm::Function &f);

/**
 * Checks if the given metadata-node is a wellformed encoding of a
 * pallas source-location.
 */
bool isWellformedPallasLocation(const llvm::MDNode *mdNode);

/**
 * Checks if the given metadata-node refers to a integer-constant.
 */
bool isConstantInt(llvm::Metadata *md);

} // namespace pallas::utils

#endif // PALLAS_MD_H
