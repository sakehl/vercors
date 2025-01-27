#ifndef PALLAS_CONSTANTS_H
#define PALLAS_CONSTANTS_H
#include <string>
/**
 * Useful string constants to use for searching out metadata nodes
 */
namespace pallas::constants {

// Pallas constants
const std::string PALLAS_FUNC_CONTRACT = "pallas.fcontract";
const std::string PALLAS_REQUIRES = "pallas.requires";
const std::string PALLAS_ENSURES = "pallas.ensures";
const std::string PALLAS_WRAPPER_FUNC = "pallas.exprWrapper";
const std::string PALLAS_SRC_LOC_ID = "pallas.srcLoc";

const std::string PALLAS_SPEC_LIB_MARKER = "pallas.specLib";
const std::string PALLAS_SPEC_RESULT = "pallas.result";
const std::string PALLAS_SPEC_FRAC_OF = "pallas.fracOf";
const std::string PALLAS_SPEC_PERM = "pallas.perm";
const std::string PALLAS_SPEC_IMPLY = "pallas.imply";
const std::string PALLAS_SPEC_STAR = "pallas.sepConj";


// Legacy VCLLVM constants
const std::string VC_PREFIX = "VC.";

const std::string METADATA_PURE_KEYWORD = VC_PREFIX + "pure";
const std::string METADATA_CONTRACT_KEYWORD = VC_PREFIX + "contract";
const std::string METADATA_GLOBAL_KEYWORD = VC_PREFIX + "global";
} // namespace pallas::constants

#endif // PALLAS_CONSTANTS_H
